from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ccg_tui.backends.base import BackendAdapter
from ccg_tui.handoff import DelegatedContextPacket, DelegatedResultPayload, HandoffPacket
from ccg_tui.models import (
    BackendName,
    BackendSessionRecord,
    EventType,
    RecordedEvent,
    RoutingDecisionRecord,
    SessionLineageRecord,
    SessionRecord,
    SummaryRecord,
    TaskRecord,
    TurnRecord,
    TurnStatus,
    NormalizedError,
)
from ccg_tui.resume_context import ResumeContextConfig, ResumeContextPayload, build_resume_context_payload
from ccg_tui.summary import generate_and_persist_summary
from ccg_tui.transcript import TranscriptStore


class SessionController:
    def __init__(self, adapter: BackendAdapter, store: TranscriptStore, cwd: Path) -> None:
        self.adapter = adapter
        self.store = store
        self.cwd = Path(cwd)
        self.active_turn: TurnRecord | None = None
        self.resume_context_config = ResumeContextConfig(enabled=False)
        self._pending_resume_context = False
        now = self._now()
        self.session = SessionRecord(
            id=f"session-{uuid4().hex[:12]}",
            backend=adapter.name,
            created_at=now,
            updated_at=now,
            workspace_cwd=str(self.cwd),
            backend_sessions=[
                BackendSessionRecord(
                    id=f"backend-{adapter.name.value}-primary",
                    backend=adapter.name,
                    created_at=now,
                    last_seen_at=now,
                )
            ],
        )
        self.store.save_session(self.session)

    @classmethod
    def resume(
        cls,
        adapter: BackendAdapter,
        store: TranscriptStore,
        cwd: Path,
        session: SessionRecord,
        resume_context_config: ResumeContextConfig | None = None,
    ) -> "SessionController":
        if session.turns and adapter.name != session.backend:
            raise ValueError(
                "One backend per session is enforced for local resume; "
                f"session {session.id} uses {session.backend.value}, requested {adapter.name.value}"
            )
        controller = cls.__new__(cls)
        controller.adapter = adapter
        controller.store = store
        controller.cwd = Path(cwd)
        controller.active_turn = None
        controller.session = session
        controller.resume_context_config = resume_context_config or ResumeContextConfig()
        controller._pending_resume_context = controller.resume_context_config.enabled
        controller._finalize_stale_incomplete_turns()
        controller._prepare_resumed_backend_session()
        controller.store.save_session(controller.session)
        return controller

    def attach_backend(self, adapter: BackendAdapter) -> None:
        if self.session.turns and adapter.name != self.session.backend:
            raise ValueError("One backend per session is enforced for Phase 1")
        if adapter is not self.adapter:
            close = getattr(self.adapter, "close", None)
            if callable(close):
                close()
        self.adapter = adapter
        self.session.backend = adapter.name
        now = self._now()
        self.session.updated_at = now
        if self.session.backend_sessions:
            active_session = self.session.backend_sessions[-1]
            if active_session.backend != adapter.name:
                self.session.backend_sessions.append(
                    BackendSessionRecord(
                        id=f"backend-{adapter.name.value}-{len(self.session.backend_sessions) + 1}",
                        backend=adapter.name,
                        created_at=now,
                        last_seen_at=now,
                    )
                )
        else:
            self.session.backend_sessions.append(
                BackendSessionRecord(
                    id=f"backend-{adapter.name.value}-primary",
                    backend=adapter.name,
                    created_at=now,
                    last_seen_at=now,
                )
            )
        self.store.save_session(self.session)

    def submit_prompt(
        self,
        prompt: str,
        on_update: Callable[[TurnRecord], None] | None = None,
        *,
        backend_prompt: str | None = None,
        metadata: dict | None = None,
    ) -> TurnRecord:
        task = self.prompt_task()
        submitted_prompt = backend_prompt if backend_prompt is not None else prompt
        turn_metadata: dict = dict(metadata or {})
        self._annotate_injected_prompt_metadata(
            turn_metadata,
            visible_prompt=prompt,
            submitted_prompt=submitted_prompt,
            backend_prompt=backend_prompt,
        )
        resume_context_payload = self._pending_resume_context_payload(prompt)
        if resume_context_payload is not None:
            submitted_prompt = resume_context_payload.backend_prompt
            turn_metadata["resume_context"] = resume_context_payload.metadata
        turn = TurnRecord(
            id=f"turn-{uuid4().hex[:12]}",
            backend=self.session.backend,
            prompt=prompt,
            output="",
            status=TurnStatus.SUBMITTING,
            started_at=self._now(),
            task_id=task.id,
            metadata=turn_metadata,
        )
        self.active_turn = turn
        self.session.turns.append(turn)
        self._record_turn_on_task(task, turn.id, turn.started_at)
        self.session.updated_at = turn.started_at
        self._update_turn_recovery_metadata(turn, state="incomplete", terminal_event_seen=False)
        self.store.save_session(self.session)
        terminal_event_seen = False
        try:
            for event in self.adapter.run(submitted_prompt, self.cwd):
                observed_at = self._now()
                turn.events.append(
                    RecordedEvent(
                        type=event.type.value,
                        observed_at=observed_at,
                        text=event.text,
                        session_id=event.session_id,
                        activity=event.activity,
                        error=event.error,
                        raw=event.raw,
                    )
                )
                if event.type is EventType.SESSION_STARTED and event.session_id:
                    self.session.vendor_session_id = event.session_id
                    turn.vendor_session_id = event.session_id
                    self._mark_backend_session(event.session_id, observed_at)
                elif event.type is EventType.ACTIVITY:
                    if turn.status is TurnStatus.SUBMITTING:
                        turn.status = TurnStatus.STREAMING
                    self._touch_backend_session(observed_at)
                elif event.type is EventType.OUTPUT_STARTED:
                    turn.status = TurnStatus.STREAMING
                elif event.type is EventType.OUTPUT_DELTA:
                    turn.output += event.text
                    if turn.status is TurnStatus.SUBMITTING:
                        turn.status = TurnStatus.STREAMING
                elif event.type is EventType.BACKEND_SUCCEEDED:
                    turn.status = TurnStatus.COMPLETED
                    turn.completed_at = observed_at
                    self._touch_backend_session(observed_at)
                    self._update_turn_recovery_metadata(turn, state="completed", terminal_event_seen=True)
                    terminal_event_seen = True
                elif event.type is EventType.BACKEND_FAILED:
                    turn.status = TurnStatus.FAILED
                    turn.error = event.error or NormalizedError(
                        kind="backend_error",
                        message="Backend reported failure without error details",
                    )
                    turn.events[-1].error = turn.error
                    turn.completed_at = observed_at
                    self._touch_backend_session(observed_at, status="failed")
                    recovery_state = "interrupted" if turn.error.kind == "interrupted" else "failed"
                    self._update_turn_recovery_metadata(turn, state=recovery_state, terminal_event_seen=True)
                    terminal_event_seen = True
                if event.type not in {EventType.BACKEND_SUCCEEDED, EventType.BACKEND_FAILED}:
                    self._update_turn_recovery_metadata(turn, state="incomplete", terminal_event_seen=False)
                self.session.updated_at = observed_at
                self.store.save_session(self.session)
                if on_update is not None:
                    on_update(turn)
        except Exception as exc:
            observed_at = self._now()
            turn.status = TurnStatus.FAILED
            turn.completed_at = observed_at
            turn.error = NormalizedError(
                kind="adapter_exception",
                message=str(exc) or exc.__class__.__name__,
                details={"exception_type": exc.__class__.__name__},
            )
            turn.events.append(
                RecordedEvent(
                    type=EventType.BACKEND_FAILED.value,
                    observed_at=observed_at,
                    error=turn.error,
                    raw={"exception_type": exc.__class__.__name__},
                )
            )
            self._touch_backend_session(observed_at, status="failed")
            self._update_turn_recovery_metadata(turn, state="failed", terminal_event_seen=False)
            self.session.updated_at = observed_at
            self.store.save_session(self.session)
            self.active_turn = None
            if on_update is not None:
                on_update(turn)
            raise
        if turn.vendor_session_id is None:
            turn.vendor_session_id = self.session.vendor_session_id
        if turn.completed_at is None:
            turn.completed_at = self._now()
        if not terminal_event_seen:
            turn.status = TurnStatus.FAILED
            if turn.error is None:
                turn.error = NormalizedError(
                    kind="interrupted",
                    message="Turn ended before completion was confirmed. Output may be partial; resume from the latest user prompt and inspect the partial assistant output manually.",
                )
            self._touch_backend_session(turn.completed_at, status="interrupted")
            self._update_turn_recovery_metadata(turn, state="interrupted", terminal_event_seen=False)
        if self._pending_resume_context:
            self._pending_resume_context = False
        self.session.updated_at = turn.completed_at or self._now()
        self.store.save_session(self.session)
        self.active_turn = None
        if on_update is not None:
            on_update(turn)
        return turn

    def close(self) -> None:
        self._finalize_active_turn_on_close()
        close = getattr(self.adapter, "close", None)
        if callable(close):
            close()

    def generate_summary(
        self,
        summary_adapter: BackendAdapter,
        *,
        scope: str = "task",
        task_id: str | None = None,
    ) -> SummaryRecord:
        if self.active_turn is not None:
            raise ValueError("Cannot summarize while a turn is active")
        resolved_task_id = task_id or self.prompt_task().id
        summary = generate_and_persist_summary(
            self.session,
            adapter=summary_adapter,
            cwd=self.cwd,
            save_session=self.store.save_session,
            scope=scope,
            task_id=resolved_task_id,
        )
        if scope == "task":
            task = self.task_by_id(resolved_task_id)
            if task is not None:
                task.summary_id = summary.id
                task.updated_at = summary.created_at
                self.session.updated_at = summary.created_at
                self.store.save_session(self.session)
        return summary

    @classmethod
    def from_handoff(
        cls,
        adapter: BackendAdapter,
        store: TranscriptStore,
        cwd: Path,
        source_session: SessionRecord,
        handoff_packet: HandoffPacket | None = None,
    ) -> "SessionController":
        controller = cls(adapter=adapter, store=store, cwd=cwd)
        source_turn_ids = (
            handoff_packet.metadata.get("source_turn_ids", [])
            if handoff_packet is not None
            else []
        )
        forked_from_turn_id = (
            str(source_turn_ids[-1])
            if source_turn_ids
            else (source_session.turns[-1].id if source_session.turns else None)
        )
        controller.session.lineage = SessionLineageRecord.for_handoff(
            source_session.id,
            forked_from_turn_id=forked_from_turn_id,
            relationship_metadata=(
                cls._handoff_lineage_metadata(handoff_packet) if handoff_packet is not None else None
            ),
        )
        controller.session.updated_at = controller._now()
        controller.store.save_session(controller.session)
        return controller

    @classmethod
    def from_delegated(
        cls,
        adapter: BackendAdapter,
        store: TranscriptStore,
        cwd: Path,
        source_session: SessionRecord,
        context_packet: DelegatedContextPacket | None = None,
    ) -> "SessionController":
        controller = cls(adapter=adapter, store=store, cwd=cwd)
        forked_from_turn_id = (
            str(context_packet.metadata.get("forked_from_turn_id"))
            if context_packet is not None and context_packet.metadata.get("forked_from_turn_id")
            else (source_session.turns[-1].id if source_session.turns else None)
        )
        controller.session.lineage = SessionLineageRecord.for_delegated(
            source_session.id,
            forked_from_turn_id=forked_from_turn_id,
        )
        controller.session.updated_at = controller._now()
        controller.store.save_session(controller.session)
        return controller

    @property
    def resume_context_pending(self) -> bool:
        return self._pending_resume_context and self.resume_context_config.enabled

    def preview_resume_context(self, user_prompt: str = "<next user prompt>") -> ResumeContextPayload | None:
        if not self.resume_context_pending:
            return None
        return build_resume_context_payload(
            self.session,
            user_prompt=user_prompt,
            config=self.resume_context_config,
        )

    def main_task(self) -> TaskRecord:
        return self.session.tasks[0]

    def task_by_id(self, task_id: str) -> TaskRecord | None:
        return next((task for task in self.session.tasks if task.id == task_id), None)

    def active_user_task(self) -> TaskRecord | None:
        return next(
            (task for task in reversed(self.session.tasks) if task.id != "task-main" and task.status == "active"),
            None,
        )

    def prompt_task(self) -> TaskRecord:
        return self.active_user_task() or self.main_task()

    def latest_closed_task(self) -> TaskRecord | None:
        return next((task for task in reversed(self.session.tasks) if task.status == "closed"), None)

    def start_task(self, title: str | None = None) -> TaskRecord:
        active_task = self.active_user_task()
        if active_task is not None:
            raise ValueError(f"Task already active: {active_task.id}")
        now = self._now()
        task = TaskRecord(
            id=f"task-{uuid4().hex[:8]}",
            created_at=now,
            updated_at=now,
            kind="task",
            title=title or None,
            status="active",
        )
        self.session.tasks.append(task)
        self.session.updated_at = now
        self.store.save_session(self.session)
        return task

    def close_task(self, closing_note: str | None = None) -> TaskRecord:
        task = self.active_user_task()
        if task is None:
            raise ValueError("No active task to close")
        now = self._now()
        task.status = "closed"
        task.updated_at = now
        task.end_turn_id = task.turn_ids[-1] if task.turn_ids else None
        task.closing_note = closing_note or None
        self.session.updated_at = now
        self.store.save_session(self.session)
        return task

    def promote_delegated_result(
        self,
        delegated_result: DelegatedResultPayload,
        *,
        scope: str | None = None,
        task_id: str | None = None,
    ) -> SummaryRecord:
        payload_metadata = dict(delegated_result.metadata)
        if payload_metadata.get("mode") != "delegated_result":
            raise ValueError("delegated result payload metadata is invalid")
        if payload_metadata.get("parent_session_id") != self.session.id:
            raise ValueError("delegated result payload does not belong to this parent session")

        selection_criteria = payload_metadata.get("selection_criteria", {})
        if not isinstance(selection_criteria, dict):
            selection_criteria = {}
        source_scope = str(selection_criteria.get("scope") or "session")
        source_task_id = selection_criteria.get("task_id")
        if not isinstance(source_task_id, str) or not source_task_id:
            source_task_id = source_scope.removeprefix("task:") if source_scope.startswith("task:") else None

        resolved_scope = scope
        resolved_task_id = task_id
        if resolved_scope is None:
            if source_scope.startswith("task:"):
                resolved_scope = "task"
                resolved_task_id = resolved_task_id or source_task_id
            else:
                resolved_scope = "session"
        if resolved_scope not in {"task", "session"}:
            raise ValueError("delegated result promotion scope must be 'task' or 'session'")

        target_task: TaskRecord | None = None
        if resolved_scope == "task":
            resolved_task_id = resolved_task_id or source_task_id or self.prompt_task().id
            target_task = self.task_by_id(resolved_task_id)
            if target_task is None:
                raise ValueError(f"unknown task id: {resolved_task_id}")
        summary_scope = f"task:{resolved_task_id}" if resolved_scope == "task" else "session"
        now = self._now()
        summary = SummaryRecord(
            id=f"summary-{uuid4().hex[:12]}",
            scope=summary_scope,
            created_at=now,
            text=delegated_result.result_text,
            kind="delegated_result",
            metadata={
                **payload_metadata,
                "promotion_scope": summary_scope,
                "promoted_to_session_id": self.session.id,
            },
        )
        self.session.summaries.append(summary)
        if target_task is not None:
            target_task.summary_id = summary.id
            target_task.updated_at = now
        self.session.updated_at = now
        self.store.save_session(self.session)
        return summary

    def record_routing_decision(
        self,
        *,
        trigger: str,
        user_decision: str,
        final_action: str,
        active_backend: BackendName | str | None = None,
        suggested_backend: BackendName | str | None = None,
        suggested_model: str | None = None,
        policy_reference: str = "README.md",
        permission_state: dict | None = None,
        reason: str = "",
        compatibility: dict | None = None,
        metadata: dict | None = None,
    ) -> RoutingDecisionRecord:
        now = self._now()
        decision = RoutingDecisionRecord(
            id=f"routing-{uuid4().hex[:12]}",
            recorded_at=now,
            active_backend=active_backend or self.session.backend,
            suggested_backend=suggested_backend,
            suggested_model=suggested_model,
            trigger=trigger,
            policy_reference=policy_reference,
            permission_state=dict(permission_state or {}),
            user_decision=user_decision,
            final_action=final_action,
            reason=reason,
            compatibility=dict(compatibility or {}),
            metadata=dict(metadata or {}),
        )
        self.session.routing_decisions.append(decision)
        self.session.updated_at = now
        self.store.save_session(self.session)
        return decision

    def _record_turn_on_task(self, task: TaskRecord, turn_id: str, timestamp: str) -> None:
        task.turn_ids.append(turn_id)
        task.updated_at = timestamp
        if task.start_turn_id is None:
            task.start_turn_id = turn_id

    def _pending_resume_context_payload(self, prompt: str) -> ResumeContextPayload | None:
        if not self.resume_context_pending:
            return None
        return build_resume_context_payload(
            self.session,
            user_prompt=prompt,
            config=self.resume_context_config,
        )

    def _finalize_stale_incomplete_turns(self) -> None:
        stale_statuses = {TurnStatus.SUBMITTING, TurnStatus.STREAMING}
        stale_turns = [turn for turn in self.session.turns if turn.status in stale_statuses]
        if not stale_turns:
            return
        observed_at = self._now()
        for turn in stale_turns:
            if turn.completed_at is None:
                turn.completed_at = observed_at
            if turn.error is None:
                turn.error = NormalizedError(
                    kind="interrupted",
                    message=(
                        "Turn was still in progress when the local session was resumed. "
                        "Output may be partial; continue from the latest user prompt and inspect recorded output manually."
                    ),
                )
            turn.status = TurnStatus.FAILED
            turn.events.append(
                RecordedEvent(
                    type=EventType.BACKEND_FAILED.value,
                    observed_at=observed_at,
                    error=turn.error,
                    raw={"reason": "resume_reconciliation"},
                )
            )
            self._update_turn_recovery_metadata(turn, state="interrupted", terminal_event_seen=False)
            recovery = turn.metadata.get("recovery")
            if isinstance(recovery, dict):
                recovery["reconciled_on_resume"] = True
                recovery["reconciled_at"] = observed_at
        if self.session.backend_sessions:
            backend_session = self.session.backend_sessions[-1]
            backend_session.status = "interrupted"
            backend_session.last_seen_at = observed_at
        self.session.updated_at = observed_at

    def _finalize_active_turn_on_close(self) -> None:
        turn = self.active_turn
        if turn is None or turn.status not in {TurnStatus.SUBMITTING, TurnStatus.STREAMING}:
            return
        observed_at = self._now()
        if turn.completed_at is None:
            turn.completed_at = observed_at
        if turn.error is None:
            turn.error = NormalizedError(
                kind="interrupted",
                message=(
                    "Turn was interrupted while the local session was closing. "
                    "Output may be partial; inspect the stored transcript before resuming or handing off."
                ),
            )
        turn.status = TurnStatus.FAILED
        if not turn.events or turn.events[-1].type != EventType.BACKEND_FAILED.value:
            turn.events.append(
                RecordedEvent(
                    type=EventType.BACKEND_FAILED.value,
                    observed_at=observed_at,
                    error=turn.error,
                    raw={"reason": "session_close"},
                )
            )
        if self.session.backend_sessions:
            self._touch_backend_session(observed_at, status="interrupted")
        self._update_turn_recovery_metadata(turn, state="interrupted", terminal_event_seen=False)
        recovery = turn.metadata.get("recovery")
        if isinstance(recovery, dict):
            recovery["interrupted_on_close"] = True
            recovery["interrupted_at"] = observed_at
        self.session.updated_at = observed_at
        self.store.save_session(self.session)
        self.active_turn = None

    def _prepare_resumed_backend_session(self) -> None:
        if not self.session.workspace_cwd:
            self.session.workspace_cwd = str(self.cwd)
        if self.adapter.name != self.session.backend:
            self.session.backend = self.adapter.name
        self.session.vendor_session_id = None
        now = self._now()
        self.session.backend_sessions.append(
            BackendSessionRecord(
                id=f"backend-{self.adapter.name.value}-{len(self.session.backend_sessions) + 1}",
                backend=self.adapter.name,
                created_at=now,
                last_seen_at=now,
            )
        )

    def _mark_backend_session(self, vendor_session_id: str, observed_at: str) -> None:
        backend_session = self.session.backend_sessions[-1]
        backend_session.vendor_session_id = vendor_session_id
        backend_session.last_seen_at = observed_at
        backend_session.status = "active"

    def _touch_backend_session(self, observed_at: str, status: str = "active") -> None:
        backend_session = self.session.backend_sessions[-1]
        backend_session.last_seen_at = observed_at
        backend_session.status = status

    def _update_turn_recovery_metadata(
        self,
        turn: TurnRecord,
        *,
        state: str,
        terminal_event_seen: bool,
    ) -> None:
        recovery = dict(turn.metadata.get("recovery", {}))
        recovery.update(
            {
                "state": state,
                "terminal_event_seen": terminal_event_seen,
                "partial_output": False if state == "completed" else bool(turn.output.strip()),
                "prompt_char_count": len(turn.prompt),
                "output_char_count": len(turn.output),
                "started_at": turn.started_at,
                "completed_at": turn.completed_at,
                "status": turn.status.value,
                "error_kind": turn.error.kind if turn.error is not None else None,
                "error_message": turn.error.message if turn.error is not None else None,
                "event_count": len(turn.events),
                "last_event_type": turn.events[-1].type if turn.events else None,
            }
        )
        if state == "failed":
            recovery["recovery_notes"] = (
                "This turn failed before a successful completion. Resume should inspect the stored error and "
                "treat any assistant output as partial unless a later completed turn supersedes it."
            )
        if state == "interrupted":
            recovery["recovery_notes"] = (
                "This turn stopped without a backend completion event. Recovery behavior is local-transcript based; "
                "backend-native resume remains backend-dependent."
            )
        turn.metadata["recovery"] = recovery

    @staticmethod
    def _annotate_injected_prompt_metadata(
        metadata: dict,
        *,
        visible_prompt: str,
        submitted_prompt: str,
        backend_prompt: str | None,
    ) -> None:
        for key in ("handoff", "delegated_context"):
            injected_metadata = metadata.get(key)
            if not isinstance(injected_metadata, dict) or not injected_metadata.get("injected"):
                continue
            annotated_metadata = dict(injected_metadata)
            annotated_metadata.setdefault("visible_prompt", visible_prompt)
            annotated_metadata["submitted_prompt_char_count"] = len(submitted_prompt)
            if backend_prompt is not None:
                annotated_metadata["submitted_prompt_kind"] = "injected_backend_prompt"
                annotated_metadata["backend_prompt_char_count"] = len(backend_prompt)
            else:
                annotated_metadata.setdefault("submitted_prompt_kind", "visible_prompt")
            metadata[key] = annotated_metadata

    @staticmethod
    def _handoff_lineage_metadata(handoff_packet: HandoffPacket) -> dict:
        metadata = handoff_packet.metadata
        lineage_keys = (
            "mode",
            "source_backend",
            "target_backend",
            "target_model",
            "source_scope",
            "source_task_id",
            "source_summary_id",
            "source_summary_scope",
            "source_turn_ids",
            "source_turn_count",
            "selection_criteria",
            "context_char_count",
            "backend_prompt_char_count",
            "recent_turn_limit",
            "max_context_chars",
            "max_turn_chars",
            "max_summary_chars",
            "audit",
        )
        return {
            key: metadata[key]
            for key in lineage_keys
            if key in metadata
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
