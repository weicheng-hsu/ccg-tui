from ccg_tui.backends.base import BackendAdapter
from ccg_tui.handoff import (
    build_delegated_context_packet,
    build_delegated_result_payload,
    build_handoff_packet,
)
from ccg_tui.models import (
    ArtifactRecord,
    BackendEvent,
    BackendName,
    EventType,
    NormalizedError,
    SessionLineageRecord,
    SessionRecord,
    SummaryRecord,
    TurnRecord,
    TurnStatus,
)
from ccg_tui.resume_context import ResumeContextConfig
from ccg_tui.session import SessionController
from ccg_tui.transcript import TranscriptStore


class FakeAdapter:
    def __init__(self, name, events):
        self.name = name
        self._events = events
        self.closed = False

    def run(self, prompt, cwd):
        yield from self._events

    def close(self):
        self.closed = True


class FakePersistentSession:
    def __init__(self):
        self.prompts = []
        self.closed = False

    def run(self, prompt):
        self.prompts.append(prompt)
        yield BackendEvent(type=EventType.OUTPUT_STARTED)
        yield BackendEvent(type=EventType.OUTPUT_DELTA, text=f"reply:{prompt}")
        yield BackendEvent(type=EventType.BACKEND_SUCCEEDED)

    def close(self):
        self.closed = True


class FakePersistentAdapter(BackendAdapter):
    name = BackendName.CODEX

    def __init__(self):
        super().__init__()
        self.open_count = 0
        self.session = FakePersistentSession()

    def build_command(self, prompt, cwd):
        return []

    def parse_stdout_line(self, line):
        return []

    def open_session(self, cwd):
        self.open_count += 1
        return self.session


class RecordingAdapter:
    name = BackendName.CODEX

    def __init__(self):
        self.prompts = []

    def run(self, prompt, cwd):
        self.prompts.append(prompt)
        yield BackendEvent(type=EventType.OUTPUT_STARTED)
        yield BackendEvent(type=EventType.OUTPUT_DELTA, text="recorded")
        yield BackendEvent(type=EventType.BACKEND_SUCCEEDED)

    def close(self):
        return None


class RaisingAdapter:
    name = BackendName.CODEX

    def run(self, prompt, cwd):
        raise RuntimeError("transport stopped")
        yield

    def close(self):
        return None


class PartialRaisingAdapter:
    name = BackendName.CODEX

    def run(self, prompt, cwd):
        yield BackendEvent(type=EventType.OUTPUT_STARTED)
        yield BackendEvent(type=EventType.OUTPUT_DELTA, text="partial before exception")
        raise RuntimeError("transport stopped")

    def close(self):
        return None


def test_session_controller_collects_output_and_persists_turn(tmp_path):
    adapter = FakeAdapter(
        BackendName.CODEX,
        [
            BackendEvent(type=EventType.SESSION_STARTED, session_id="vendor-1"),
            BackendEvent(type=EventType.OUTPUT_STARTED),
            BackendEvent(type=EventType.OUTPUT_DELTA, text="hello "),
            BackendEvent(type=EventType.OUTPUT_DELTA, text="world"),
            BackendEvent(type=EventType.BACKEND_SUCCEEDED),
        ],
    )
    controller = SessionController(adapter=adapter, store=TranscriptStore(tmp_path), cwd=tmp_path)

    turn = controller.submit_prompt("say hi")

    assert turn.output == "hello world"
    assert turn.status is TurnStatus.COMPLETED
    assert controller.session.vendor_session_id == "vendor-1"

    loaded = controller.store.load_session(controller.session.id)
    assert loaded.turns[0].output == "hello world"
    assert loaded.workspace_cwd == str(tmp_path)
    assert loaded.backend_sessions[0].vendor_session_id == "vendor-1"
    assert loaded.backend_sessions[0].status == "active"
    assert loaded.tasks[0].turn_ids == [turn.id]
    assert loaded.turns[0].task_id == "task-main"
    assert loaded.turns[0].vendor_session_id == "vendor-1"
    assert [event.type for event in loaded.turns[0].events] == [
        "session_started",
        "output_started",
        "output_delta",
        "output_delta",
        "backend_succeeded",
    ]
    assert loaded.turns[0].metadata["recovery"]["state"] == "completed"
    assert loaded.turns[0].metadata["recovery"]["terminal_event_seen"] is True
    assert loaded.turns[0].metadata["recovery"]["partial_output"] is False
    assert loaded.turns[0].metadata["recovery"]["output_char_count"] == len("hello world")


def test_session_controller_routes_turns_to_active_task_and_back_to_main(tmp_path):
    controller = SessionController(
        adapter=FakeAdapter(BackendName.CODEX, [BackendEvent(type=EventType.BACKEND_SUCCEEDED)]),
        store=TranscriptStore(tmp_path),
        cwd=tmp_path,
    )

    task = controller.start_task("Phase 4")
    task_turn = controller.submit_prompt("implement task boundary")
    controller.close_task("done")
    main_turn = controller.submit_prompt("follow up on main flow")
    loaded = controller.store.load_session(controller.session.id)

    assert task_turn.task_id == task.id
    assert main_turn.task_id == "task-main"
    assert loaded.tasks[0].turn_ids == [main_turn.id]
    assert loaded.tasks[0].start_turn_id == main_turn.id
    closed_task = next(saved_task for saved_task in loaded.tasks if saved_task.id == task.id)
    assert closed_task.status == "closed"
    assert closed_task.title == "Phase 4"
    assert closed_task.start_turn_id == task_turn.id
    assert closed_task.end_turn_id == task_turn.id
    assert closed_task.closing_note == "done"
    assert closed_task.turn_ids == [task_turn.id]


def test_session_controller_rejects_nested_active_tasks(tmp_path):
    controller = SessionController(
        adapter=FakeAdapter(BackendName.CODEX, [BackendEvent(type=EventType.BACKEND_SUCCEEDED)]),
        store=TranscriptStore(tmp_path),
        cwd=tmp_path,
    )

    controller.start_task("first")

    try:
        controller.start_task("second")
    except ValueError as exc:
        assert "Task already active" in str(exc)
    else:
        raise AssertionError("expected nested task creation to fail")


def test_session_controller_emits_streaming_updates_via_callback(tmp_path):
    adapter = FakeAdapter(
        BackendName.CLAUDE,
        [
            BackendEvent(type=EventType.OUTPUT_STARTED),
            BackendEvent(type=EventType.OUTPUT_DELTA, text="hello"),
            BackendEvent(type=EventType.OUTPUT_DELTA, text=" world"),
            BackendEvent(type=EventType.BACKEND_SUCCEEDED),
        ],
    )
    controller = SessionController(adapter=adapter, store=TranscriptStore(tmp_path), cwd=tmp_path)
    snapshots: list[tuple[str, str]] = []

    turn = controller.submit_prompt(
        "say hi",
        on_update=lambda current_turn: snapshots.append((current_turn.status.value, current_turn.output)),
    )

    assert snapshots[0] == ("streaming", "")
    assert snapshots[1] == ("streaming", "hello")
    assert snapshots[2] == ("streaming", "hello world")
    assert snapshots[-1] == ("completed", "hello world")
    assert turn.status is TurnStatus.COMPLETED


def test_session_controller_emits_activity_updates_via_callback(tmp_path):
    adapter = FakeAdapter(
        BackendName.CODEX,
        [
            BackendEvent(type=EventType.ACTIVITY, text="tool: exec_command rg --files"),
            BackendEvent(type=EventType.OUTPUT_DELTA, text="done"),
            BackendEvent(type=EventType.BACKEND_SUCCEEDED),
        ],
    )
    controller = SessionController(adapter=adapter, store=TranscriptStore(tmp_path), cwd=tmp_path)
    snapshots: list[tuple[str, str, list[str]]] = []

    turn = controller.submit_prompt(
        "inspect",
        on_update=lambda current_turn: snapshots.append(
            (
                current_turn.status.value,
                current_turn.output,
                [event.text for event in current_turn.events if event.type == "activity"],
            )
        ),
    )

    assert snapshots[0] == ("streaming", "", ["tool: exec_command rg --files"])
    assert snapshots[1] == ("streaming", "done", ["tool: exec_command rg --files"])
    assert snapshots[-1][0] == "completed"
    assert turn.events[0].type == "activity"


def test_session_controller_marks_missing_terminal_event_as_interrupted_failure(tmp_path):
    adapter = FakeAdapter(
        BackendName.CODEX,
        [
            BackendEvent(type=EventType.OUTPUT_STARTED),
            BackendEvent(type=EventType.OUTPUT_DELTA, text="partial output"),
        ],
    )
    controller = SessionController(adapter=adapter, store=TranscriptStore(tmp_path), cwd=tmp_path)

    turn = controller.submit_prompt("continue")
    loaded = controller.store.load_session(controller.session.id)

    assert turn.status is TurnStatus.FAILED
    assert turn.output == "partial output"
    assert turn.error is not None
    assert turn.error.kind == "interrupted"
    assert "partial" in turn.error.message.lower()
    assert turn.metadata["recovery"]["state"] == "interrupted"
    assert turn.metadata["recovery"]["partial_output"] is True
    assert loaded.turns[0].status is TurnStatus.FAILED
    assert loaded.turns[0].metadata["recovery"]["state"] == "interrupted"
    assert loaded.backend_sessions[0].status == "interrupted"


def test_session_controller_marks_missing_terminal_event_without_output_as_interrupted(tmp_path):
    controller = SessionController(
        adapter=FakeAdapter(BackendName.CODEX, []),
        store=TranscriptStore(tmp_path),
        cwd=tmp_path,
    )

    turn = controller.submit_prompt("continue")
    loaded = controller.store.load_session(controller.session.id)
    recovery = loaded.turns[0].metadata["recovery"]

    assert turn.status is TurnStatus.FAILED
    assert turn.output == ""
    assert turn.completed_at is not None
    assert turn.error is not None
    assert turn.error.kind == "interrupted"
    assert recovery["state"] == "interrupted"
    assert recovery["terminal_event_seen"] is False
    assert recovery["partial_output"] is False
    assert recovery["prompt_char_count"] == len("continue")
    assert recovery["output_char_count"] == 0
    assert recovery["status"] == "failed"
    assert recovery["event_count"] == 0
    assert recovery["last_event_type"] is None
    assert loaded.backend_sessions[0].status == "interrupted"


def test_session_controller_persists_backend_failure_with_partial_output(tmp_path):
    adapter = FakeAdapter(
        BackendName.CODEX,
        [
            BackendEvent(type=EventType.OUTPUT_STARTED),
            BackendEvent(type=EventType.OUTPUT_DELTA, text="partial output"),
            BackendEvent(
                type=EventType.BACKEND_FAILED,
                error=NormalizedError(kind="backend_error", message="backend boom", exit_code=2),
            ),
        ],
    )
    controller = SessionController(adapter=adapter, store=TranscriptStore(tmp_path), cwd=tmp_path)

    turn = controller.submit_prompt("run tests")
    loaded = controller.store.load_session(controller.session.id)
    saved_turn = loaded.turns[0]
    recovery = saved_turn.metadata["recovery"]

    assert turn.status is TurnStatus.FAILED
    assert saved_turn.prompt == "run tests"
    assert saved_turn.output == "partial output"
    assert saved_turn.completed_at is not None
    assert saved_turn.error is not None
    assert saved_turn.error.kind == "backend_error"
    assert saved_turn.error.message == "backend boom"
    assert saved_turn.error.exit_code == 2
    assert [event.type for event in saved_turn.events] == [
        "output_started",
        "output_delta",
        "backend_failed",
    ]
    assert recovery["state"] == "failed"
    assert recovery["terminal_event_seen"] is True
    assert recovery["partial_output"] is True
    assert recovery["started_at"] == saved_turn.started_at
    assert recovery["completed_at"] == saved_turn.completed_at
    assert recovery["error_kind"] == "backend_error"
    assert recovery["event_count"] == 3
    assert recovery["last_event_type"] == "backend_failed"
    assert loaded.backend_sessions[0].status == "failed"


def test_session_controller_uses_fallback_error_for_backend_failure_without_error(tmp_path):
    adapter = FakeAdapter(
        BackendName.CODEX,
        [
            BackendEvent(type=EventType.OUTPUT_STARTED),
            BackendEvent(type=EventType.BACKEND_FAILED),
        ],
    )
    controller = SessionController(adapter=adapter, store=TranscriptStore(tmp_path), cwd=tmp_path)

    turn = controller.submit_prompt("run tests")
    loaded = controller.store.load_session(controller.session.id)

    assert turn.status is TurnStatus.FAILED
    assert loaded.turns[0].error is not None
    assert loaded.turns[0].error.kind == "backend_error"
    assert loaded.turns[0].error.message == "Backend reported failure without error details"
    assert loaded.turns[0].events[-1].error is not None
    assert loaded.turns[0].events[-1].error.kind == "backend_error"
    assert loaded.turns[0].metadata["recovery"]["terminal_event_seen"] is True


def test_session_controller_marks_terminal_interrupted_event_as_interrupted(tmp_path):
    adapter = FakeAdapter(
        BackendName.CODEX,
        [
            BackendEvent(type=EventType.OUTPUT_STARTED),
            BackendEvent(type=EventType.OUTPUT_DELTA, text="partial output"),
            BackendEvent(
                type=EventType.BACKEND_FAILED,
                error=NormalizedError(kind="interrupted", message="transport interrupted"),
            ),
        ],
    )
    controller = SessionController(adapter=adapter, store=TranscriptStore(tmp_path), cwd=tmp_path)

    turn = controller.submit_prompt("continue")
    loaded = controller.store.load_session(controller.session.id)
    recovery = loaded.turns[0].metadata["recovery"]

    assert turn.status is TurnStatus.FAILED
    assert loaded.turns[0].error is not None
    assert loaded.turns[0].error.kind == "interrupted"
    assert recovery["state"] == "interrupted"
    assert recovery["terminal_event_seen"] is True
    assert recovery["partial_output"] is True


def test_session_controller_distinguishes_process_exit_from_adapter_exception(tmp_path):
    adapter = FakeAdapter(
        BackendName.CODEX,
        [
            BackendEvent(
                type=EventType.BACKEND_FAILED,
                error=NormalizedError(kind="process_exit", message="process exited", exit_code=130),
            ),
        ],
    )
    controller = SessionController(adapter=adapter, store=TranscriptStore(tmp_path), cwd=tmp_path)

    turn = controller.submit_prompt("continue")
    loaded = controller.store.load_session(controller.session.id)
    recovery = loaded.turns[0].metadata["recovery"]

    assert turn.status is TurnStatus.FAILED
    assert loaded.turns[0].error is not None
    assert loaded.turns[0].error.kind == "process_exit"
    assert loaded.turns[0].error.exit_code == 130
    assert recovery["state"] == "failed"
    assert recovery["terminal_event_seen"] is True
    assert recovery["error_kind"] == "process_exit"


def test_session_controller_finalizes_active_turn_when_adapter_raises(tmp_path):
    controller = SessionController(adapter=RaisingAdapter(), store=TranscriptStore(tmp_path), cwd=tmp_path)
    snapshots: list[TurnStatus] = []

    try:
        controller.submit_prompt("continue", on_update=lambda turn: snapshots.append(turn.status))
    except RuntimeError as exc:
        assert "transport stopped" in str(exc)
    else:
        raise AssertionError("expected adapter exception to propagate")

    loaded = controller.store.load_session(controller.session.id)

    assert controller.active_turn is None
    assert snapshots == [TurnStatus.FAILED]
    assert loaded.turns[0].status is TurnStatus.FAILED
    assert loaded.turns[0].error is not None
    assert loaded.turns[0].error.kind == "adapter_exception"
    assert loaded.turns[0].events[-1].type == "backend_failed"
    assert loaded.turns[0].events[-1].error is not None
    assert loaded.turns[0].events[-1].error.kind == "adapter_exception"
    assert loaded.turns[0].metadata["recovery"]["state"] == "failed"
    assert loaded.turns[0].metadata["recovery"]["terminal_event_seen"] is False
    assert loaded.backend_sessions[0].status == "failed"


def test_session_controller_persists_adapter_exception_after_partial_output(tmp_path):
    controller = SessionController(adapter=PartialRaisingAdapter(), store=TranscriptStore(tmp_path), cwd=tmp_path)

    try:
        controller.submit_prompt("continue")
    except RuntimeError as exc:
        assert "transport stopped" in str(exc)
    else:
        raise AssertionError("expected adapter exception to propagate")

    loaded = controller.store.load_session(controller.session.id)
    saved_turn = loaded.turns[0]
    recovery = saved_turn.metadata["recovery"]

    assert saved_turn.status is TurnStatus.FAILED
    assert saved_turn.prompt == "continue"
    assert saved_turn.output == "partial before exception"
    assert saved_turn.completed_at is not None
    assert saved_turn.error is not None
    assert saved_turn.error.kind == "adapter_exception"
    assert [event.type for event in saved_turn.events] == [
        "output_started",
        "output_delta",
        "backend_failed",
    ]
    assert saved_turn.events[-1].error is not None
    assert saved_turn.events[-1].error.kind == "adapter_exception"
    assert recovery["state"] == "failed"
    assert recovery["terminal_event_seen"] is False
    assert recovery["partial_output"] is True
    assert recovery["output_char_count"] == len("partial before exception")
    assert recovery["error_kind"] == "adapter_exception"
    assert recovery["last_event_type"] == "backend_failed"
    assert loaded.backend_sessions[0].status == "failed"


def test_session_controller_persists_structured_activity(tmp_path):
    adapter = FakeAdapter(
        BackendName.GEMINI,
        [
            BackendEvent(
                type=EventType.ACTIVITY,
                text="tools: ReadFile pyproject.toml",
                activity={
                    "kind": "tool",
                    "title": "Gemini tool call",
                    "backend_label": "tools: ReadFile pyproject.toml",
                    "status": "finished",
                    "details": {"tool_calls": [{"name": "read_file", "description": "pyproject.toml"}]},
                },
            ),
            BackendEvent(type=EventType.BACKEND_SUCCEEDED),
        ],
    )
    controller = SessionController(adapter=adapter, store=TranscriptStore(tmp_path), cwd=tmp_path)

    turn = controller.submit_prompt("inspect")
    loaded = controller.store.load_session(controller.session.id)

    assert turn.events[0].activity["details"]["tool_calls"][0]["description"] == "pyproject.toml"
    assert loaded.turns[0].events[0].activity["status"] == "finished"


def test_session_controller_close_finalizes_active_turn_as_interrupted(tmp_path):
    controller = SessionController(
        adapter=FakeAdapter(BackendName.CODEX, []),
        store=TranscriptStore(tmp_path),
        cwd=tmp_path,
    )
    turn = TurnRecord(
        id="turn-active",
        backend=BackendName.CODEX,
        prompt="still running",
        output="partial",
        status=TurnStatus.STREAMING,
        started_at="2026-04-21T15:05:01+00:00",
    )
    controller.session.turns.append(turn)
    controller.active_turn = turn
    controller.store.save_session(controller.session)

    controller.close()
    loaded = controller.store.load_session(controller.session.id)
    saved_turn = loaded.turns[0]
    recovery = saved_turn.metadata["recovery"]

    assert controller.active_turn is None
    assert saved_turn.status is TurnStatus.FAILED
    assert saved_turn.error is not None
    assert saved_turn.error.kind == "interrupted"
    assert saved_turn.events[-1].type == "backend_failed"
    assert recovery["state"] == "interrupted"
    assert recovery["terminal_event_seen"] is False
    assert recovery["partial_output"] is True
    assert recovery["interrupted_on_close"] is True
    assert loaded.backend_sessions[-1].status == "interrupted"


def test_session_controller_records_routing_decision_audit(tmp_path):
    controller = SessionController(
        adapter=FakeAdapter(BackendName.CODEX, [BackendEvent(type=EventType.BACKEND_SUCCEEDED)]),
        store=TranscriptStore(tmp_path),
        cwd=tmp_path,
    )

    decision = controller.record_routing_decision(
        suggested_backend=BackendName.CLAUDE,
        suggested_model="sonnet",
        trigger="manual_handoff",
        permission_state={
            "backend": "codex",
            "values": {
                "approval_policy": "on-request",
                "sandbox_mode": "workspace-write",
            },
        },
        user_decision="deferred",
        final_action="previewed",
        compatibility={"widens_permissions": False},
        metadata={"source_turn_ids": ["turn-1"]},
    )
    loaded = controller.store.load_session(controller.session.id)

    assert decision.active_backend is BackendName.CODEX
    assert loaded.routing_decisions[0].suggested_backend is BackendName.CLAUDE
    assert loaded.routing_decisions[0].suggested_model == "sonnet"
    assert loaded.routing_decisions[0].trigger == "manual_handoff"
    assert loaded.routing_decisions[0].permission_state["backend"] == "codex"
    assert loaded.routing_decisions[0].user_decision == "deferred"
    assert loaded.routing_decisions[0].final_action == "previewed"
    assert loaded.routing_decisions[0].compatibility["widens_permissions"] is False


def test_session_controller_rejects_backend_switch_with_existing_turns(tmp_path):
    adapter = FakeAdapter(BackendName.CODEX, [BackendEvent(type=EventType.BACKEND_SUCCEEDED)])
    controller = SessionController(adapter=adapter, store=TranscriptStore(tmp_path), cwd=tmp_path)
    controller.submit_prompt("one")

    other = FakeAdapter(BackendName.CLAUDE, [BackendEvent(type=EventType.BACKEND_SUCCEEDED)])

    try:
        controller.attach_backend(other)
    except ValueError as exc:
        assert "one backend per session" in str(exc).lower()
    else:
        raise AssertionError("expected backend switch to be rejected")


def test_session_controller_reuses_persistent_backend_session_and_closes_it(tmp_path):
    adapter = FakePersistentAdapter()
    controller = SessionController(adapter=adapter, store=TranscriptStore(tmp_path), cwd=tmp_path)

    first = controller.submit_prompt("one")
    second = controller.submit_prompt("two")
    controller.close()

    assert first.output == "reply:one"
    assert second.output == "reply:two"
    assert adapter.open_count == 1
    assert adapter.session.prompts == ["one", "two"]
    assert adapter.session.closed is True


def test_session_controller_resumes_existing_session_and_appends_turn(tmp_path):
    store = TranscriptStore(tmp_path)
    existing = SessionRecord(
        id="session-existing",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd=str(tmp_path),
        vendor_session_id="old-vendor",
        turns=[
            TurnRecord(
                id="turn-1",
                backend=BackendName.CODEX,
                prompt="first",
                output="old output",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:01+00:00",
                completed_at="2026-04-21T15:05:02+00:00",
            )
        ],
    )
    store.save_session(existing)
    adapter = FakeAdapter(
        BackendName.CODEX,
        [
            BackendEvent(type=EventType.OUTPUT_STARTED),
            BackendEvent(type=EventType.OUTPUT_DELTA, text="new output"),
            BackendEvent(type=EventType.BACKEND_SUCCEEDED),
        ],
    )

    controller = SessionController.resume(
        adapter=adapter,
        store=store,
        cwd=tmp_path,
        session=store.load_session("session-existing"),
    )
    turn = controller.submit_prompt("second")
    loaded = store.load_session("session-existing")

    assert controller.session.id == "session-existing"
    assert turn.output == "new output"
    assert loaded.id == "session-existing"
    assert [saved_turn.prompt for saved_turn in loaded.turns] == ["first", "second"]


def test_session_controller_resume_reconciles_stale_incomplete_turn(tmp_path):
    store = TranscriptStore(tmp_path)
    existing = SessionRecord(
        id="session-existing",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd=str(tmp_path),
        turns=[
            TurnRecord(
                id="turn-stale",
                backend=BackendName.CODEX,
                prompt="unfinished prompt",
                output="partial output",
                status=TurnStatus.STREAMING,
                started_at="2026-04-21T15:05:01+00:00",
                metadata={"recovery": {"state": "incomplete", "partial_output": False}},
            )
        ],
    )
    store.save_session(existing)

    controller = SessionController.resume(
        adapter=RecordingAdapter(),
        store=store,
        cwd=tmp_path,
        session=store.load_session("session-existing"),
    )
    loaded = store.load_session("session-existing")
    reconciled = loaded.turns[0]
    recovery = reconciled.metadata["recovery"]

    assert reconciled.status is TurnStatus.FAILED
    assert reconciled.output == "partial output"
    assert reconciled.completed_at is not None
    assert reconciled.error is not None
    assert reconciled.error.kind == "interrupted"
    assert reconciled.events[-1].type == "backend_failed"
    assert reconciled.events[-1].error is not None
    assert reconciled.events[-1].error.kind == "interrupted"
    assert recovery["state"] == "interrupted"
    assert recovery["terminal_event_seen"] is False
    assert recovery["partial_output"] is True
    assert recovery["reconciled_on_resume"] is True
    assert recovery["reconciled_at"]
    assert loaded.backend_sessions[0].status == "interrupted"
    assert loaded.backend_sessions[-1].status == "pending"
    assert controller.preview_resume_context("continue") is not None


def test_session_controller_starts_new_session_from_handoff_lineage(tmp_path):
    store = TranscriptStore(tmp_path)
    source = SessionRecord(
        id="session-source",
        backend=BackendName.CLAUDE,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd=str(tmp_path),
        summaries=[
            SummaryRecord(
                id="summary-source",
                scope="session",
                created_at="2026-04-21T15:05:03+00:00",
                text="Source checkpoint",
            )
        ],
        turns=[
            TurnRecord(
                id="turn-source",
                backend=BackendName.CLAUDE,
                prompt="source prompt",
                output="source output",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:01+00:00",
                completed_at="2026-04-21T15:05:02+00:00",
            )
        ],
    )
    store.save_session(source)
    adapter = RecordingAdapter()
    packet = build_handoff_packet(
        source,
        target_backend=BackendName.CODEX,
        target_model="gpt-test",
        user_goal="continue on target",
    )
    source_before = store.load_session(source.id).to_dict()

    controller = SessionController.from_handoff(
        adapter=adapter,
        store=store,
        cwd=tmp_path,
        source_session=source,
        handoff_packet=packet,
    )
    turn = controller.submit_prompt(
        "continue on target",
        backend_prompt=packet.backend_prompt,
        metadata={
            "handoff": {
                **packet.metadata,
                "injected": True,
                "source_context_char_count": len(packet.context_text),
            }
        },
    )
    loaded_target = store.load_session(controller.session.id)
    loaded_source = store.load_session(source.id)

    assert controller.session.id != source.id
    assert adapter.prompts == [packet.backend_prompt]
    assert turn.prompt == "continue on target"
    assert turn.metadata["handoff"]["injected"] is True
    assert turn.metadata["handoff"]["source_session_id"] == "session-source"
    assert turn.metadata["handoff"]["source_backend"] == "claude"
    assert turn.metadata["handoff"]["target_backend"] == "codex"
    assert turn.metadata["handoff"]["target_model"] == "gpt-test"
    assert turn.metadata["handoff"]["source_summary_id"] == "summary-source"
    assert turn.metadata["handoff"]["source_turn_ids"] == ["turn-source"]
    assert turn.metadata["handoff"]["visible_prompt"] == "continue on target"
    assert turn.metadata["handoff"]["submitted_prompt_kind"] == "injected_backend_prompt"
    assert turn.metadata["handoff"]["backend_prompt_char_count"] == len(packet.backend_prompt)
    assert turn.metadata["handoff"]["submitted_prompt_char_count"] == len(packet.backend_prompt)
    assert loaded_target.lineage.kind == "handoff"
    assert loaded_target.lineage.parent_session_id == "session-source"
    assert loaded_target.lineage.resumed_from_session_id == "session-source"
    assert loaded_target.lineage.forked_from_turn_id == "turn-source"
    assert [(relationship.kind, relationship.session_id) for relationship in loaded_target.lineage.relationships] == [
        ("parent", "session-source"),
        ("handoff", "session-source"),
    ]
    handoff_relationship = next(
        relationship for relationship in loaded_target.lineage.relationships if relationship.kind == "handoff"
    )
    assert handoff_relationship.metadata["source_turn_ids"] == ["turn-source"]
    assert handoff_relationship.metadata["target_model"] == "gpt-test"
    assert handoff_relationship.metadata["audit"]["turns"]["included_source_ids"] == ["turn-source"]
    assert loaded_target.turns[0].prompt == "continue on target"
    assert loaded_target.turns[0].metadata["handoff"]["visible_prompt"] == "continue on target"
    assert loaded_source.to_dict() == source_before
    assert loaded_target.turns[-1].vendor_session_id is None
    assert loaded_target.backend_sessions[-1].vendor_session_id is None


def test_session_controller_handoff_lineage_uses_packet_final_turns(tmp_path):
    store = TranscriptStore(tmp_path)
    source = SessionRecord(
        id="session-curated-source",
        backend=BackendName.CLAUDE,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd=str(tmp_path),
        summaries=[
            SummaryRecord(
                id="summary-alpha",
                scope="task:task-alpha",
                created_at="2026-04-21T15:05:03+00:00",
                text="Alpha checkpoint",
            )
        ],
        turns=[
            TurnRecord(
                id="turn-alpha",
                backend=BackendName.CLAUDE,
                prompt="alpha prompt",
                output="alpha output",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:01+00:00",
                completed_at="2026-04-21T15:05:02+00:00",
                task_id="task-alpha",
            ),
            TurnRecord(
                id="turn-latest",
                backend=BackendName.CLAUDE,
                prompt="latest unrelated",
                output="latest output",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:06:01+00:00",
                completed_at="2026-04-21T15:06:02+00:00",
                task_id="task-main",
            ),
        ],
    )
    store.save_session(source)
    packet = build_handoff_packet(
        source,
        target_backend=BackendName.CODEX,
        target_model="gpt-test",
        user_goal="continue alpha",
        scope="task",
        task_id="task-alpha",
        turn_ids=["turn-alpha"],
        statuses=["completed"],
        recent_turn_limit=1,
    )

    controller = SessionController.from_handoff(
        adapter=RecordingAdapter(),
        store=store,
        cwd=tmp_path,
        source_session=source,
        handoff_packet=packet,
    )
    loaded_target = store.load_session(controller.session.id)

    assert loaded_target.lineage.forked_from_turn_id == "turn-alpha"
    handoff_relationship = next(
        relationship for relationship in loaded_target.lineage.relationships if relationship.kind == "handoff"
    )
    assert handoff_relationship.source_turn_id == "turn-alpha"
    assert handoff_relationship.metadata["source_scope"] == "task:task-alpha"
    assert handoff_relationship.metadata["source_summary_id"] == "summary-alpha"
    assert handoff_relationship.metadata["source_turn_ids"] == ["turn-alpha"]
    assert handoff_relationship.metadata["selection_criteria"] == {
        "scope": "task:task-alpha",
        "task_id": "task-alpha",
        "turn_ids": ["turn-alpha"],
        "statuses": ["completed"],
        "recent_turn_limit": 1,
    }


def test_session_controller_starts_new_session_from_delegated_lineage(tmp_path):
    store = TranscriptStore(tmp_path)
    source = SessionRecord(
        id="session-parent",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd=str(tmp_path),
        summaries=[
            SummaryRecord(
                id="summary-parent",
                scope="task:task-main",
                created_at="2026-04-21T15:05:03+00:00",
                text="Parent checkpoint",
            )
        ],
        turns=[
            TurnRecord(
                id="turn-parent",
                backend=BackendName.CODEX,
                prompt="investigate parser",
                output="Need delegated verification",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:01+00:00",
                completed_at="2026-04-21T15:05:02+00:00",
            )
        ],
    )
    store.save_session(source)
    adapter = RecordingAdapter()
    packet = build_delegated_context_packet(
        source,
        target_backend=BackendName.CODEX,
        target_model="gpt-test",
        permission_mode="read-only",
        delegate_goal="verify parser assumptions",
        scope="task",
        task_id="task-main",
        turn_ids=["turn-parent"],
    )
    source_before = store.load_session(source.id).to_dict()

    controller = SessionController.from_delegated(
        adapter=adapter,
        store=store,
        cwd=tmp_path,
        source_session=source,
        context_packet=packet,
    )
    turn = controller.submit_prompt(
        "verify parser assumptions",
        backend_prompt=packet.backend_prompt,
        metadata={
            "delegated_context": {
                **packet.metadata,
                "injected": True,
                "source_context_char_count": len(packet.context_text),
            }
        },
    )
    loaded_child = store.load_session(controller.session.id)
    loaded_source = store.load_session(source.id)

    assert controller.session.id != source.id
    assert adapter.prompts == [packet.backend_prompt]
    assert turn.prompt == "verify parser assumptions"
    assert turn.metadata["delegated_context"]["injected"] is True
    assert turn.metadata["delegated_context"]["parent_session_id"] == "session-parent"
    assert turn.metadata["delegated_context"]["child_backend"] == "codex"
    assert turn.metadata["delegated_context"]["child_model"] == "gpt-test"
    assert turn.metadata["delegated_context"]["permission_mode"] == "read-only"
    assert turn.metadata["delegated_context"]["source_summary_id"] == "summary-parent"
    assert turn.metadata["delegated_context"]["source_turn_ids"] == ["turn-parent"]
    assert turn.metadata["delegated_context"]["visible_prompt"] == "verify parser assumptions"
    assert turn.metadata["delegated_context"]["submitted_prompt_kind"] == "injected_backend_prompt"
    assert turn.metadata["delegated_context"]["backend_prompt_char_count"] == len(packet.backend_prompt)
    assert turn.metadata["delegated_context"]["submitted_prompt_char_count"] == len(packet.backend_prompt)
    assert loaded_child.lineage.kind == "delegated"
    assert loaded_child.lineage.parent_session_id == "session-parent"
    assert loaded_child.lineage.forked_from_turn_id == "turn-parent"
    assert [(relationship.kind, relationship.session_id) for relationship in loaded_child.lineage.relationships] == [
        ("parent", "session-parent"),
        ("delegated", "session-parent"),
    ]
    assert loaded_child.turns[0].metadata["delegated_context"]["visible_prompt"] == "verify parser assumptions"
    assert loaded_source.to_dict() == source_before


def test_session_controller_promotes_delegated_result_into_parent_summary(tmp_path):
    store = TranscriptStore(tmp_path)
    parent = SessionRecord(
        id="session-parent",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd=str(tmp_path),
    )
    store.save_session(parent)
    controller = SessionController.resume(
        adapter=RecordingAdapter(),
        store=store,
        cwd=tmp_path,
        session=store.load_session("session-parent"),
        resume_context_config=ResumeContextConfig(enabled=False),
    )
    child = SessionRecord(
        id="session-child",
        backend=BackendName.CLAUDE,
        created_at="2026-04-21T15:06:00+00:00",
        workspace_cwd=str(tmp_path),
        summaries=[
            SummaryRecord(
                id="summary-child",
                scope="task:task-main",
                created_at="2026-04-21T15:06:03+00:00",
                text="Child checkpoint",
            )
        ],
        artifacts=[
            ArtifactRecord(
                id="artifact-log",
                kind="file",
                created_at="2026-04-21T15:06:04+00:00",
                path="logs/pytest.txt",
            )
        ],
        turns=[
            TurnRecord(
                id="turn-child",
                backend=BackendName.CLAUDE,
                prompt="verify parser assumptions",
                output="Verified with targeted checks.",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:06:01+00:00",
                completed_at="2026-04-21T15:06:02+00:00",
                metadata={
                    "delegated_context": {
                        "injected": True,
                        "parent_session_id": "session-parent",
                        "child_model": "sonnet",
                        "permission_mode": "workspace-write",
                        "source_summary_id": "summary-parent",
                        "source_turn_ids": ["turn-parent"],
                        "selection_criteria": {"scope": "task:task-main", "turn_ids": ["turn-parent"]},
                    }
                },
            )
        ],
        lineage=SessionLineageRecord.for_delegated("session-parent", forked_from_turn_id="turn-parent"),
    )

    payload = build_delegated_result_payload(
        child,
        result_text="Parser assumptions verified. See the delegated log reference.",
        summary_id="summary-child",
        turn_id="turn-child",
        artifact_ids=["artifact-log"],
    )
    summary = controller.promote_delegated_result(payload)
    loaded_parent = store.load_session("session-parent")

    assert summary.kind == "delegated_result"
    assert summary.scope == "task:task-main"
    assert summary.text == "Parser assumptions verified. See the delegated log reference."
    assert summary.metadata["delegated_session_id"] == "session-child"
    assert summary.metadata["delegated_summary_id"] == "summary-child"
    assert summary.metadata["delegated_turn_id"] == "turn-child"
    assert summary.metadata["delegated_artifact_ids"] == ["artifact-log"]
    assert summary.metadata["delegated_backend"] == "claude"
    assert summary.metadata["delegated_model"] == "sonnet"
    assert summary.metadata["permission_mode"] == "workspace-write"
    assert summary.metadata["source_turn_ids"] == ["turn-parent"]
    assert summary.metadata["promotion_scope"] == "task:task-main"
    assert loaded_parent.summaries[-1].kind == "delegated_result"
    assert loaded_parent.summaries[-1].metadata["delegated_session_id"] == "session-child"
    assert loaded_parent.tasks[0].summary_id == summary.id


def test_delegated_result_promotion_defaults_to_original_source_task(tmp_path):
    store = TranscriptStore(tmp_path)
    parent = SessionRecord(
        id="session-parent",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd=str(tmp_path),
    )
    store.save_session(parent)
    controller = SessionController.resume(
        adapter=RecordingAdapter(),
        store=store,
        cwd=tmp_path,
        session=store.load_session("session-parent"),
        resume_context_config=ResumeContextConfig(enabled=False),
    )
    source_task = controller.start_task("source task")
    controller.close_task("delegated")
    active_task = controller.start_task("active task")
    child = SessionRecord(
        id="session-child",
        backend=BackendName.CLAUDE,
        created_at="2026-04-21T15:06:00+00:00",
        turns=[
            TurnRecord(
                id="turn-child",
                backend=BackendName.CLAUDE,
                prompt="verify parser assumptions",
                output="Verified with targeted checks.",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:06:01+00:00",
                completed_at="2026-04-21T15:06:02+00:00",
                metadata={
                    "delegated_context": {
                        "injected": True,
                        "parent_session_id": "session-parent",
                        "selection_criteria": {
                            "scope": f"task:{source_task.id}",
                            "task_id": source_task.id,
                        },
                    }
                },
            )
        ],
        lineage=SessionLineageRecord.for_delegated("session-parent"),
    )

    payload = build_delegated_result_payload(child, result_text="Source task result", turn_id="turn-child")
    summary = controller.promote_delegated_result(payload)
    loaded_parent = store.load_session("session-parent")
    loaded_source_task = next(task for task in loaded_parent.tasks if task.id == source_task.id)
    loaded_active_task = next(task for task in loaded_parent.tasks if task.id == active_task.id)

    assert summary.scope == f"task:{source_task.id}"
    assert loaded_source_task.summary_id == summary.id
    assert loaded_active_task.summary_id is None


def test_delegated_result_promotion_defaults_to_session_scope_for_session_delegation(tmp_path):
    store = TranscriptStore(tmp_path)
    parent = SessionRecord(
        id="session-parent",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd=str(tmp_path),
    )
    store.save_session(parent)
    controller = SessionController.resume(
        adapter=RecordingAdapter(),
        store=store,
        cwd=tmp_path,
        session=store.load_session("session-parent"),
        resume_context_config=ResumeContextConfig(enabled=False),
    )
    active_task = controller.start_task("active task")
    child = SessionRecord(
        id="session-child",
        backend=BackendName.CLAUDE,
        created_at="2026-04-21T15:06:00+00:00",
        turns=[
            TurnRecord(
                id="turn-child",
                backend=BackendName.CLAUDE,
                prompt="verify session",
                output="Verified.",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:06:01+00:00",
                completed_at="2026-04-21T15:06:02+00:00",
                metadata={
                    "delegated_context": {
                        "injected": True,
                        "parent_session_id": "session-parent",
                        "selection_criteria": {"scope": "session", "task_id": None},
                    }
                },
            )
        ],
        lineage=SessionLineageRecord.for_delegated("session-parent"),
    )

    payload = build_delegated_result_payload(child, result_text="Session scoped result", turn_id="turn-child")
    summary = controller.promote_delegated_result(payload)
    loaded_parent = store.load_session("session-parent")
    loaded_active_task = next(task for task in loaded_parent.tasks if task.id == active_task.id)

    assert summary.scope == "session"
    assert loaded_parent.summaries[-1].id == summary.id
    assert loaded_active_task.summary_id is None


def test_delegated_result_promotion_validates_task_before_mutating_parent(tmp_path):
    store = TranscriptStore(tmp_path)
    parent = SessionRecord(
        id="session-parent",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd=str(tmp_path),
    )
    store.save_session(parent)
    controller = SessionController.resume(
        adapter=RecordingAdapter(),
        store=store,
        cwd=tmp_path,
        session=store.load_session("session-parent"),
        resume_context_config=ResumeContextConfig(enabled=False),
    )
    child = SessionRecord(
        id="session-child",
        backend=BackendName.CLAUDE,
        created_at="2026-04-21T15:06:00+00:00",
        turns=[
            TurnRecord(
                id="turn-child",
                backend=BackendName.CLAUDE,
                prompt="verify",
                output="Verified.",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:06:01+00:00",
                completed_at="2026-04-21T15:06:02+00:00",
                metadata={
                    "delegated_context": {
                        "injected": True,
                        "parent_session_id": "session-parent",
                        "selection_criteria": {"scope": "session", "task_id": None},
                    }
                },
            )
        ],
        lineage=SessionLineageRecord.for_delegated("session-parent"),
    )
    payload = build_delegated_result_payload(child, result_text="Will not promote", turn_id="turn-child")

    try:
        controller.promote_delegated_result(payload, scope="task", task_id="task-missing")
    except ValueError as exc:
        assert "unknown task id: task-missing" in str(exc)
    else:
        raise AssertionError("expected missing task to reject promotion")

    assert controller.session.summaries == []
    assert store.load_session("session-parent").summaries == []


def test_session_controller_rejects_cross_backend_resume_with_existing_turns(tmp_path):
    existing = SessionRecord(
        id="session-existing",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            TurnRecord(
                id="turn-1",
                backend=BackendName.CODEX,
                prompt="first",
                output="old output",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:01+00:00",
                completed_at="2026-04-21T15:05:02+00:00",
            )
        ],
    )

    try:
        SessionController.resume(
            adapter=FakeAdapter(BackendName.CLAUDE, []),
            store=TranscriptStore(tmp_path),
            cwd=tmp_path,
            session=existing,
        )
    except ValueError as exc:
        assert "one backend per session" in str(exc).lower()
    else:
        raise AssertionError("expected cross-backend resume to be rejected")


def test_session_controller_injects_resume_context_once_and_preserves_user_prompt(tmp_path):
    store = TranscriptStore(tmp_path)
    existing = SessionRecord(
        id="session-resume-context",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd=str(tmp_path),
        summaries=[
            SummaryRecord(
                id="summary-resume",
                scope="task:task-main",
                created_at="2026-04-21T15:05:03+00:00",
                text="Keep local resume context portable.",
            )
        ],
        turns=[
            TurnRecord(
                id="turn-old",
                backend=BackendName.CODEX,
                prompt="old prompt",
                output="old output",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:01+00:00",
                completed_at="2026-04-21T15:05:02+00:00",
            )
        ],
    )
    store.save_session(existing)
    adapter = RecordingAdapter()

    controller = SessionController.resume(
        adapter=adapter,
        store=store,
        cwd=tmp_path,
        session=store.load_session("session-resume-context"),
        resume_context_config=ResumeContextConfig(recent_turn_limit=1),
    )
    preview = controller.preview_resume_context()
    first = controller.submit_prompt("continue")
    second = controller.submit_prompt("again")
    loaded = store.load_session("session-resume-context")

    assert preview is not None
    assert "\n" not in adapter.prompts[0]
    assert "CCG LOCAL RESUME CONTEXT" in adapter.prompts[0]
    assert "Keep local resume context portable." in adapter.prompts[0]
    assert 'Current user prompt JSON: "continue"' in adapter.prompts[0]
    assert adapter.prompts[1] == "again"
    assert first.prompt == "continue"
    assert first.metadata["resume_context"]["injected_summary_id"] == "summary-resume"
    assert first.metadata["resume_context"]["injected_turn_ids"] == ["turn-old"]
    assert second.metadata["recovery"]["state"] == "completed"
    assert loaded.turns[-2].prompt == "continue"
    assert loaded.turns[-2].metadata["resume_context"]["injected"] is True


def test_session_controller_resume_context_can_be_disabled(tmp_path):
    existing = SessionRecord(
        id="session-resume-context-off",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            TurnRecord(
                id="turn-old",
                backend=BackendName.CODEX,
                prompt="old prompt",
                output="old output",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:01+00:00",
                completed_at="2026-04-21T15:05:02+00:00",
            )
        ],
    )
    adapter = RecordingAdapter()

    controller = SessionController.resume(
        adapter=adapter,
        store=TranscriptStore(tmp_path),
        cwd=tmp_path,
        session=existing,
        resume_context_config=ResumeContextConfig(enabled=False),
    )
    turn = controller.submit_prompt("continue")

    assert adapter.prompts == ["continue"]
    assert turn.metadata["recovery"]["state"] == "completed"
