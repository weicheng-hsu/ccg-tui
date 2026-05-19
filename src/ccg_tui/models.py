from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BackendName(str, Enum):
    CODEX = "codex"
    CLAUDE = "claude"
    GEMINI = "gemini"


class TurnStatus(str, Enum):
    SUBMITTING = "submitting"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    SESSION_STARTED = "session_started"
    ACTIVITY = "activity"
    OUTPUT_STARTED = "output_started"
    OUTPUT_DELTA = "output_delta"
    BACKEND_SUCCEEDED = "backend_succeeded"
    BACKEND_FAILED = "backend_failed"


class RoutingDecision(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    NOT_APPLICABLE = "not_applicable"


_ROUTING_DECISION_ALIASES = {
    "": RoutingDecision.NOT_APPLICABLE.value,
    "none": RoutingDecision.NOT_APPLICABLE.value,
    "unknown": RoutingDecision.NOT_APPLICABLE.value,
}


def normalize_routing_decision(decision: str | RoutingDecision | None) -> str:
    if isinstance(decision, RoutingDecision):
        return decision.value
    normalized = str(decision or "").strip().lower()
    normalized = _ROUTING_DECISION_ALIASES.get(normalized, normalized)
    try:
        return RoutingDecision(normalized).value
    except ValueError as exc:
        allowed = ", ".join(item.value for item in RoutingDecision)
        raise ValueError(f"unknown routing decision: {decision!r}; expected one of {allowed}") from exc


NORMALIZED_ERROR_KINDS = frozenset(
    {
        "adapter_exception",
        "auth_error",
        "backend_error",
        "interrupted",
        "process_exit",
        "rate_limit",
        "timeout_error",
    }
)

_ERROR_KIND_ALIASES = {
    "authentication_error": "auth_error",
    "missing_api_key": "auth_error",
    "quota_exceeded": "rate_limit",
    "rate_limit_exceeded": "rate_limit",
    "timeout": "timeout_error",
    "usage_limit_exceeded": "rate_limit",
}


def normalize_error_kind(kind: str | None) -> str:
    normalized = str(kind or "").strip().lower().replace("-", "_")
    if not normalized:
        return "backend_error"
    if normalized in NORMALIZED_ERROR_KINDS:
        return normalized
    return _ERROR_KIND_ALIASES.get(normalized, "backend_error")


@dataclass(slots=True)
class NormalizedError:
    kind: str
    message: str
    exit_code: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        original_kind = str(self.kind or "").strip()
        self.kind = normalize_error_kind(original_kind)
        if original_kind and original_kind.lower().replace("-", "_") != self.kind:
            self.details.setdefault("original_kind", original_kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "exit_code": self.exit_code,
            "details": dict(self.details),
        }


@dataclass(slots=True)
class BackendEvent:
    type: EventType
    text: str = ""
    session_id: str | None = None
    activity: dict[str, Any] | None = None
    error: NormalizedError | None = None
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class RecordedEvent:
    type: str
    observed_at: str
    text: str = ""
    session_id: str | None = None
    activity: dict[str, Any] | None = None
    error: NormalizedError | None = None
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "observed_at": self.observed_at,
            "text": self.text,
            "session_id": self.session_id,
            "activity": self.activity,
            "error": self.error.to_dict() if self.error is not None else None,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecordedEvent":
        error = data.get("error")
        return cls(
            type=data["type"],
            observed_at=data.get("observed_at", ""),
            text=data.get("text", ""),
            session_id=data.get("session_id"),
            activity=data.get("activity"),
            error=NormalizedError(**error) if error else None,
            raw=data.get("raw"),
        )


@dataclass(slots=True)
class BackendSessionRecord:
    id: str
    backend: BackendName
    created_at: str
    last_seen_at: str
    vendor_session_id: str | None = None
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "backend": self.backend.value,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "vendor_session_id": self.vendor_session_id,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackendSessionRecord":
        return cls(
            id=data["id"],
            backend=BackendName(data["backend"]),
            created_at=data["created_at"],
            last_seen_at=data.get("last_seen_at", data["created_at"]),
            vendor_session_id=data.get("vendor_session_id"),
            status=data.get("status", "pending"),
        )


@dataclass(slots=True)
class TaskRecord:
    id: str
    created_at: str
    updated_at: str
    kind: str = "primary"
    title: str | None = None
    status: str = "active"
    start_turn_id: str | None = None
    end_turn_id: str | None = None
    summary_id: str | None = None
    closing_note: str | None = None
    turn_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "start_turn_id": self.start_turn_id,
            "end_turn_id": self.end_turn_id,
            "summary_id": self.summary_id,
            "closing_note": self.closing_note,
            "turn_ids": list(self.turn_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskRecord":
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            updated_at=data.get("updated_at", data["created_at"]),
            kind=data.get("kind", "primary"),
            title=data.get("title"),
            status=data.get("status", "active"),
            start_turn_id=data.get("start_turn_id"),
            end_turn_id=data.get("end_turn_id"),
            summary_id=data.get("summary_id"),
            closing_note=data.get("closing_note"),
            turn_ids=list(data.get("turn_ids", [])),
        )


@dataclass(slots=True)
class SummaryRecord:
    id: str
    scope: str
    created_at: str
    text: str = ""
    source_turn_ids: list[str] = field(default_factory=list)
    kind: str = "summary"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "created_at": self.created_at,
            "text": self.text,
            "source_turn_ids": list(self.source_turn_ids),
            "kind": self.kind,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummaryRecord":
        return cls(
            id=data["id"],
            scope=data["scope"],
            created_at=data["created_at"],
            text=data.get("text", ""),
            source_turn_ids=list(data.get("source_turn_ids", [])),
            kind=data.get("kind", "summary"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class ArtifactRecord:
    id: str
    kind: str
    created_at: str
    label: str = ""
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "created_at": self.created_at,
            "label": self.label,
            "path": self.path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRecord":
        return cls(
            id=data["id"],
            kind=data["kind"],
            created_at=data["created_at"],
            label=data.get("label", ""),
            path=data.get("path"),
            metadata=dict(data.get("metadata", {})),
        )


SESSION_LINEAGE_KINDS = frozenset({"root", "child", "fork", "handoff", "delegated"})
SESSION_RELATIONSHIP_KINDS = frozenset({"parent", "child", "fork", "handoff", "delegated"})


def _normalize_lineage_kind(kind: str) -> str:
    normalized = str(kind or "root").strip() or "root"
    if normalized not in SESSION_LINEAGE_KINDS:
        raise ValueError(f"unknown session lineage kind: {kind!r}")
    return normalized


def _normalize_relationship_kind(kind: str) -> str:
    normalized = str(kind or "").strip()
    if normalized not in SESSION_RELATIONSHIP_KINDS:
        raise ValueError(f"unknown session relationship kind: {kind!r}")
    return normalized


@dataclass(slots=True)
class SessionRelationshipRecord:
    kind: str
    session_id: str
    source_turn_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kind = _normalize_relationship_kind(self.kind)
        if not self.session_id:
            raise ValueError("session relationship requires session_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "session_id": self.session_id,
            "source_turn_id": self.source_turn_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionRelationshipRecord":
        return cls(
            kind=data["kind"],
            session_id=data["session_id"],
            source_turn_id=data.get("source_turn_id"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class SubagentRunRecord:
    id: str
    created_at: str
    updated_at: str
    status: str = "pending"
    backend: BackendName | None = None
    task_id: str | None = None
    parent_turn_id: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "backend": self.backend.value if self.backend is not None else None,
            "task_id": self.task_id,
            "parent_turn_id": self.parent_turn_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubagentRunRecord":
        backend = data.get("backend")
        return cls(
            id=data["id"],
            created_at=data["created_at"],
            updated_at=data.get("updated_at", data["created_at"]),
            status=data.get("status", "pending"),
            backend=BackendName(backend) if backend else None,
            task_id=data.get("task_id"),
            parent_turn_id=data.get("parent_turn_id"),
            session_id=data.get("session_id"),
        )


@dataclass(slots=True)
class RoutingDecisionRecord:
    id: str
    recorded_at: str
    active_backend: BackendName
    trigger: str
    user_decision: str
    final_action: str
    suggested_backend: BackendName | None = None
    suggested_model: str | None = None
    policy_reference: str = "README.md"
    permission_state: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    compatibility: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.active_backend, BackendName):
            self.active_backend = BackendName(str(self.active_backend))
        if self.suggested_backend is not None and not isinstance(self.suggested_backend, BackendName):
            self.suggested_backend = BackendName(str(self.suggested_backend))
        self.user_decision = normalize_routing_decision(self.user_decision)

    @property
    def decision(self) -> str:
        return self.user_decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "recorded_at": self.recorded_at,
            "active_backend": self.active_backend.value,
            "suggested_backend": self.suggested_backend.value if self.suggested_backend is not None else None,
            "suggested_model": self.suggested_model,
            "trigger": self.trigger,
            "policy_reference": self.policy_reference,
            "permission_state": dict(self.permission_state),
            "decision": self.user_decision,
            "user_decision": self.user_decision,
            "final_action": self.final_action,
            "reason": self.reason,
            "compatibility": dict(self.compatibility),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoutingDecisionRecord":
        suggested_backend = data.get("suggested_backend")
        return cls(
            id=data["id"],
            recorded_at=data["recorded_at"],
            active_backend=BackendName(data["active_backend"]),
            suggested_backend=BackendName(suggested_backend) if suggested_backend else None,
            suggested_model=data.get("suggested_model"),
            trigger=data.get("trigger", ""),
            policy_reference=data.get("policy_reference", "README.md"),
            permission_state=dict(data.get("permission_state", {})),
            user_decision=data.get("user_decision", data.get("decision", "not_applicable")),
            final_action=data.get("final_action", "none"),
            reason=data.get("reason", ""),
            compatibility=dict(data.get("compatibility", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class SessionLineageRecord:
    kind: str = "root"
    parent_session_id: str | None = None
    resumed_from_session_id: str | None = None
    forked_from_turn_id: str | None = None
    relationships: list[SessionRelationshipRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.kind = _normalize_lineage_kind(self.kind)
        if self.parent_session_id is None and self.resumed_from_session_id is not None:
            self.parent_session_id = self.resumed_from_session_id
        self.relationships = self._normalized_relationships(self.relationships)

    @classmethod
    def root(cls) -> "SessionLineageRecord":
        return cls(kind="root")

    @classmethod
    def for_child(
        cls,
        parent_session_id: str,
        *,
        forked_from_turn_id: str | None = None,
    ) -> "SessionLineageRecord":
        return cls(
            kind="child",
            parent_session_id=parent_session_id,
            forked_from_turn_id=forked_from_turn_id,
        )

    @classmethod
    def for_fork(
        cls,
        parent_session_id: str,
        *,
        forked_from_turn_id: str | None,
    ) -> "SessionLineageRecord":
        return cls(
            kind="fork",
            parent_session_id=parent_session_id,
            forked_from_turn_id=forked_from_turn_id,
        )

    @classmethod
    def for_handoff(
        cls,
        parent_session_id: str,
        *,
        forked_from_turn_id: str | None = None,
        relationship_metadata: dict[str, Any] | None = None,
    ) -> "SessionLineageRecord":
        relationships: list[SessionRelationshipRecord] = []
        if relationship_metadata:
            relationships = [
                SessionRelationshipRecord(
                    kind="parent",
                    session_id=parent_session_id,
                    source_turn_id=forked_from_turn_id,
                    metadata=dict(relationship_metadata),
                ),
                SessionRelationshipRecord(
                    kind="handoff",
                    session_id=parent_session_id,
                    source_turn_id=forked_from_turn_id,
                    metadata=dict(relationship_metadata),
                ),
            ]
        return cls(
            kind="handoff",
            parent_session_id=parent_session_id,
            resumed_from_session_id=parent_session_id,
            forked_from_turn_id=forked_from_turn_id,
            relationships=relationships,
        )

    @classmethod
    def for_delegated(
        cls,
        parent_session_id: str,
        *,
        forked_from_turn_id: str | None = None,
    ) -> "SessionLineageRecord":
        return cls(
            kind="delegated",
            parent_session_id=parent_session_id,
            forked_from_turn_id=forked_from_turn_id,
        )

    def _normalized_relationships(
        self,
        relationships: list[SessionRelationshipRecord | dict[str, Any]],
    ) -> list[SessionRelationshipRecord]:
        normalized: list[SessionRelationshipRecord] = []
        for relationship in relationships:
            if isinstance(relationship, SessionRelationshipRecord):
                normalized.append(relationship)
            else:
                normalized.append(SessionRelationshipRecord.from_dict(relationship))

        derived_parent_session_id = self.parent_session_id
        if derived_parent_session_id:
            normalized.append(
                SessionRelationshipRecord(
                    kind="parent",
                    session_id=derived_parent_session_id,
                    source_turn_id=self.forked_from_turn_id,
                )
            )
            if self.kind in {"fork", "handoff", "delegated"}:
                normalized.append(
                    SessionRelationshipRecord(
                        kind=self.kind,
                        session_id=derived_parent_session_id,
                        source_turn_id=self.forked_from_turn_id,
                    )
                )

        deduped: dict[tuple[str, str], SessionRelationshipRecord] = {}
        for relationship in normalized:
            key = (relationship.kind, relationship.session_id)
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = relationship
                continue
            if existing.source_turn_id is None and relationship.source_turn_id is not None:
                existing.source_turn_id = relationship.source_turn_id
            if relationship.metadata:
                existing.metadata.update(relationship.metadata)
        return list(deduped.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "parent_session_id": self.parent_session_id,
            "resumed_from_session_id": self.resumed_from_session_id,
            "forked_from_turn_id": self.forked_from_turn_id,
            "relationships": [relationship.to_dict() for relationship in self.relationships],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionLineageRecord":
        return cls(
            kind=data.get("kind", "root"),
            parent_session_id=data.get("parent_session_id"),
            resumed_from_session_id=data.get("resumed_from_session_id"),
            forked_from_turn_id=data.get("forked_from_turn_id"),
            relationships=list(data.get("relationships", [])),
        )


@dataclass(slots=True)
class TurnRecord:
    id: str
    backend: BackendName
    prompt: str
    output: str
    status: TurnStatus
    started_at: str
    completed_at: str | None = None
    error: NormalizedError | None = None
    task_id: str = "task-main"
    vendor_session_id: str | None = None
    agent_role: str = "primary_agent"
    events: list[RecordedEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "backend": self.backend.value,
            "prompt": self.prompt,
            "output": self.output,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error.to_dict() if self.error is not None else None,
            "task_id": self.task_id,
            "vendor_session_id": self.vendor_session_id,
            "agent_role": self.agent_role,
            "events": [event.to_dict() for event in self.events],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TurnRecord":
        error = data.get("error")
        return cls(
            id=data["id"],
            backend=BackendName(data["backend"]),
            prompt=data["prompt"],
            output=data["output"],
            status=TurnStatus(data["status"]),
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            error=NormalizedError(**error) if error else None,
            task_id=data.get("task_id", "task-main"),
            vendor_session_id=data.get("vendor_session_id"),
            agent_role=data.get("agent_role", "primary_agent"),
            events=[RecordedEvent.from_dict(event) for event in data.get("events", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class SessionRecord:
    id: str
    backend: BackendName
    created_at: str
    turns: list[TurnRecord] = field(default_factory=list)
    vendor_session_id: str | None = None
    schema_version: int = 5
    updated_at: str | None = None
    workspace_cwd: str = ""
    backend_sessions: list[BackendSessionRecord] = field(default_factory=list)
    tasks: list[TaskRecord] = field(default_factory=list)
    summaries: list[SummaryRecord] = field(default_factory=list)
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    subagent_runs: list[SubagentRunRecord] = field(default_factory=list)
    routing_decisions: list[RoutingDecisionRecord] = field(default_factory=list)
    lineage: SessionLineageRecord = field(default_factory=SessionLineageRecord)

    def __post_init__(self) -> None:
        if self.updated_at is None:
            self.updated_at = self.created_at
        if not self.backend_sessions:
            self.backend_sessions.append(
                BackendSessionRecord(
                    id=f"backend-{self.backend.value}-primary",
                    backend=self.backend,
                    created_at=self.created_at,
                    last_seen_at=self.updated_at,
                    vendor_session_id=self.vendor_session_id,
                    status="active" if self.vendor_session_id else "pending",
                )
            )
        if not self.tasks:
            self.tasks.append(
                TaskRecord(
                    id="task-main",
                    created_at=self.created_at,
                    updated_at=self.updated_at,
                )
            )
        self.routing_decisions = [
            decision
            if isinstance(decision, RoutingDecisionRecord)
            else RoutingDecisionRecord.from_dict(decision)
            for decision in self.routing_decisions
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "backend": self.backend.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "workspace_cwd": self.workspace_cwd,
            "vendor_session_id": self.vendor_session_id,
            "backend_sessions": [session.to_dict() for session in self.backend_sessions],
            "tasks": [task.to_dict() for task in self.tasks],
            "summaries": [summary.to_dict() for summary in self.summaries],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "subagent_runs": [run.to_dict() for run in self.subagent_runs],
            "routing_decisions": [decision.to_dict() for decision in self.routing_decisions],
            "lineage": self.lineage.to_dict(),
            "turns": [turn.to_dict() for turn in self.turns],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionRecord":
        backend = BackendName(data["backend"])
        created_at = data["created_at"]
        vendor_session_id = data.get("vendor_session_id")
        backend_sessions_data = data.get("backend_sessions")
        if backend_sessions_data:
            backend_sessions = [BackendSessionRecord.from_dict(item) for item in backend_sessions_data]
        else:
            updated_at = data.get("updated_at", created_at)
            backend_sessions = [
                BackendSessionRecord(
                    id=f"backend-{backend.value}-primary",
                    backend=backend,
                    created_at=created_at,
                    last_seen_at=updated_at,
                    vendor_session_id=vendor_session_id,
                    status="active" if vendor_session_id else "pending",
                )
            ]
        tasks_data = data.get("tasks")
        tasks = [TaskRecord.from_dict(item) for item in tasks_data] if tasks_data else []
        return cls(
            schema_version=data.get("schema_version", 1),
            id=data["id"],
            backend=backend,
            created_at=created_at,
            updated_at=data.get("updated_at"),
            workspace_cwd=data.get("workspace_cwd", ""),
            vendor_session_id=vendor_session_id,
            backend_sessions=backend_sessions,
            tasks=tasks,
            summaries=[SummaryRecord.from_dict(item) for item in data.get("summaries", [])],
            artifacts=[ArtifactRecord.from_dict(item) for item in data.get("artifacts", [])],
            subagent_runs=[SubagentRunRecord.from_dict(item) for item in data.get("subagent_runs", [])],
            routing_decisions=[
                RoutingDecisionRecord.from_dict(item)
                for item in data.get("routing_decisions", [])
            ],
            lineage=SessionLineageRecord.from_dict(data.get("lineage", {})),
            turns=[TurnRecord.from_dict(turn) for turn in data.get("turns", [])],
        )
