from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ccg_tui.models import SessionRecord, TurnRecord, TurnStatus


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    id: str
    backend: str
    updated_at: str
    created_at: str
    turn_count: int
    summary_count: int
    latest_status: str
    resumable: bool
    workspace_basename: str

    @classmethod
    def from_session(cls, session: SessionRecord) -> "SessionMetadata":
        workspace = session.workspace_cwd or ""
        workspace_basename = Path(workspace).name or workspace
        latest_status = turn_transcript_state(session.turns[-1]) if session.turns else "idle"
        resumable = session_is_resumable(latest_status, len(session.turns))
        return cls(
            id=session.id,
            backend=session.backend.value,
            updated_at=session.updated_at or session.created_at,
            created_at=session.created_at,
            turn_count=len(session.turns),
            summary_count=len(session.summaries),
            latest_status=latest_status,
            resumable=resumable,
            workspace_basename=workspace_basename,
        )


@dataclass(frozen=True, slots=True)
class RelatedSessionMetadata:
    session_id: str
    lineage_kind: str
    relationship_kinds: tuple[str, ...]
    source_turn_ids: tuple[str, ...]


class TranscriptStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def session_path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def save_session(self, session: SessionRecord) -> Path:
        path = self.session_path(session.id)
        path.write_text(json.dumps(session.to_dict(), indent=2) + "\n")
        return path

    def load_session(self, session_id: str) -> SessionRecord:
        return SessionRecord.from_dict(json.loads(self.session_path(session_id).read_text()))

    def list_sessions(self) -> list[SessionMetadata]:
        sessions = [
            SessionMetadata.from_session(SessionRecord.from_dict(json.loads(path.read_text())))
            for path in self.root.glob("*.json")
        ]
        return sorted(
            sessions,
            key=lambda session: (session.updated_at or session.created_at, session.id),
            reverse=True,
        )

    def related_sessions(
        self,
        session_id: str,
        *,
        relationship_kinds: Iterable[str] | None = None,
    ) -> list[RelatedSessionMetadata]:
        current = self.load_session(session_id)
        filtered_kinds = {str(kind) for kind in relationship_kinds} if relationship_kinds is not None else None
        relationships: dict[str, dict[str, set[str]]] = {}
        lineage_kinds: dict[str, str] = {}
        stored_sessions = {
            path.stem: SessionRecord.from_dict(json.loads(path.read_text()))
            for path in self.root.glob("*.json")
            if path.stem != session_id
        }

        def register(related_session_id: str, *, kind: str, source_turn_id: str | None, lineage_kind: str) -> None:
            entry = relationships.setdefault(related_session_id, {"kinds": set(), "turn_ids": set()})
            entry["kinds"].add(kind)
            if source_turn_id:
                entry["turn_ids"].add(source_turn_id)
            lineage_kinds.setdefault(related_session_id, lineage_kind)

        for relationship in current.lineage.relationships:
            related_session = stored_sessions.get(relationship.session_id)
            register(
                relationship.session_id,
                kind=relationship.kind,
                source_turn_id=relationship.source_turn_id,
                lineage_kind=related_session.lineage.kind if related_session is not None else "root",
            )

        for other in stored_sessions.values():
            for relationship in other.lineage.relationships:
                if relationship.session_id != session_id:
                    continue
                inverse_kinds = _inverse_relationship_kinds(relationship.kind)
                for inverse_kind in inverse_kinds:
                    register(
                        other.id,
                        kind=inverse_kind,
                        source_turn_id=relationship.source_turn_id,
                        lineage_kind=other.lineage.kind,
                    )

        related = [
            RelatedSessionMetadata(
                session_id=related_session_id,
                lineage_kind=lineage_kinds.get(related_session_id, "root"),
                relationship_kinds=tuple(
                    kind for kind in _RELATIONSHIP_KIND_ORDER if kind in data["kinds"]
                ),
                source_turn_ids=tuple(sorted(data["turn_ids"])),
            )
            for related_session_id, data in relationships.items()
            if filtered_kinds is None or data["kinds"] & filtered_kinds
        ]
        return sorted(related, key=lambda item: item.session_id)


@dataclass(frozen=True, slots=True)
class SelectedTurnContext:
    source_turn_ids: tuple[str, ...]
    turn_count: int
    prompt_char_count: int
    output_char_count: int
    rendered_char_count: int


_TRANSCRIPT_STATUS_FILTERS = frozenset(
    {status.value for status in TurnStatus} | {"interrupted", "incomplete"}
)


def _normalize_statuses(statuses: Iterable[str | TurnStatus] | None) -> set[str]:
    if statuses is None:
        return set()
    normalized: set[str] = set()
    for status in statuses:
        if isinstance(status, TurnStatus):
            normalized.add(status.value)
            continue
        normalized_status = str(status).strip()
        if normalized_status not in _TRANSCRIPT_STATUS_FILTERS:
            raise ValueError(f"unknown turn status: {status!r}")
        normalized.add(normalized_status)
    return normalized


def filter_transcript_turns(
    session: SessionRecord,
    *,
    task_id: str | None = None,
    turn_ids: Iterable[str] | None = None,
    statuses: Iterable[str | TurnStatus] | None = None,
    recent_count: int | None = None,
) -> list[TurnRecord]:
    if recent_count is not None and recent_count < 0:
        raise ValueError("recent_count must be >= 0")

    selected = list(session.turns)
    if task_id is not None:
        selected = [turn for turn in selected if turn.task_id == task_id]

    normalized_statuses = _normalize_statuses(statuses)
    if normalized_statuses:
        selected = [
            turn
            for turn in selected
            if turn.status.value in normalized_statuses or turn_transcript_state(turn) in normalized_statuses
        ]

    if turn_ids is not None:
        all_turn_ids = {turn.id for turn in session.turns}
        selected_by_id = {turn.id: turn for turn in selected}
        missing_turn_ids = [turn_id for turn_id in turn_ids if turn_id not in all_turn_ids]
        if missing_turn_ids:
            missing = ", ".join(sorted(set(missing_turn_ids)))
            raise ValueError(f"unknown turn id(s): {missing}")
        excluded_turn_ids = [turn_id for turn_id in turn_ids if turn_id not in selected_by_id]
        if excluded_turn_ids:
            excluded = ", ".join(sorted(set(excluded_turn_ids)))
            raise ValueError(f"turn id(s) excluded by active filters: {excluded}")
        selected = [selected_by_id[turn_id] for turn_id in turn_ids]

    if recent_count is not None:
        selected = selected[-recent_count:] if recent_count > 0 else []

    return selected


def build_selected_turn_context(
    turns: list[TurnRecord],
    *,
    rendered_turns: list[str] | None = None,
) -> SelectedTurnContext:
    rendered = rendered_turns or []
    return SelectedTurnContext(
        source_turn_ids=tuple(turn.id for turn in turns),
        turn_count=len(turns),
        prompt_char_count=sum(len(turn.prompt) for turn in turns),
        output_char_count=sum(len(turn.output) for turn in turns),
        rendered_char_count=sum(len(text) for text in rendered),
    )


def turn_transcript_state(turn: TurnRecord) -> str:
    recovery = turn.metadata.get("recovery") if isinstance(turn.metadata, dict) else None
    if isinstance(recovery, dict):
        state = recovery.get("state")
        if isinstance(state, str) and state:
            return state
    if turn.status is TurnStatus.FAILED:
        return "failed"
    if turn.status in {TurnStatus.SUBMITTING, TurnStatus.STREAMING, TurnStatus.CANCELLED}:
        return "incomplete"
    return turn.status.value


def turn_has_partial_output(turn: TurnRecord) -> bool:
    recovery = turn.metadata.get("recovery") if isinstance(turn.metadata, dict) else None
    if isinstance(recovery, dict) and recovery.get("state") == "completed":
        return False
    if isinstance(recovery, dict) and "partial_output" in recovery:
        return bool(recovery["partial_output"]) or bool(turn.output.strip())
    return bool(turn.output.strip())


def session_is_resumable(latest_status: str, turn_count: int) -> bool:
    return turn_count > 0 and latest_status != "idle"


_RELATIONSHIP_KIND_ORDER = ("parent", "child", "fork", "handoff", "delegated")


def _inverse_relationship_kinds(kind: str) -> tuple[str, ...]:
    if kind == "parent":
        return ("child",)
    if kind == "child":
        return ("parent",)
    if kind in {"fork", "handoff", "delegated"}:
        return (kind,)
    raise ValueError(f"unknown session relationship kind: {kind!r}")
