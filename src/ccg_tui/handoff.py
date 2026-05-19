from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ccg_tui.models import BackendName, EventType, SessionRecord, SummaryRecord, TurnRecord, TurnStatus
from ccg_tui.transcript import (
    SelectedTurnContext,
    build_selected_turn_context,
    filter_transcript_turns,
    turn_has_partial_output,
    turn_transcript_state,
)

DEFAULT_HANDOFF_CONTEXT_TURNS = 6
DEFAULT_MAX_HANDOFF_CONTEXT_CHARS = 18_000
DEFAULT_MAX_HANDOFF_TURN_CHARS = 2_400
DEFAULT_MAX_HANDOFF_SUMMARY_CHARS = 6_000


@dataclass(frozen=True, slots=True)
class HandoffConfig:
    recent_turn_limit: int = DEFAULT_HANDOFF_CONTEXT_TURNS
    max_context_chars: int = DEFAULT_MAX_HANDOFF_CONTEXT_CHARS
    max_turn_chars: int = DEFAULT_MAX_HANDOFF_TURN_CHARS
    max_summary_chars: int = DEFAULT_MAX_HANDOFF_SUMMARY_CHARS


@dataclass(frozen=True, slots=True)
class HandoffPacket:
    backend_prompt: str
    context_text: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class HandoffSelectedContext:
    turns: list[TurnRecord]
    rendered_turns: list[str]
    selected_context: SelectedTurnContext


@dataclass(frozen=True, slots=True)
class DelegatedContextPacket:
    backend_prompt: str
    context_text: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DelegatedResultPayload:
    result_text: str
    metadata: dict[str, Any]


def _clip_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    marker = f"\n\n[truncated {len(text) - limit} chars]"
    keep = max(0, limit - len(marker))
    omitted = len(text) - keep
    marker = f"\n\n[truncated {omitted} chars]"
    keep = max(0, limit - len(marker))
    if keep == 0:
        return marker[-limit:]
    return f"{text[:keep].rstrip()}{marker}"


def _scope_name(scope: str, task_id: str | None) -> str:
    if scope == "task":
        if not task_id:
            raise ValueError("task-scoped handoff requires task_id")
        return f"task:{task_id}"
    if scope == "session":
        return "session"
    raise ValueError("handoff scope must be 'task' or 'session'")


def _latest_summary_for_scope(
    session: SessionRecord,
    *,
    scope: str,
    task_id: str | None,
) -> SummaryRecord | None:
    summary_scope = _scope_name(scope, task_id)
    for summary in reversed(session.summaries):
        if summary.scope == summary_scope and summary.text.strip():
            return summary
    return None


def _recent_activity_lines(turn: TurnRecord) -> list[str]:
    return [
        event.text.strip()
        for event in turn.events
        if event.type == EventType.ACTIVITY.value and event.text.strip()
    ][-5:]


def _scope_turns(session: SessionRecord, *, scope: str, task_id: str | None) -> list[TurnRecord]:
    selected_task_id = task_id if scope == "task" else None
    if selected_task_id is None:
        return list(session.turns)
    return [turn for turn in session.turns if turn.task_id == selected_task_id]


def _status_value(status: str | TurnStatus) -> str:
    return status.value if isinstance(status, TurnStatus) else str(status)


def _normalize_turn_statuses(statuses: list[str | TurnStatus] | None) -> list[str]:
    if statuses is None:
        return []
    return [_status_value(status) for status in statuses]


def _build_handoff_audit(
    session: SessionRecord,
    *,
    scope: str,
    task_id: str | None,
    turn_ids: list[str] | None,
    statuses: list[str | TurnStatus] | None,
    recent_turn_limit: int,
    summary: SummaryRecord | None,
    raw_summary_text: str,
    rendered_summary_text: str,
    final_turns: list[TurnRecord],
    context_text_clipped: bool,
    context_text: str,
    backend_prompt: str,
    max_context_chars: int,
    max_turn_chars: int,
    max_summary_chars: int,
) -> dict[str, Any]:
    scope_turns = _scope_turns(session, scope=scope, task_id=task_id)
    selected_task_id = task_id if scope == "task" else None
    criteria_turns = filter_transcript_turns(
        session,
        task_id=selected_task_id,
        turn_ids=turn_ids,
        statuses=statuses,
        recent_count=None,
    )
    criteria_turn_ids = [turn.id for turn in criteria_turns]
    criteria_turn_id_set = set(criteria_turn_ids)
    recent_turns = filter_transcript_turns(
        session,
        task_id=selected_task_id,
        turn_ids=turn_ids,
        statuses=statuses,
        recent_count=recent_turn_limit,
    )
    recent_turn_ids = [turn.id for turn in recent_turns]
    final_turn_ids = [turn.id for turn in final_turns]
    recent_limit_dropped_ids = criteria_turn_ids[: len(criteria_turn_ids) - len(recent_turn_ids)]
    dropped_for_context_limit_ids = recent_turn_ids[: len(recent_turn_ids) - len(final_turn_ids)]
    filtered_by_criteria_ids = [turn.id for turn in scope_turns if turn.id not in criteria_turn_id_set]

    summary_source_id = summary.id if summary is not None else None
    summary_exclusion_reason = None if summary is not None else "summary_absent"
    summary_was_truncated = summary is not None and len(raw_summary_text) != len(rendered_summary_text)

    return {
        "turns": {
            "included_source_ids": final_turn_ids,
            "selected_before_recent_limit_source_ids": criteria_turn_ids,
            "selected_before_context_limit_source_ids": recent_turn_ids,
            "excluded_source_ids": [
                *(
                    {"source_id": turn_id, "reason": "filtered_by_criteria"}
                    for turn_id in filtered_by_criteria_ids
                ),
                *(
                    {"source_id": turn_id, "reason": "recent_turn_limit"}
                    for turn_id in recent_limit_dropped_ids
                ),
                *(
                    {"source_id": turn_id, "reason": "dropped_for_context_limit"}
                    for turn_id in dropped_for_context_limit_ids
                ),
            ],
        },
        "summary": {
            "source_id": summary_source_id,
            "source_scope": summary.scope if summary is not None else _scope_name(scope, task_id),
            "included": summary is not None,
            "exclusion_reason": summary_exclusion_reason,
            "source_text_char_count": len(raw_summary_text),
            "rendered_text_char_count": len(rendered_summary_text),
            "truncated": summary_was_truncated,
        },
        "limits": {
            "scope": _scope_name(scope, task_id),
            "task_id": selected_task_id,
            "recent_turn_limit": recent_turn_limit,
            "max_context_chars": max_context_chars,
            "max_turn_chars": max_turn_chars,
            "max_summary_chars": max_summary_chars,
        },
        "truncation": {
            "context_text_clipped": context_text_clipped,
            "context_char_count": len(context_text),
            "backend_prompt_char_count": len(backend_prompt),
            "summary_text_clipped": summary_was_truncated,
            "summary_source_char_count": len(raw_summary_text),
            "summary_rendered_char_count": len(rendered_summary_text),
            "selected_before_recent_limit_count": len(criteria_turn_ids),
            "selected_before_context_limit_count": len(recent_turn_ids),
            "selected_final_count": len(final_turn_ids),
            "dropped_for_recent_limit_count": len(recent_limit_dropped_ids),
            "dropped_for_context_limit_count": len(dropped_for_context_limit_ids),
        },
    }


def _render_turn(turn: TurnRecord, *, max_turn_chars: int) -> str:
    prompt_limit = max(1, max_turn_chars // 2)
    transcript_state = turn_transcript_state(turn)
    has_partial_output = turn_has_partial_output(turn)
    output_label = "Assistant output:"
    if transcript_state != "completed":
        output_label = (
            "Assistant output (partial, not authoritative):"
            if has_partial_output
            else "Assistant output (not confirmed):"
        )
    parts = [
        f"Turn: {turn.id}",
        f"Backend: {turn.backend.value}",
        f"Status: {transcript_state}",
        f"Task: {turn.task_id}",
        "User prompt:",
        _clip_text(turn.prompt.strip() or "<empty>", prompt_limit),
        output_label,
        _clip_text(turn.output.strip() or "<empty>", max_turn_chars),
    ]
    activity_lines = _recent_activity_lines(turn)
    if activity_lines:
        parts.extend(["Recent activity:", *[f"- {line}" for line in activity_lines]])
    if turn.error is not None:
        parts.extend(
            [
                "Error:",
                f"- kind: {turn.error.kind}",
                f"- message: {turn.error.message}",
            ]
        )
    if transcript_state != "completed":
        parts.extend(
            [
                "Recovery guidance:",
                (
                    "Treat this source turn as unfinished. Do not treat the assistant output as an authoritative completion."
                    if has_partial_output
                    else "Treat this source turn as unfinished. No reliable assistant output was confirmed."
                ),
            ]
        )
    return "\n".join(parts)


def _compose_context_text(
    session: SessionRecord,
    *,
    target_backend: BackendName,
    target_model: str | None,
    user_goal: str,
    scope: str,
    task_id: str | None,
    summary: SummaryRecord | None,
    summary_text: str,
    rendered_turns: list[str],
) -> str:
    if summary is None:
        summary_header = "Latest summary checkpoint: <none>"
        summary_block = "<none>"
    else:
        summary_header = (
            "Latest summary checkpoint: "
            f"{summary.id} ({summary.kind}, {summary.scope}, {summary.created_at})"
        )
        summary_block = summary_text or "<empty>"
    turn_block = "\n\n---\n\n".join(rendered_turns) if rendered_turns else "<none>"
    goal_block = user_goal.strip() or "<unspecified>"
    return f"""CCG MANUAL HANDOFF PACKET

Use this packet as portable context for a new target-backend session. This is not a vendor-native resume. The target backend should treat the user goal as the current task and the packet as prior working context.

Source session:
- session_id: {session.id}
- backend: {session.backend.value}
- workspace_cwd: {session.workspace_cwd or "<unknown>"}
- total_prior_turns: {len(session.turns)}
- total_summaries: {len(session.summaries)}
- source_scope: {_scope_name(scope, task_id)}
- source_task_id: {task_id or "<none>"}

Target intent:
- backend: {target_backend.value}
- model: {target_model or "<default>"}
- user_goal: {goal_block}

{summary_header}
{summary_block}

Recent source turns:
{turn_block}

END CCG MANUAL HANDOFF PACKET"""


def _build_backend_prompt(context_text: str, user_goal: str) -> str:
    return (
        "You are starting a new CCG TUI target-backend session from a manual handoff packet. "
        "Use the packet as prior working context, but follow the current user goal as the active request. "
        "Do not claim vendor-native session continuity. "
        f"Handoff packet JSON: {json.dumps(context_text, ensure_ascii=False)}. "
        f"Current user goal JSON: {json.dumps(user_goal, ensure_ascii=False)}."
    )


def _compose_delegated_context_text(
    session: SessionRecord,
    *,
    target_backend: BackendName,
    target_model: str | None,
    permission_mode: str,
    delegate_goal: str,
    scope: str,
    task_id: str | None,
    summary: SummaryRecord | None,
    summary_text: str,
    rendered_turns: list[str],
) -> str:
    if summary is None:
        summary_header = "Latest summary checkpoint: <none>"
        summary_block = "<none>"
    else:
        summary_header = (
            "Latest summary checkpoint: "
            f"{summary.id} ({summary.kind}, {summary.scope}, {summary.created_at})"
        )
        summary_block = summary_text or "<empty>"
    turn_block = "\n\n---\n\n".join(rendered_turns) if rendered_turns else "<none>"
    goal_block = delegate_goal.strip() or "<unspecified>"
    return f"""CCG DELEGATED CONTEXT PACKET

Use this packet as delegated parent context for a new child CCG TUI session. This is not a vendor-native continuation. The child session should treat the delegated goal as the active task and the packet as curated background from the parent.

Parent session:
- session_id: {session.id}
- backend: {session.backend.value}
- workspace_cwd: {session.workspace_cwd or "<unknown>"}
- total_prior_turns: {len(session.turns)}
- total_summaries: {len(session.summaries)}
- source_scope: {_scope_name(scope, task_id)}
- source_task_id: {task_id or "<none>"}

Child intent:
- backend: {target_backend.value}
- model: {target_model or "<default>"}
- permission_mode: {permission_mode}
- delegated_goal: {goal_block}

{summary_header}
{summary_block}

Selected parent turns:
{turn_block}

END CCG DELEGATED CONTEXT PACKET"""


def _build_delegated_backend_prompt(
    context_text: str,
    *,
    delegate_goal: str,
    permission_mode: str,
) -> str:
    return (
        "You are starting a new delegated CCG TUI child session from explicit parent context. "
        "Treat this as delegated context, not vendor-native continuation. "
        "Follow the delegated goal as the active request and keep the recorded permission mode explicit. "
        f"Delegated context JSON: {json.dumps(context_text, ensure_ascii=False)}. "
        f"Delegated goal JSON: {json.dumps(delegate_goal, ensure_ascii=False)}. "
        f"Permission mode JSON: {json.dumps(permission_mode, ensure_ascii=False)}."
    )


def _find_latest_injected_delegated_context(session: SessionRecord) -> dict[str, Any]:
    for turn in reversed(session.turns):
        metadata = getattr(turn, "metadata", {})
        delegated_context = metadata.get("delegated_context")
        if isinstance(delegated_context, dict) and delegated_context.get("injected"):
            return delegated_context
    return {}


def _require_summary(session: SessionRecord, summary_id: str) -> SummaryRecord:
    for summary in session.summaries:
        if summary.id == summary_id:
            return summary
    raise ValueError(f"unknown summary id: {summary_id}")


def _require_turn(session: SessionRecord, turn_id: str) -> TurnRecord:
    for turn in session.turns:
        if turn.id == turn_id:
            return turn
    raise ValueError(f"unknown turn id: {turn_id}")


def _require_artifact_ids(session: SessionRecord, artifact_ids: list[str]) -> list[str]:
    known_ids = {artifact.id for artifact in session.artifacts}
    missing = [artifact_id for artifact_id in artifact_ids if artifact_id not in known_ids]
    if missing:
        missing_ids = ", ".join(sorted(set(missing)))
        raise ValueError(f"unknown artifact id(s): {missing_ids}")
    return list(artifact_ids)


def build_handoff_selected_context(
    session: SessionRecord,
    *,
    scope: str = "session",
    task_id: str | None = None,
    turn_ids: list[str] | None = None,
    statuses: list[str | TurnStatus] | None = None,
    recent_turn_limit: int = DEFAULT_HANDOFF_CONTEXT_TURNS,
    max_turn_chars: int = DEFAULT_MAX_HANDOFF_TURN_CHARS,
) -> HandoffSelectedContext:
    selected_task_id = task_id if scope == "task" else None
    _scope_name(scope, task_id)
    turns = filter_transcript_turns(
        session,
        task_id=selected_task_id,
        turn_ids=turn_ids,
        statuses=statuses,
        recent_count=recent_turn_limit,
    )
    rendered_turns = [_render_turn(turn, max_turn_chars=max_turn_chars) for turn in turns]
    return HandoffSelectedContext(
        turns=turns,
        rendered_turns=rendered_turns,
        selected_context=build_selected_turn_context(turns, rendered_turns=rendered_turns),
    )


def build_delegated_context_packet(
    session: SessionRecord,
    *,
    target_backend: BackendName,
    delegate_goal: str,
    permission_mode: str,
    target_model: str | None = None,
    scope: str = "session",
    task_id: str | None = None,
    turn_ids: list[str] | None = None,
    statuses: list[str | TurnStatus] | None = None,
    recent_turn_limit: int | None = None,
    config: HandoffConfig | None = None,
) -> DelegatedContextPacket:
    normalized_permission_mode = permission_mode.strip()
    if not normalized_permission_mode:
        raise ValueError("permission_mode is required for delegated context")

    handoff_config = config or HandoffConfig()
    effective_recent_turn_limit = (
        handoff_config.recent_turn_limit if recent_turn_limit is None else recent_turn_limit
    )
    summary = _latest_summary_for_scope(session, scope=scope, task_id=task_id)
    selected_task_id = task_id if scope == "task" else None
    selected = build_handoff_selected_context(
        session,
        scope=scope,
        task_id=task_id,
        turn_ids=turn_ids,
        statuses=statuses,
        recent_turn_limit=effective_recent_turn_limit,
        max_turn_chars=handoff_config.max_turn_chars,
    )
    source_turns = list(selected.turns)
    raw_summary_text = summary.text.strip() if summary is not None else ""
    summary_text = _clip_text(raw_summary_text, handoff_config.max_summary_chars) if summary is not None else ""
    rendered_turns = list(selected.rendered_turns)
    selected_context = selected.selected_context
    context_text = _compose_delegated_context_text(
        session,
        target_backend=target_backend,
        target_model=target_model,
        permission_mode=normalized_permission_mode,
        delegate_goal=delegate_goal,
        scope=scope,
        task_id=selected_task_id,
        summary=summary,
        summary_text=summary_text,
        rendered_turns=rendered_turns,
    )
    context_text_clipped = False
    while len(context_text) > handoff_config.max_context_chars and rendered_turns:
        rendered_turns.pop(0)
        source_turns.pop(0)
        selected_context = build_selected_turn_context(source_turns, rendered_turns=rendered_turns)
        context_text = _compose_delegated_context_text(
            session,
            target_backend=target_backend,
            target_model=target_model,
            permission_mode=normalized_permission_mode,
            delegate_goal=delegate_goal,
            scope=scope,
            task_id=selected_task_id,
            summary=summary,
            summary_text=summary_text,
            rendered_turns=rendered_turns,
        )
        context_text_clipped = True
    if len(context_text) > handoff_config.max_context_chars:
        context_text = _clip_text(context_text, handoff_config.max_context_chars)
        context_text_clipped = True

    backend_prompt = _build_delegated_backend_prompt(
        context_text,
        delegate_goal=delegate_goal,
        permission_mode=normalized_permission_mode,
    )
    forked_from_turn_id = source_turns[-1].id if source_turns else None
    metadata = {
        "mode": "delegated_context",
        "serialized_format": "single_line_json",
        "parent_session_id": session.id,
        "parent_backend": session.backend.value,
        "child_backend": target_backend.value,
        "child_model": target_model,
        "permission_mode": normalized_permission_mode,
        "source_workspace_cwd": session.workspace_cwd,
        "source_scope": _scope_name(scope, task_id),
        "source_task_id": selected_task_id,
        "source_summary_id": summary.id if summary is not None else None,
        "source_turn_ids": [turn.id for turn in source_turns],
        "source_turn_count": len(source_turns),
        "selected_context": {
            "source_turn_ids": list(selected_context.source_turn_ids),
            "turn_count": selected_context.turn_count,
            "prompt_char_count": selected_context.prompt_char_count,
            "output_char_count": selected_context.output_char_count,
            "rendered_char_count": selected_context.rendered_char_count,
        },
        "selection_criteria": {
            "scope": _scope_name(scope, task_id),
            "task_id": selected_task_id,
            "turn_ids": list(turn_ids or []),
            "statuses": _normalize_turn_statuses(statuses),
            "recent_turn_limit": effective_recent_turn_limit,
        },
        "delegate_goal": delegate_goal,
        "forked_from_turn_id": forked_from_turn_id,
        "total_source_turn_count": len(session.turns),
        "context_char_count": len(context_text),
        "backend_prompt_char_count": len(backend_prompt),
        "recent_turn_limit": effective_recent_turn_limit,
        "max_context_chars": handoff_config.max_context_chars,
        "max_turn_chars": handoff_config.max_turn_chars,
        "max_summary_chars": handoff_config.max_summary_chars,
    }
    metadata["audit"] = _build_handoff_audit(
        session,
        scope=scope,
        task_id=task_id,
        turn_ids=turn_ids,
        statuses=statuses,
        recent_turn_limit=effective_recent_turn_limit,
        summary=summary,
        raw_summary_text=raw_summary_text,
        rendered_summary_text=summary_text,
        final_turns=source_turns,
        context_text_clipped=context_text_clipped,
        context_text=context_text,
        backend_prompt=backend_prompt,
        max_context_chars=handoff_config.max_context_chars,
        max_turn_chars=handoff_config.max_turn_chars,
        max_summary_chars=handoff_config.max_summary_chars,
    )
    if summary is not None:
        metadata["source_summary_scope"] = summary.scope
    return DelegatedContextPacket(
        backend_prompt=backend_prompt,
        context_text=context_text,
        metadata=metadata,
    )


def build_handoff_packet(
    session: SessionRecord,
    *,
    target_backend: BackendName,
    user_goal: str = "",
    target_model: str | None = None,
    scope: str = "session",
    task_id: str | None = None,
    turn_ids: list[str] | None = None,
    statuses: list[str | TurnStatus] | None = None,
    recent_turn_limit: int | None = None,
    config: HandoffConfig | None = None,
) -> HandoffPacket:
    handoff_config = config or HandoffConfig()
    effective_recent_turn_limit = (
        handoff_config.recent_turn_limit if recent_turn_limit is None else recent_turn_limit
    )
    summary = _latest_summary_for_scope(session, scope=scope, task_id=task_id)
    selected_task_id = task_id if scope == "task" else None
    selected = build_handoff_selected_context(
        session,
        scope=scope,
        task_id=task_id,
        turn_ids=turn_ids,
        statuses=statuses,
        recent_turn_limit=effective_recent_turn_limit,
        max_turn_chars=handoff_config.max_turn_chars,
    )
    source_turns = list(selected.turns)

    raw_summary_text = summary.text.strip() if summary is not None else ""
    summary_text = _clip_text(raw_summary_text, handoff_config.max_summary_chars) if summary is not None else ""
    rendered_turns = list(selected.rendered_turns)
    selected_context = selected.selected_context
    context_text = _compose_context_text(
        session,
        target_backend=target_backend,
        target_model=target_model,
        user_goal=user_goal,
        scope=scope,
        task_id=selected_task_id,
        summary=summary,
        summary_text=summary_text,
        rendered_turns=rendered_turns,
    )
    context_text_clipped = False
    while len(context_text) > handoff_config.max_context_chars and rendered_turns:
        rendered_turns.pop(0)
        source_turns.pop(0)
        selected_context = build_selected_turn_context(source_turns, rendered_turns=rendered_turns)
        context_text = _compose_context_text(
            session,
            target_backend=target_backend,
            target_model=target_model,
            user_goal=user_goal,
            scope=scope,
            task_id=selected_task_id,
            summary=summary,
            summary_text=summary_text,
            rendered_turns=rendered_turns,
        )
        context_text_clipped = True
    if len(context_text) > handoff_config.max_context_chars:
        context_text = _clip_text(context_text, handoff_config.max_context_chars)
        context_text_clipped = True

    backend_prompt = _build_backend_prompt(context_text, user_goal)
    metadata = {
        "mode": "manual_handoff",
        "serialized_format": "single_line_json",
        "source_session_id": session.id,
        "source_backend": session.backend.value,
        "target_backend": target_backend.value,
        "target_model": target_model,
        "source_workspace_cwd": session.workspace_cwd,
        "source_scope": _scope_name(scope, task_id),
        "source_task_id": selected_task_id,
        "source_summary_id": summary.id if summary is not None else None,
        "source_turn_ids": [turn.id for turn in source_turns],
        "source_turn_count": len(source_turns),
        "selected_context": {
            "source_turn_ids": list(selected_context.source_turn_ids),
            "turn_count": selected_context.turn_count,
            "prompt_char_count": selected_context.prompt_char_count,
            "output_char_count": selected_context.output_char_count,
            "rendered_char_count": selected_context.rendered_char_count,
        },
        "selection_criteria": {
            "scope": _scope_name(scope, task_id),
            "task_id": selected_task_id,
            "turn_ids": list(turn_ids or []),
            "statuses": _normalize_turn_statuses(statuses),
            "recent_turn_limit": effective_recent_turn_limit,
        },
        "total_source_turn_count": len(session.turns),
        "context_char_count": len(context_text),
        "backend_prompt_char_count": len(backend_prompt),
        "recent_turn_limit": effective_recent_turn_limit,
        "max_context_chars": handoff_config.max_context_chars,
        "max_turn_chars": handoff_config.max_turn_chars,
        "max_summary_chars": handoff_config.max_summary_chars,
    }
    metadata["audit"] = _build_handoff_audit(
        session,
        scope=scope,
        task_id=task_id,
        turn_ids=turn_ids,
        statuses=statuses,
        recent_turn_limit=effective_recent_turn_limit,
        summary=summary,
        raw_summary_text=raw_summary_text,
        rendered_summary_text=summary_text,
        final_turns=source_turns,
        context_text_clipped=context_text_clipped,
        context_text=context_text,
        backend_prompt=backend_prompt,
        max_context_chars=handoff_config.max_context_chars,
        max_turn_chars=handoff_config.max_turn_chars,
        max_summary_chars=handoff_config.max_summary_chars,
    )
    if summary is not None:
        metadata["source_summary_scope"] = summary.scope
    return HandoffPacket(
        backend_prompt=backend_prompt,
        context_text=context_text,
        metadata=metadata,
    )


def build_delegated_result_payload(
    session: SessionRecord,
    *,
    result_text: str,
    summary_id: str | None = None,
    turn_id: str | None = None,
    artifact_ids: list[str] | None = None,
) -> DelegatedResultPayload:
    if session.lineage.kind != "delegated":
        raise ValueError("delegated result payload requires a delegated child session")

    normalized_result_text = result_text.strip()
    if not normalized_result_text:
        raise ValueError("delegated result text must not be empty")

    summary = _require_summary(session, summary_id) if summary_id is not None else None
    turn = _require_turn(session, turn_id) if turn_id is not None else None
    resolved_artifact_ids = _require_artifact_ids(session, artifact_ids or [])
    delegated_context = _find_latest_injected_delegated_context(session)

    metadata = {
        "mode": "delegated_result",
        "delegated_session_id": session.id,
        "delegated_lineage_kind": session.lineage.kind,
        "parent_session_id": session.lineage.parent_session_id,
        "delegated_backend": session.backend.value,
        "delegated_model": delegated_context.get("child_model"),
        "permission_mode": delegated_context.get("permission_mode"),
        "source_summary_id": delegated_context.get("source_summary_id"),
        "source_turn_ids": list(delegated_context.get("source_turn_ids", [])),
        "selection_criteria": dict(delegated_context.get("selection_criteria", {})),
        "delegated_summary_id": summary.id if summary is not None else None,
        "delegated_summary_scope": summary.scope if summary is not None else None,
        "delegated_turn_id": turn.id if turn is not None else None,
        "delegated_artifact_ids": resolved_artifact_ids,
    }
    return DelegatedResultPayload(
        result_text=normalized_result_text,
        metadata=metadata,
    )
