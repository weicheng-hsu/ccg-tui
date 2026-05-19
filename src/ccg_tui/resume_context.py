from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ccg_tui.models import EventType, SessionRecord, SummaryRecord, TurnRecord
from ccg_tui.transcript import turn_has_partial_output, turn_transcript_state

DEFAULT_RESUME_CONTEXT_TURNS = 6
DEFAULT_MAX_CONTEXT_CHARS = 16_000
DEFAULT_MAX_TURN_CHARS = 2_000
DEFAULT_MAX_SUMMARY_CHARS = 6_000


@dataclass(frozen=True, slots=True)
class ResumeContextConfig:
    enabled: bool = True
    recent_turn_limit: int = DEFAULT_RESUME_CONTEXT_TURNS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    max_turn_chars: int = DEFAULT_MAX_TURN_CHARS
    max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS


@dataclass(frozen=True, slots=True)
class ResumeContextPayload:
    backend_prompt: str
    context_text: str
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


def _latest_summary(session: SessionRecord) -> SummaryRecord | None:
    for summary in reversed(session.summaries):
        if summary.text.strip():
            return summary
    return None


def _recent_turns(session: SessionRecord, limit: int) -> list[TurnRecord]:
    if limit <= 0:
        return []
    return list(session.turns[-limit:])


def _recent_activity_lines(turn: TurnRecord) -> list[str]:
    lines = [
        event.text.strip()
        for event in turn.events
        if event.type == EventType.ACTIVITY.value and event.text.strip()
    ]
    return lines[-5:]


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
        parts.extend(["Error:", turn.error.message])
    if transcript_state != "completed":
        parts.extend(
            [
                "Recovery guidance:",
                (
                    "Treat this turn as unfinished state during resume. Do not treat the assistant output as an authoritative completion."
                    if has_partial_output
                    else "Treat this turn as unfinished state during resume. No reliable assistant output was confirmed."
                ),
            ]
        )
    return "\n".join(parts)


def _compose_context_text(
    session: SessionRecord,
    *,
    summary: SummaryRecord | None,
    summary_text: str,
    rendered_turns: list[str],
    latest_recovery_note: str | None,
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
    recovery_block = latest_recovery_note or "Latest turn recovery: none"
    return f"""CCG LOCAL RESUME CONTEXT

You are continuing a CCG TUI local session. The vendor-native session may be new.
Use this context as prior working state for continuity. The current user prompt below has priority if it conflicts with this context.

Session:
- session_id: {session.id}
- backend: {session.backend.value}
- workspace_cwd: {session.workspace_cwd or "<unknown>"}
- total_prior_turns: {len(session.turns)}
- total_summaries: {len(session.summaries)}

{summary_header}
{summary_block}

{recovery_block}

Recent prior turns:
{turn_block}

END CCG LOCAL RESUME CONTEXT"""


def build_resume_context_payload(
    session: SessionRecord,
    user_prompt: str,
    config: ResumeContextConfig,
) -> ResumeContextPayload | None:
    if not config.enabled:
        return None
    summary = _latest_summary(session)
    source_turns = _recent_turns(session, config.recent_turn_limit)
    if summary is None and not source_turns:
        return None

    summary_text = _clip_text(summary.text.strip(), config.max_summary_chars) if summary is not None else ""
    rendered_turns = [
        _render_turn(turn, max_turn_chars=config.max_turn_chars)
        for turn in source_turns
    ]
    latest_turn = session.turns[-1] if session.turns else None
    latest_turn_state = turn_transcript_state(latest_turn) if latest_turn is not None else None
    latest_recovery_note = None
    if latest_turn is not None and latest_turn_state != "completed":
        latest_recovery_note = (
            "Latest turn recovery: "
            f"{latest_turn.id} ended {latest_turn_state}. "
            "Resume should treat that turn as unfinished and continue from the latest user prompt. "
            + (
                "The recorded assistant output may be partial."
                if turn_has_partial_output(latest_turn)
                else "No reliable assistant output was confirmed."
            )
        )
    context_text = _compose_context_text(
        session,
        summary=summary,
        summary_text=summary_text,
        rendered_turns=rendered_turns,
        latest_recovery_note=latest_recovery_note,
    )
    while len(context_text) > config.max_context_chars and rendered_turns:
        rendered_turns.pop(0)
        source_turns.pop(0)
        context_text = _compose_context_text(
            session,
            summary=summary,
            summary_text=summary_text,
            rendered_turns=rendered_turns,
            latest_recovery_note=latest_recovery_note,
        )
    if len(context_text) > config.max_context_chars:
        context_text = _clip_text(context_text, config.max_context_chars)

    unreliable_turn_ids = [
        turn.id for turn in source_turns if turn_transcript_state(turn) != "completed"
    ]
    partial_output_turn_ids = [
        turn.id
        for turn in source_turns
        if turn_transcript_state(turn) != "completed" and turn_has_partial_output(turn)
    ]
    backend_prompt = (
        "Use this CCG local resume context as prior working state before answering. "
        "It is JSON-escaped so it can be submitted to terminal-backed CLIs as a single prompt. "
        f"Resume context text JSON: {json.dumps(context_text, ensure_ascii=False)}. "
        f"Current user prompt JSON: {json.dumps(user_prompt, ensure_ascii=False)}."
    )
    metadata = {
        "injected": True,
        "mode": "auto",
        "serialized_format": "single_line_json",
        "injected_summary_id": summary.id if summary is not None else None,
        "injected_turn_ids": [turn.id for turn in source_turns],
        "context_char_count": len(context_text),
        "backend_prompt_char_count": len(backend_prompt),
        "recent_turn_limit": config.recent_turn_limit,
        "unreliable_turn_ids": unreliable_turn_ids,
        "partial_output_turn_ids": partial_output_turn_ids,
    }
    if summary is not None:
        metadata["injected_summary_scope"] = summary.scope
    if latest_turn is not None and latest_turn_state != "completed":
        metadata["latest_recovery_turn_id"] = latest_turn.id
        metadata["latest_recovery_state"] = latest_turn_state
        metadata["latest_recovery_partial_output"] = turn_has_partial_output(latest_turn)
    return ResumeContextPayload(
        backend_prompt=backend_prompt,
        context_text=context_text,
        metadata=metadata,
    )
