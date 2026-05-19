from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ccg_tui.backends.base import BackendAdapter
from ccg_tui.models import BackendEvent, EventType, SessionRecord, SummaryRecord, TurnRecord

DEFAULT_RECENT_TURN_LIMIT = 12
DEFAULT_MAX_PROMPT_CHARS = 24_000
DEFAULT_MAX_TURN_CHARS = 3_000
DEFAULT_MAX_PRIOR_SUMMARY_CHARS = 6_000


class SummaryGenerationError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(UTC).isoformat()


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


def _clip_text_preserving_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = f"[truncated {len(text) - limit} chars from prompt prefix]\n"
    if len(marker) >= limit:
        return text[-limit:]
    return marker + text[-(limit - len(marker)) :]


def _prior_summary_block(text: str, max_prompt_chars: int) -> str:
    if not text.strip():
        return "<none>"
    budget = min(DEFAULT_MAX_PRIOR_SUMMARY_CHARS, max_prompt_chars // 3)
    if budget <= 0:
        return "<omitted to preserve recent source turns>"
    return _clip_text(text.strip(), budget)


def _turns_for_scope(
    session: SessionRecord,
    *,
    scope: str,
    task_id: str,
    recent_turn_limit: int,
) -> list[TurnRecord]:
    turns = list(session.turns)
    if scope == "task":
        turns = [turn for turn in turns if turn.task_id == task_id]
    return turns[-recent_turn_limit:]


def _latest_summary(session: SessionRecord, scope: str) -> SummaryRecord | None:
    for summary in reversed(session.summaries):
        if summary.scope == scope and summary.text.strip():
            return summary
    return None


def _render_turn(turn: TurnRecord, *, max_turn_chars: int) -> str:
    activity_lines = [
        event.text.strip()
        for event in turn.events
        if event.type == EventType.ACTIVITY.value and event.text.strip()
    ]
    parts = [
        f"Turn: {turn.id}",
        f"Backend: {turn.backend.value}",
        f"Status: {turn.status.value}",
        f"Task: {turn.task_id}",
        "Prompt:",
        _clip_text(turn.prompt.strip() or "<empty>", max_turn_chars // 2),
        "Output:",
        _clip_text(turn.output.strip() or "<empty>", max_turn_chars),
    ]
    if activity_lines:
        parts.extend(["Recent activity:", *[f"- {line}" for line in activity_lines[-8:]]])
    if turn.error is not None:
        parts.extend(["Error:", turn.error.message])
    return "\n".join(parts)


def build_summary_prompt(
    session: SessionRecord,
    *,
    scope: str = "task",
    task_id: str = "task-main",
    recent_turn_limit: int = DEFAULT_RECENT_TURN_LIMIT,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    max_turn_chars: int = DEFAULT_MAX_TURN_CHARS,
) -> tuple[str, list[str]]:
    if scope not in {"task", "session"}:
        raise ValueError("summary scope must be 'task' or 'session'")

    summary_scope = f"task:{task_id}" if scope == "task" else "session"
    source_turns = _turns_for_scope(
        session,
        scope=scope,
        task_id=task_id,
        recent_turn_limit=recent_turn_limit,
    )
    latest_summary = _latest_summary(session, summary_scope)
    latest_summary_text = latest_summary.text.strip() if latest_summary is not None else ""
    prior_summary_block = _prior_summary_block(latest_summary_text, max_prompt_chars)
    rendered_turns = [_render_turn(turn, max_turn_chars=max_turn_chars) for turn in source_turns]

    def compose_prompt() -> str:
        source_block = "\n\n---\n\n".join(rendered_turns)
        return f"""You are maintaining a durable project task summary for CCG TUI.

Write a concise checkpoint summary that can be used later for resume or handoff.
Do not summarize every turn mechanically. Capture only durable working context.

Return Markdown with exactly these sections:

## Goal
## Current State
## Decisions
## Relevant Files
## Open Questions
## Next Steps

Rules:
- Prefer concrete implementation facts over vague narrative.
- Preserve unresolved questions and user preferences.
- Mention that no per-turn summary should be generated unless the transcript indicates otherwise.
- If a section has nothing important, write "None".

Session:
- session_id: {session.id}
- backend: {session.backend.value}
- workspace_cwd: {session.workspace_cwd or "<unknown>"}
- total_turns: {len(session.turns)}
- scope: {summary_scope}

Previous summary for this scope:
{prior_summary_block}

Recent source turns:
{source_block or "<no turns>"}
"""

    prompt = compose_prompt()
    while len(prompt) > max_prompt_chars and len(rendered_turns) > 1:
        rendered_turns.pop(0)
        source_turns.pop(0)
        prompt = compose_prompt()
    if len(prompt) > max_prompt_chars and prior_summary_block != "<omitted to preserve recent source turns>":
        prior_summary_block = "<omitted to preserve recent source turns>"
        prompt = compose_prompt()
    if len(prompt) > max_prompt_chars:
        prompt = _clip_text_preserving_tail(prompt, max_prompt_chars)
        visible_turn_ids = {turn.id for turn in source_turns if turn.id in prompt}
        if visible_turn_ids:
            source_turns = [turn for turn in source_turns if turn.id in visible_turn_ids]
    return prompt, [turn.id for turn in source_turns]


def collect_summary_text(events: Iterable[BackendEvent]) -> str:
    output: list[str] = []
    failure_message = ""
    for event in events:
        if event.type is EventType.OUTPUT_DELTA:
            output.append(event.text)
        elif event.type is EventType.BACKEND_FAILED:
            failure_message = event.error.message if event.error is not None else "Summary backend failed"
            break
    if failure_message:
        raise SummaryGenerationError(failure_message)
    text = "".join(output).strip()
    if not text:
        raise SummaryGenerationError("Summary backend returned an empty summary")
    return text


def generate_summary_record(
    session: SessionRecord,
    *,
    adapter: BackendAdapter,
    cwd: Path,
    scope: str = "task",
    task_id: str = "task-main",
    recent_turn_limit: int = DEFAULT_RECENT_TURN_LIMIT,
) -> SummaryRecord:
    summary_scope = f"task:{task_id}" if scope == "task" else "session"
    latest_summary = _latest_summary(session, summary_scope)
    prompt, source_turn_ids = build_summary_prompt(
        session,
        scope=scope,
        task_id=task_id,
        recent_turn_limit=recent_turn_limit,
    )
    text = collect_summary_text(adapter.run(prompt, Path(cwd)))
    return SummaryRecord(
        id=f"summary-{uuid4().hex[:12]}",
        scope=summary_scope,
        created_at=_now(),
        text=text,
        source_turn_ids=source_turn_ids,
        kind=f"{scope}_checkpoint",
        metadata={
            "backend": adapter.name.value,
            "prompt_chars": len(prompt),
            "source_turn_count": len(source_turn_ids),
            "session_id": session.id,
            "summary_scope": summary_scope,
            "task_id": task_id if scope == "task" else None,
            "source_summary_id": latest_summary.id if latest_summary is not None else None,
        },
    )


def persist_summary_record(
    session: SessionRecord,
    summary: SummaryRecord,
    save_session: Callable[[SessionRecord], object],
) -> SummaryRecord:
    session.summaries.append(summary)
    session.updated_at = summary.created_at
    save_session(session)
    return summary


def generate_and_persist_summary(
    session: SessionRecord,
    *,
    adapter: BackendAdapter,
    cwd: Path,
    save_session: Callable[[SessionRecord], object],
    scope: str = "task",
    task_id: str = "task-main",
    recent_turn_limit: int = DEFAULT_RECENT_TURN_LIMIT,
) -> SummaryRecord:
    summary = generate_summary_record(
        session,
        adapter=adapter,
        cwd=cwd,
        scope=scope,
        task_id=task_id,
        recent_turn_limit=recent_turn_limit,
    )
    return persist_summary_record(session, summary, save_session)
