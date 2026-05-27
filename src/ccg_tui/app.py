from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ccg_tui.backends.base import BackendAdapter
from ccg_tui.backends import build_backend
from ccg_tui.backends.antigravity import antigravity_model_options, current_antigravity_model, set_antigravity_model
from ccg_tui.handoff import HandoffPacket, build_handoff_packet
from ccg_tui.models import BackendName, EventType, RoutingDecisionRecord, SessionRecord, TurnRecord, TurnStatus
from ccg_tui.resume_context import DEFAULT_RESUME_CONTEXT_TURNS, ResumeContextConfig
from ccg_tui.routing_capabilities import (
    DEFAULT_PERMISSION_PRESET_KEY,
    PERMISSION_PRESET_SPECS,
    ROUTING_POLICY_REFERENCE,
    all_backend_capabilities,
    backend_capability_facts,
    compare_permission_compatibility,
    permission_state_for_backend,
    permission_values_for_backend,
)
from ccg_tui.session import SessionController
from ccg_tui.slash_commands import (
    SlashCommandAction,
    SlashCommandCompleter,
    format_slash_command_help,
    parse_slash_command,
    ordered_slash_commands_for_palette,
    slash_command_palette_group,
)
from ccg_tui.summary import SummaryGenerationError, generate_and_persist_summary
from ccg_tui.transcript import (
    SessionMetadata,
    TranscriptStore,
    session_is_resumable,
    turn_has_partial_output,
    turn_transcript_state,
)

BACKEND_CHOICES = ("codex", "claude", "gemini", "antigravity")
SCREEN_CLEAR = "\x1b[2J\x1b[H"
ANSI_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"
BACKEND_ANSI = {
    "codex": "\x1b[38;5;45m",
    "claude": "\x1b[38;5;214m",
    "gemini": "\x1b[38;5;141m",
    "antigravity": "\x1b[38;5;82m",
}
BACKEND_GLYPHS = {
    "codex": "C",
    "claude": "K",
    "gemini": "G",
    "antigravity": "A",
}
BACKEND_VENDOR_LABELS = {
    "codex": "OpenAI",
    "claude": "Anthropic",
    "gemini": "Google",
    "antigravity": "Google",
}
SHIFT_ENTER_SEQUENCES = (
    "\x1b[27;2;13~",  # xterm modifyOtherKeys / formatOtherKeys
    "\x1b[13;2u",  # CSI-u / Kitty keyboard protocol
)


@dataclass(frozen=True)
class ModelOption:
    label: str
    value: str | None
    description: str


@dataclass(frozen=True)
class PermissionOption:
    key: str
    label: str
    description: str
    codex_approval_policy: str
    codex_sandbox_mode: str
    claude_permission_mode: str
    gemini_approval_mode: str
    antigravity_permission_mode: str


FALLBACK_MODEL_OPTIONS: dict[str, tuple[ModelOption, ...]] = {
    "codex": (
        ModelOption("Default", None, "Use the Codex CLI default model."),
        ModelOption("GPT-5.5", "gpt-5.5", "Frontier model for complex coding, research, and real-world work."),
        ModelOption("gpt-5.4", "gpt-5.4", "Strong model for everyday coding."),
        ModelOption("GPT-5.4-Mini", "gpt-5.4-mini", "Small, fast, and cost-efficient model for simpler coding tasks."),
        ModelOption("gpt-5.3-codex", "gpt-5.3-codex", "Coding-optimized model."),
        ModelOption("gpt-5.2", "gpt-5.2", "Optimized for professional work and long-running agents."),
    ),
    "claude": (
        ModelOption("Default", None, "Use the Claude Code default model."),
        ModelOption("Sonnet", "sonnet", "Latest Claude Sonnet alias selected by Claude Code."),
        ModelOption("Opus", "opus", "Latest Claude Opus alias selected by Claude Code."),
        ModelOption("Haiku", "haiku", "Latest Claude Haiku alias selected by Claude Code."),
        ModelOption("Sonnet (1M context)", "sonnet[1m]", "Claude Sonnet with extended context when available."),
        ModelOption("Opus (1M context)", "opus[1m]", "Claude Opus with extended context when available."),
        ModelOption("Opus Plan Mode", "opusplan", "Use Opus in plan mode and Sonnet otherwise."),
    ),
    "gemini": (
        ModelOption("Default", None, "Use the Gemini CLI default model."),
        ModelOption("Auto (Gemini 3)", "auto-gemini-3", "Let Gemini CLI route between Gemini 3 Pro and Flash."),
        ModelOption("Auto (Gemini 2.5)", "auto-gemini-2.5", "Let Gemini CLI route between Gemini 2.5 Pro and Flash."),
        ModelOption("Gemini 3.1 Pro Preview", "gemini-3.1-pro-preview", "Preview Gemini 3.1 Pro model when available."),
        ModelOption("Gemini 3 Pro Preview", "gemini-3-pro-preview", "Preview Gemini 3 Pro model."),
        ModelOption("Gemini 3 Flash Preview", "gemini-3-flash-preview", "Preview Gemini 3 Flash model."),
        ModelOption("Gemini 3.1 Flash Lite Preview", "gemini-3.1-flash-lite-preview", "Preview Gemini 3.1 Flash Lite model when available."),
        ModelOption("Gemini 2.5 Pro", "gemini-2.5-pro", "Stable Gemini 2.5 Pro model."),
        ModelOption("Gemini 2.5 Flash", "gemini-2.5-flash", "Stable Gemini 2.5 Flash model."),
        ModelOption("Gemini 2.5 Flash Lite", "gemini-2.5-flash-lite", "Stable Gemini 2.5 Flash Lite model."),
    ),
}

PERMISSION_OPTIONS: tuple[PermissionOption, ...] = tuple(
    PermissionOption(
        key=spec.key,
        label=spec.label,
        description=spec.description,
        codex_approval_policy=spec.codex_approval_policy,
        codex_sandbox_mode=spec.codex_sandbox_mode,
        claude_permission_mode=spec.claude_permission_mode,
        gemini_approval_mode=spec.gemini_approval_mode,
        antigravity_permission_mode=spec.antigravity_permission_mode,
    )
    for spec in PERMISSION_PRESET_SPECS
)
DEFAULT_PERMISSION_KEY = DEFAULT_PERMISSION_PRESET_KEY


def _default_permission_index() -> int:
    return next(
        (index for index, option in enumerate(PERMISSION_OPTIONS) if option.key == DEFAULT_PERMISSION_KEY),
        0,
    )


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def install_prompt_toolkit_shift_enter_sequences() -> None:
    from prompt_toolkit.input import vt100_parser
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.keys import Keys

    changed = False
    for sequence in SHIFT_ENTER_SEQUENCES:
        if ANSI_SEQUENCES.get(sequence) != Keys.ControlM:
            ANSI_SEQUENCES[sequence] = Keys.ControlM
            changed = True
    if changed:
        vt100_parser._IS_PREFIX_OF_LONGER_MATCH_CACHE.clear()


def is_prompt_toolkit_shift_enter_event(event) -> bool:
    return any(
        getattr(key_press, "data", None) in SHIFT_ENTER_SEQUENCES
        for key_press in getattr(event, "key_sequence", ())
    )


def format_turn_summary(turn) -> str:
    lines = [
        f"Backend : {turn.backend.value}",
        f"Status  : {turn.status.value}",
        f"Prompt  : {turn.prompt}",
        f"Output  : {turn.output or '<empty>'}",
    ]
    recovery_status = format_turn_recovery_status(turn)
    if recovery_status is not None:
        lines.append(f"Recovery: {recovery_status}")
    resume_context = getattr(turn, "metadata", {}).get("resume_context", {})
    if isinstance(resume_context, dict) and resume_context.get("injected"):
        summary_id = resume_context.get("injected_summary_id") or "<none>"
        turn_count = len(resume_context.get("injected_turn_ids", []))
        lines.append(f"Context : injected summary={summary_id} turns={turn_count}")
    if turn.error is not None:
        lines.append(f"Error   : {turn.error.message}")
    return "\n".join(lines)


def format_summary_record(summary) -> str:
    lines = [
        f"Summary : {summary.id}",
        f"Scope   : {summary.scope}",
        f"Kind    : {summary.kind}",
        f"Backend : {summary.metadata.get('backend', 'unknown')}",
        "",
        summary.text or "<empty>",
    ]
    return "\n".join(lines)


def format_handoff_packet(
    packet: HandoffPacket,
    *,
    source_permission_values: dict[str, str] | None = None,
) -> str:
    turn_ids = packet.metadata.get("source_turn_ids", [])
    rendered_turn_ids = ", ".join(turn_ids) if turn_ids else "<none>"
    target_model = packet.metadata.get("target_model") or "<default>"
    target_backend = packet.metadata.get("target_backend") or "<unknown>"
    source_backend = packet.metadata.get("source_backend") or "<unknown>"
    summary_id = packet.metadata.get("source_summary_id") or "<none>"
    source_scope = packet.metadata.get("source_scope") or "session"
    selected_context = packet.metadata.get("selected_context", {})
    selected_ids = selected_context.get("source_turn_ids", [])
    rendered_selected_ids = ", ".join(selected_ids) if selected_ids else "<none>"
    selection_criteria = packet.metadata.get("selection_criteria", {})
    selected_statuses = selection_criteria.get("statuses", [])
    rendered_statuses = ", ".join(selected_statuses) if selected_statuses else "<none>"
    audit = packet.metadata.get("audit", {})
    audit_turns = audit.get("turns", {}) if isinstance(audit, dict) else {}
    audit_summary = audit.get("summary", {}) if isinstance(audit, dict) else {}
    audit_limits = audit.get("limits", {}) if isinstance(audit, dict) else {}
    audit_truncation = audit.get("truncation", {}) if isinstance(audit, dict) else {}
    audit_included_ids = audit_turns.get("included_source_ids", selected_ids)
    rendered_audit_included_ids = ", ".join(audit_included_ids) if audit_included_ids else "<none>"
    audit_selected_before_recent = audit_turns.get("selected_before_recent_limit_source_ids", [])
    rendered_audit_selected_before_recent = ", ".join(audit_selected_before_recent) if audit_selected_before_recent else "<none>"
    audit_selected_before_context = audit_turns.get("selected_before_context_limit_source_ids", [])
    rendered_audit_selected_before_context = ", ".join(audit_selected_before_context) if audit_selected_before_context else "<none>"
    audit_excluded = audit_turns.get("excluded_source_ids", [])
    rendered_audit_excluded = ", ".join(
        f"{item.get('source_id', '<unknown>')}={item.get('reason', 'unknown')}"
        for item in audit_excluded
        if isinstance(item, dict)
    ) if audit_excluded else "<none>"
    audit_summary_source_id = audit_summary.get("source_id") or "<none>"
    audit_summary_reason = (
        "included" if audit_summary.get("included") else audit_summary.get("exclusion_reason") or "<none>"
    )
    recent_limit = audit_limits.get("recent_turn_limit", selection_criteria.get("recent_turn_limit", "<default>"))
    max_context_chars = audit_limits.get("max_context_chars", packet.metadata.get("max_context_chars", "<unknown>"))
    max_turn_chars = audit_limits.get("max_turn_chars", packet.metadata.get("max_turn_chars", "<unknown>"))
    max_summary_chars = audit_limits.get("max_summary_chars", packet.metadata.get("max_summary_chars", "<unknown>"))
    recent_dropped = audit_truncation.get("dropped_for_recent_limit_count", 0)
    context_dropped = audit_truncation.get("dropped_for_context_limit_count", 0)
    context_clipped = "yes" if audit_truncation.get("context_text_clipped") else "no"
    summary_clipped = "yes" if audit_truncation.get("summary_text_clipped") else "no"
    permission_compatibility = (
        format_permission_compatibility_line(source_backend, source_permission_values, target_backend)
        if source_permission_values is not None
        else "source permission state unavailable; no target permission change is automatic"
    )
    lines = [
        "Handoff Packet",
        f"Source Session : {packet.metadata.get('source_session_id')}",
        f"Source Backend : {source_backend}",
        f"Source Scope   : {source_scope}",
        f"Target Backend : {target_backend}",
        f"Target Model   : {target_model}",
        f"Compatibility  : {permission_compatibility}",
        f"Summary        : {summary_id}",
        f"Source Turns   : {rendered_turn_ids}",
        f"Included IDs   : summary={summary_id} turns={rendered_selected_ids}",
        f"Criteria       : task={selection_criteria.get('task_id') or '<none>'} statuses={rendered_statuses} recent={selection_criteria.get('recent_turn_limit', '<default>')}",
        "Audit",
        f"  Turns included : {rendered_audit_included_ids}",
        f"  Turns before recent limit : {rendered_audit_selected_before_recent}",
        f"  Turns before context limit : {rendered_audit_selected_before_context}",
        f"  Turn exclusions : {rendered_audit_excluded}",
        f"  Summary        : {audit_summary_source_id} ({audit_summary_reason})",
        f"  Limits         : recent={recent_limit} context={max_context_chars} turn={max_turn_chars} summary={max_summary_chars}",
        f"  Truncation     : recent_drop={recent_dropped} context_drop={context_dropped} context_clipped={context_clipped} summary_clipped={summary_clipped}",
        f"Context Chars  : {packet.metadata.get('context_char_count', 0)}",
        f"Backend Prompt : {packet.metadata.get('backend_prompt_char_count', 0)} chars",
        "",
        "Context Text",
        packet.context_text,
        "",
        "Backend Prompt",
        packet.backend_prompt,
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class HandoffPreviewData:
    ok: bool
    preview: str
    packet: HandoffPacket | None = None


def _build_handoff_preview_data(
    session: SessionRecord,
    args: str,
    *,
    scope: str | None = None,
    task_id: str | None = None,
    source_permission_values: dict[str, str] | None = None,
) -> HandoffPreviewData:
    try:
        parts = shlex.split(args)
    except ValueError as exc:
        return HandoffPreviewData(False, f"Handoff preview error: {exc}")
    if not parts:
        return HandoffPreviewData(False, "Usage: /handoff <target-backend> [target-model] [goal...]")
    try:
        parsed = parse_handoff_args(parts)
    except ValueError as exc:
        return HandoffPreviewData(False, f"Handoff preview error: {exc}")
    target_backend = normalize_backend_choice(parsed["target_backend"])
    if target_backend is None:
        return HandoffPreviewData(False, f"Unsupported target backend: {parsed['target_backend']}")
    target_model = parsed["target_model"]
    user_goal = parsed["user_goal"]
    if parsed["task_id"] is not None:
        resolved_scope, resolved_task_id = "task", parsed["task_id"]
    elif scope is None:
        resolved_scope, resolved_task_id = default_handoff_scope(session)
    else:
        resolved_scope, resolved_task_id = scope, task_id
    packet = build_handoff_packet(
        session,
        target_backend=BackendName(target_backend),
        target_model=target_model,
        user_goal=user_goal,
        scope=resolved_scope,
        task_id=resolved_task_id,
        turn_ids=parsed["turn_ids"] or None,
        statuses=parsed["statuses"] or None,
        recent_turn_limit=parsed["recent_turn_limit"],
    )
    return HandoffPreviewData(True, format_handoff_packet(packet, source_permission_values=source_permission_values), packet)


def _format_handoff_status_line(
    packet: HandoffPacket,
    *,
    source_permission_values: dict[str, str] | None = None,
) -> str:
    source = packet.metadata.get("source_session_id", "<unknown>")
    target = packet.metadata.get("target_backend", "<unknown>")
    model = packet.metadata.get("target_model") or "<default>"
    summary = packet.metadata.get("source_summary_id") or "<none>"
    if source_permission_values is not None:
        compatibility = format_permission_compatibility_line(
            packet.metadata.get("source_backend", "<unknown>"),
            source_permission_values,
            str(target),
        )
    else:
        compatibility = "source permission state unavailable; no target permission change is automatic"
    audit = packet.metadata.get("audit", {})
    audit_turns = audit.get("turns", {}) if isinstance(audit, dict) else {}
    included = ", ".join(audit_turns.get("included_source_ids", [])) or "<none>"
    excluded = audit_turns.get("excluded_source_ids", [])
    omitted = ", ".join(
        f"{item.get('source_id', '<unknown>')}={item.get('reason', 'unknown')}"
        for item in excluded
        if isinstance(item, dict)
    ) if excluded else "<none>"
    truncation = audit.get("truncation", {}) if isinstance(audit, dict) else {}
    context_clipped = "yes" if truncation.get("context_text_clipped") else "no"
    summary_clipped = "yes" if truncation.get("summary_text_clipped") else "no"
    return (
        f"Handoff preview: source={source} target={target} model={model} summary={summary} "
        f"compatibility={compatibility} included={included} omitted={omitted} "
        f"truncation=context_clipped={context_clipped} summary_clipped={summary_clipped} "
        f"chars={packet.metadata.get('context_char_count', 0)}"
    )


def _latest_session_task(session: SessionRecord, *, status: str) -> str | None:
    for task in reversed(session.tasks):
        if task.id != "task-main" and task.status == status:
            return task.id
    return None


def default_handoff_scope(session: SessionRecord) -> tuple[str, str | None]:
    task_id = _latest_session_task(session, status="active")
    if task_id is not None:
        return "task", task_id
    task_id = _latest_session_task(session, status="closed")
    if task_id is not None:
        return "task", task_id
    return "session", None


def build_handoff_preview(
    session: SessionRecord,
    args: str,
    *,
    scope: str | None = None,
    task_id: str | None = None,
    source_permission_values: dict[str, str] | None = None,
) -> tuple[bool, str]:
    preview_data = _build_handoff_preview_data(
        session,
        args,
        scope=scope,
        task_id=task_id,
        source_permission_values=source_permission_values,
    )
    return preview_data.ok, preview_data.preview


def split_csv_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    for value in values:
        for part in value.split(","):
            item = part.strip()
            if item:
                normalized.append(item)
    return normalized


def parse_handoff_args(parts: list[str]) -> dict[str, str | list[str] | int | None]:
    if not parts:
        raise ValueError("missing target backend")
    target_backend = parts[0]
    index = 1
    target_model: str | None = None
    if index < len(parts) and not parts[index].startswith("--"):
        target_model = parts[index]
        index += 1
    user_goal_parts: list[str] = []
    task_id: str | None = None
    turn_ids: list[str] = []
    statuses: list[str] = []
    recent_turn_limit: int | None = None
    while index < len(parts):
        token = parts[index]
        if token == "--task-id":
            index += 1
            if index >= len(parts):
                raise ValueError("--task-id requires a value")
            task_id = parts[index].strip() or None
        elif token == "--turn-id":
            index += 1
            if index >= len(parts):
                raise ValueError("--turn-id requires a value")
            turn_ids.extend(split_csv_values([parts[index]]))
        elif token == "--status":
            index += 1
            if index >= len(parts):
                raise ValueError("--status requires a value")
            statuses.extend(split_csv_values([parts[index]]))
        elif token == "--recent":
            index += 1
            if index >= len(parts):
                raise ValueError("--recent requires a value")
            try:
                recent_turn_limit = int(parts[index])
            except ValueError as exc:
                raise ValueError("--recent must be an integer") from exc
            if recent_turn_limit < 0:
                raise ValueError("--recent must be >= 0")
        elif token.startswith("--"):
            raise ValueError(f"unknown handoff option: {token}")
        else:
            user_goal_parts.append(token)
        index += 1
    return {
        "target_backend": target_backend,
        "target_model": target_model,
        "user_goal": " ".join(user_goal_parts),
        "task_id": task_id,
        "turn_ids": turn_ids,
        "statuses": statuses,
        "recent_turn_limit": recent_turn_limit,
    }


def handoff_status_message(
    session: SessionRecord,
    args: str,
    *,
    source_permission_values: dict[str, str] | None = None,
) -> str:
    preview_data = _build_handoff_preview_data(
        session,
        args,
        source_permission_values=source_permission_values,
    )
    if not preview_data.ok:
        return preview_data.preview
    assert preview_data.packet is not None
    return _format_handoff_status_line(
        preview_data.packet,
        source_permission_values=source_permission_values,
    )


def _routing_decision_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _handoff_routing_target(args: str) -> tuple[BackendName | None, str | None, dict]:
    metadata: dict = {}
    try:
        parts = shlex.split(args)
        parsed = parse_handoff_args(parts)
    except ValueError as exc:
        metadata["parse_error"] = str(exc)
        return None, None, metadata
    target_backend = normalize_backend_choice(str(parsed["target_backend"]))
    metadata["raw_target_backend"] = parsed["target_backend"]
    metadata["user_goal"] = parsed["user_goal"]
    metadata["selection_criteria"] = {
        "task_id": parsed["task_id"],
        "turn_ids": parsed["turn_ids"],
        "statuses": parsed["statuses"],
        "recent_turn_limit": parsed["recent_turn_limit"],
    }
    if target_backend is None:
        metadata["parse_error"] = f"unsupported target backend: {parsed['target_backend']}"
        return None, parsed["target_model"] if isinstance(parsed["target_model"], str) else None, metadata
    target_model = parsed["target_model"] if isinstance(parsed["target_model"], str) else None
    return BackendName(target_backend), target_model, metadata


def _routing_permission_payload(
    *,
    active_backend: BackendName,
    suggested_backend: BackendName | None,
    source_permission_values: dict[str, str] | None,
    target_permission_values: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    if source_permission_values is None:
        permission_state = {
            "backend": active_backend.value,
            "status": "unavailable",
        }
        return permission_state, {}
    source_state = permission_state_for_backend(active_backend, source_permission_values)
    if suggested_backend is None:
        return source_state.to_dict(), {}
    compatibility = compare_permission_compatibility(
        active_backend,
        source_permission_values,
        suggested_backend,
        target_values=target_permission_values,
    )
    return source_state.to_dict(), compatibility.to_dict()


def build_handoff_routing_decision(
    session: SessionRecord,
    args: str,
    *,
    source_permission_values: dict[str, str] | None,
    user_decision: str,
    final_action: str,
    reason: str = "",
) -> RoutingDecisionRecord:
    suggested_backend, suggested_model, metadata = _handoff_routing_target(args)
    permission_state, compatibility = _routing_permission_payload(
        active_backend=session.backend,
        suggested_backend=suggested_backend,
        source_permission_values=source_permission_values,
    )
    resolved_reason = reason or metadata.get("parse_error", "") or "manual handoff route inspected"
    return RoutingDecisionRecord(
        id=f"routing-{uuid4().hex[:12]}",
        recorded_at=_routing_decision_timestamp(),
        active_backend=session.backend,
        suggested_backend=suggested_backend,
        suggested_model=suggested_model,
        trigger="manual_handoff",
        policy_reference=ROUTING_POLICY_REFERENCE,
        permission_state=permission_state,
        user_decision=user_decision,
        final_action=final_action,
        reason=resolved_reason,
        compatibility=compatibility,
        metadata=metadata,
    )


def record_controller_handoff_routing_decision(
    controller: SessionController,
    args: str,
    *,
    source_permission_values: dict[str, str] | None,
    user_decision: str,
    final_action: str,
    reason: str = "",
) -> RoutingDecisionRecord | None:
    if not hasattr(controller, "record_routing_decision"):
        return None
    decision = build_handoff_routing_decision(
        controller.session,
        args,
        source_permission_values=source_permission_values,
        user_decision=user_decision,
        final_action=final_action,
        reason=reason,
    )
    return controller.record_routing_decision(
        active_backend=decision.active_backend,
        suggested_backend=decision.suggested_backend,
        suggested_model=decision.suggested_model,
        trigger=decision.trigger,
        policy_reference=decision.policy_reference,
        permission_state=decision.permission_state,
        user_decision=decision.user_decision,
        final_action=decision.final_action,
        reason=decision.reason,
        compatibility=decision.compatibility,
        metadata=decision.metadata,
    )


def record_capability_inspection(controller: SessionController) -> RoutingDecisionRecord | None:
    if not hasattr(controller, "record_routing_decision"):
        return None
    return controller.record_routing_decision(
        trigger="capability_registry_inspected",
        user_decision="deferred",
        final_action="capabilities_displayed",
        permission_state=current_permission_state(controller),
        reason="user inspected advisory backend capabilities",
        metadata={"policy_invariant": "no automatic backend switch"},
    )


def build_packet_routing_decision(
    source_session: SessionRecord,
    packet: HandoffPacket,
    *,
    source_permission_values: dict[str, str] | None,
    target_permission_values: dict[str, str] | None = None,
    user_decision: str,
    final_action: str,
    reason: str,
    metadata: dict | None = None,
) -> RoutingDecisionRecord:
    target_backend = BackendName(packet.metadata["target_backend"])
    permission_state, compatibility = _routing_permission_payload(
        active_backend=source_session.backend,
        suggested_backend=target_backend,
        source_permission_values=source_permission_values,
        target_permission_values=target_permission_values,
    )
    return RoutingDecisionRecord(
        id=f"routing-{uuid4().hex[:12]}",
        recorded_at=_routing_decision_timestamp(),
        active_backend=source_session.backend,
        suggested_backend=target_backend,
        suggested_model=packet.metadata.get("target_model"),
        trigger="manual_handoff",
        policy_reference=ROUTING_POLICY_REFERENCE,
        permission_state=permission_state,
        user_decision=user_decision,
        final_action=final_action,
        reason=reason,
        compatibility=compatibility,
        metadata={
            "source_session_id": source_session.id,
            "source_turn_ids": packet.metadata.get("source_turn_ids", []),
            "target_backend": target_backend.value,
            "policy_invariant": "no automatic backend switch",
            **dict(metadata or {}),
        },
    )


def persist_session_routing_decision(
    store: TranscriptStore,
    session: SessionRecord,
    decision: RoutingDecisionRecord,
) -> RoutingDecisionRecord:
    session.routing_decisions.append(decision)
    session.updated_at = decision.recorded_at
    store.save_session(session)
    return decision


def format_handoff_execution_confirmation(packet: HandoffPacket, *, confirmation_method: str) -> str:
    target_model = packet.metadata.get("target_model") or "<default>"
    summary_id = packet.metadata.get("source_summary_id") or "<none>"
    source_scope = packet.metadata.get("source_scope") or "session"
    source_turn_ids = packet.metadata.get("source_turn_ids", [])
    turns = ", ".join(source_turn_ids) if source_turn_ids else "<none>"
    audit = packet.metadata.get("audit", {})
    audit_turns = audit.get("turns", {}) if isinstance(audit, dict) else {}
    excluded = audit_turns.get("excluded_source_ids", []) if isinstance(audit_turns, dict) else []
    exclusions = ", ".join(
        f"{item.get('source_id', '<unknown>')}={item.get('reason', 'unknown')}"
        for item in excluded
        if isinstance(item, dict)
    ) if excluded else "<none>"
    truncation = audit.get("truncation", {}) if isinstance(audit, dict) else {}
    context_clipped = "yes" if truncation.get("context_text_clipped") else "no"
    summary_clipped = "yes" if truncation.get("summary_text_clipped") else "no"
    return "\n".join(
        [
            f"Handoff execution confirmed by {confirmation_method}",
            f"Source Session : {packet.metadata.get('source_session_id')}",
            f"Source Scope   : {source_scope}",
            f"Target Backend : {packet.metadata.get('target_backend')}",
            f"Target Model   : {target_model}",
            f"Summary        : {summary_id}",
            f"Included Turns : {turns}",
            f"Exclusions     : {exclusions}",
            f"Truncation     : context_clipped={context_clipped} summary_clipped={summary_clipped}",
        ]
    )


def execute_handoff_packet(
    *,
    adapter: BackendAdapter,
    store: TranscriptStore,
    cwd: Path,
    source_session: SessionRecord,
    packet: HandoffPacket,
    user_goal: str,
    confirmation_method: str,
    source_permission_values: dict[str, str] | None = None,
) -> tuple[SessionController, TurnRecord]:
    controller = SessionController.from_handoff(
        adapter=adapter,
        store=store,
        cwd=cwd,
        source_session=source_session,
        handoff_packet=packet,
    )
    target_backend = BackendName(packet.metadata["target_backend"])
    permission_state, compatibility = _routing_permission_payload(
        active_backend=source_session.backend,
        suggested_backend=target_backend,
        source_permission_values=source_permission_values,
        target_permission_values=current_permission_values(controller),
    )
    controller.record_routing_decision(
        active_backend=source_session.backend,
        suggested_backend=target_backend,
        suggested_model=packet.metadata.get("target_model"),
        trigger="manual_handoff",
        policy_reference=ROUTING_POLICY_REFERENCE,
        permission_state=permission_state,
        user_decision="confirmed",
        final_action="handoff_session_started",
        reason=f"manual handoff execution confirmed by {confirmation_method}",
        compatibility=compatibility,
        metadata={
            "source_session_id": source_session.id,
            "target_session_id": controller.session.id,
            "confirmation_method": confirmation_method,
            "source_turn_ids": packet.metadata.get("source_turn_ids", []),
            "policy_invariant": "no automatic backend switch",
        },
    )
    try:
        turn = controller.submit_prompt(
            user_goal,
            backend_prompt=packet.backend_prompt,
            metadata={
                "handoff": {
                    **packet.metadata,
                    "injected": True,
                    "execution_confirmed": True,
                    "confirmation_method": confirmation_method,
                    "source_context_char_count": len(packet.context_text),
                }
            },
        )
    except Exception:
        controller.close()
        raise
    return controller, turn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCG TUI")
    parser.add_argument("--backend", choices=BACKEND_CHOICES)
    parser.add_argument("--prompt")
    parser.add_argument("--transcript-dir", default="runtime/transcripts")
    parser.add_argument("--simple-ui", action="store_true", help="Use the line-by-line fallback UI instead of fullscreen prompt_toolkit mode")
    session_actions = parser.add_mutually_exclusive_group()
    session_actions.add_argument("--summarize-session", help="Generate a summary checkpoint for an existing transcript session id")
    session_actions.add_argument("--list-sessions", action="store_true", help="List local transcript sessions")
    session_actions.add_argument("--resume-session", help="Resume a local transcript session")
    session_actions.add_argument("--handoff-session", help="Preview or export a manual handoff packet for an existing transcript session id")
    parser.add_argument("--target-backend", help="Target backend for --handoff-session: codex, claude, gemini, or antigravity")
    parser.add_argument("--target-model", help="Optional target model to record in the handoff packet")
    parser.add_argument("--handoff-goal", default="", help="Current user goal to include in the handoff packet")
    parser.add_argument("--handoff-output", help="Optional file path to write the handoff preview instead of printing it")
    parser.add_argument("--handoff-execute", action="store_true", help="Explicitly confirm and start a new target-backend session from --handoff-session instead of only previewing")
    parser.add_argument(
        "--handoff-task-id",
        help="Optional task id to scope handoff source context",
    )
    parser.add_argument(
        "--handoff-turn-id",
        action="append",
        default=[],
        help="Optional source turn id(s) to include (repeat flag or pass comma-separated ids)",
    )
    parser.add_argument(
        "--handoff-status",
        action="append",
        default=[],
        help="Optional source turn status filter(s) (repeat flag or pass comma-separated statuses)",
    )
    parser.add_argument(
        "--handoff-recent",
        type=non_negative_int,
        help="Optional recent turn count applied after other handoff filters",
    )
    parser.add_argument("--resume-context", choices=("auto", "off"), default="auto")
    parser.add_argument("--resume-context-turns", type=non_negative_int, default=DEFAULT_RESUME_CONTEXT_TURNS)
    parser.add_argument("--summary-backend", choices=("gemini", "antigravity"), default="gemini")
    parser.add_argument("--summary-scope", choices=("task", "session"), default="task")
    parser.add_argument("--summary-task-id", default="task-main")
    return parser


def normalize_backend_choice(choice: str) -> str | None:
    mapping = {"1": "codex", "2": "claude", "3": "gemini", "4": "antigravity"}
    normalized = choice.strip().lower()
    backend = mapping.get(normalized, normalized)
    return backend if backend in BACKEND_CHOICES else None


def build_backend_picker_lines(selected_backend: str = "codex") -> list[str]:
    return [
        "Select a backend",
        "",
        f"  1. codex{'   <' if selected_backend == 'codex' else ''}",
        f"  2. claude{'  <' if selected_backend == 'claude' else ''}",
        f"  3. gemini{'  <' if selected_backend == 'gemini' else ''}",
        f"  4. antigravity{'  <' if selected_backend == 'antigravity' else ''}",
        "",
        f"> {selected_backend}",
        "",
        "Enter choose backend",
        "↑/↓ move",
        "Esc quit",
    ]


def choose_backend(
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> str:
    print_fn("Select backend:")
    print_fn("  1) codex")
    print_fn("  2) claude")
    print_fn("  3) gemini")
    print_fn("  4) antigravity")
    while True:
        answer = input_fn("backend> ")
        backend = normalize_backend_choice(answer)
        if backend is not None:
            return backend
        print_fn("Invalid selection. Choose 1/2/3/4 or codex/claude/gemini/antigravity.")


def default_controller_factory(transcript_dir: str, cwd: Path) -> Callable[[str], SessionController]:
    def factory(backend: str) -> SessionController:
        store = TranscriptStore(cwd / transcript_dir)
        return SessionController(adapter=build_backend(backend), store=store, cwd=cwd)

    return factory


def resume_controller_factory(
    store: TranscriptStore,
    session: SessionRecord,
    cwd: Path,
    resume_context_config: ResumeContextConfig,
) -> Callable[[str], SessionController]:
    def factory(backend: str) -> SessionController:
        resume_cwd = Path(session.workspace_cwd) if session.workspace_cwd else cwd
        return SessionController.resume(
            adapter=build_backend(backend),
            store=store,
            cwd=resume_cwd,
            session=session,
            resume_context_config=resume_context_config,
        )

    return factory


def format_session_list(sessions: list[SessionMetadata]) -> str:
    if not sessions:
        return "No sessions found."
    headers = [
        "session_id",
        "backend",
        "updated_at",
        "created_at",
        "turn_count",
        "summary_count",
        "latest_status",
        "resumable",
        "workspace_basename",
    ]
    rows = [
        [
            session.id,
            session.backend,
            session.updated_at,
            session.created_at,
            str(session.turn_count),
            str(session.summary_count),
            session.latest_status,
            "yes" if session.resumable else "no",
            session.workspace_basename,
        ]
        for session in sessions
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def build_summary_backend(name: str = "gemini"):
    if name not in {"gemini", "antigravity"}:
        raise ValueError("Only Gemini and Antigravity summary backends are implemented")
    return build_backend(name)


def current_model(controller: SessionController) -> str | None:
    if controller.session.backend is BackendName.ANTIGRAVITY:
        try:
            return current_antigravity_model()
        except ValueError:
            return None
    adapter = getattr(controller, "adapter", None)
    model = getattr(adapter, "model", None)
    return model if isinstance(model, str) and model else None


def current_model_label(controller: SessionController) -> str:
    return current_model(controller) or "default"


def current_permission_values(controller: SessionController) -> dict[str, str]:
    adapter = getattr(controller, "adapter", None)
    backend = controller.session.backend.value
    if backend == "codex":
        return {
            "approval_policy": getattr(adapter, "approval_policy", "on-request"),
            "sandbox_mode": getattr(adapter, "sandbox_mode", "workspace-write"),
        }
    if backend == "claude":
        return {"permission_mode": getattr(adapter, "permission_mode", "default")}
    if backend == "gemini":
        return {"approval_mode": getattr(adapter, "approval_mode", "default")}
    if backend == "antigravity":
        return {"permission_mode": getattr(adapter, "permission_mode", "default")}
    return {}


def permission_kwargs_for_backend(controller: SessionController) -> dict[str, str]:
    return current_permission_values(controller)


def permission_option_values_for_backend(option: PermissionOption, backend: str) -> dict[str, str]:
    return permission_values_for_backend(option.key, backend)


def current_permission_option(controller: SessionController) -> PermissionOption | None:
    backend = controller.session.backend.value
    current = current_permission_values(controller)
    for option in PERMISSION_OPTIONS:
        if permission_option_values_for_backend(option, backend) == current:
            return option
    return None


def current_permission_label(controller: SessionController) -> str:
    option = current_permission_option(controller)
    if option is not None:
        return option.label
    values = current_permission_values(controller)
    return ", ".join(f"{key}={value}" for key, value in values.items()) or "default"


def current_permission_state(controller: SessionController) -> dict:
    return permission_state_for_backend(
        controller.session.backend,
        current_permission_values(controller),
    ).to_dict()


def default_permission_values_for_backend(backend: BackendName | str) -> dict[str, str]:
    return permission_values_for_backend(DEFAULT_PERMISSION_KEY, backend)


def format_permission_compatibility_line(source_backend: str, source_values: dict[str, str], target_backend: str) -> str:
    compatibility = compare_permission_compatibility(source_backend, source_values, target_backend)
    source_label = compatibility.source_state.label
    target_label = compatibility.target_state.label
    target_values = ", ".join(
        f"{key}={value}"
        for key, value in compatibility.target_state.values.items()
    )
    prefix = "blocked" if compatibility.widens_permissions else "compatible"
    warning = f"; {'; '.join(compatibility.warnings)}" if compatibility.warnings else ""
    return (
        f"{prefix}: {source_label} -> {target_label}"
        f" ({target_values}); confirmation required{warning}"
    )


def format_capability_registry(controller: SessionController) -> str:
    source_backend = controller.session.backend.value
    source_values = current_permission_values(controller)
    source_state = permission_state_for_backend(source_backend, source_values)
    source_rendered_values = ", ".join(f"{key}={value}" for key, value in source_state.values.items())
    lines = [
        "Routing Capability Registry",
        f"Policy : {ROUTING_POLICY_REFERENCE}",
        f"Active : {source_backend}",
        f"Perms  : {source_state.label} ({source_rendered_values})",
        "Action : advisory only; backend, model, and permission changes require explicit confirmation.",
        "",
    ]
    for profile in all_backend_capabilities():
        lines.extend(
            [
                profile.display_name,
                f"  Backend : {profile.backend.value}",
                f"  Summary : {profile.summary}",
                f"  Strengths: {', '.join(profile.strengths)}",
                f"  Limits  : {', '.join(profile.limitations)}",
                f"  Triggers: {', '.join(profile.routing_triggers)}",
            ]
        )
        for fact in backend_capability_facts(profile.backend):
            marker = "yes" if fact.supported else "no"
            lines.append(f"  {fact.label}: {marker} - {fact.explanation}")
        if profile.backend.value == source_backend:
            lines.append("  Compatibility: active backend")
        else:
            lines.append(
                "  Compatibility: "
                + format_permission_compatibility_line(source_backend, source_values, profile.backend.value)
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def compact_routing_advisory() -> str:
    return "advisory only; /capabilities for registry"


def _session_continuation_state(controller: SessionController) -> tuple[int, str, bool]:
    turns = controller.session.turns
    active_turn = getattr(controller, "active_turn", None)
    latest_turn = _latest_status_turn(controller)
    latest_status = turn_transcript_state(latest_turn) if latest_turn is not None else "idle"
    turn_count = len(turns)
    if active_turn is not None and active_turn not in turns:
        turn_count += 1
    return turn_count, latest_status, session_is_resumable(latest_status, turn_count)


def _latest_status_turn(controller: SessionController) -> TurnRecord | None:
    active_turn = getattr(controller, "active_turn", None)
    if active_turn is not None:
        return active_turn
    return controller.session.turns[-1] if controller.session.turns else None


def _fallback_model_options(backend: str) -> tuple[ModelOption, ...]:
    return FALLBACK_MODEL_OPTIONS.get(backend, (ModelOption("Default", None, "Use the backend default model."),))


def _default_model_option(backend: str) -> ModelOption:
    return _fallback_model_options(backend)[0]


def _dedupe_model_options(options: list[ModelOption]) -> tuple[ModelOption, ...]:
    deduped: list[ModelOption] = []
    seen_values: set[str] = set()
    for option in options:
        key = option.value or "<default>"
        if key in seen_values:
            continue
        seen_values.add(key)
        deduped.append(option)
    return tuple(deduped)


def _with_default_model_option(backend: str, options: list[ModelOption]) -> tuple[ModelOption, ...]:
    return _dedupe_model_options([_default_model_option(backend), *options])


def _codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".codex"


def _read_json_object(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _string_value(value) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _codex_model_options_from_cache(cache_path: Path) -> tuple[ModelOption, ...]:
    payload = _read_json_object(cache_path)
    if payload is None:
        return ()
    raw_models = payload.get("models")
    if isinstance(raw_models, dict):
        entries = list(raw_models.values())
    elif isinstance(raw_models, list):
        entries = raw_models
    else:
        return ()

    indexed_entries = [
        (index, entry)
        for index, entry in enumerate(entries)
        if isinstance(entry, dict)
    ]
    indexed_entries.sort(
        key=lambda item: (
            item[1].get("priority") if isinstance(item[1].get("priority"), int) else item[0],
            item[0],
        )
    )

    options: list[ModelOption] = []
    for _, entry in indexed_entries:
        visibility = str(entry.get("visibility", "")).lower()
        if visibility in {"hide", "hidden"}:
            continue
        if entry.get("hidden") is True or entry.get("show_in_picker") is False:
            continue
        value = _string_value(entry.get("slug")) or _string_value(entry.get("id")) or _string_value(entry.get("model"))
        if value is None:
            continue
        label = (
            _string_value(entry.get("display_name"))
            or _string_value(entry.get("displayName"))
            or _string_value(entry.get("title"))
            or _string_value(entry.get("name"))
            or value
        )
        description = _string_value(entry.get("description")) or "Codex model."
        options.append(ModelOption(label, value, description))
    return tuple(options)


def _codex_model_options() -> tuple[ModelOption, ...]:
    options = list(_codex_model_options_from_cache(_codex_home() / "models_cache.json"))
    return _with_default_model_option("codex", options) if options else _fallback_model_options("codex")


def _env_text(name: str) -> str | None:
    return _string_value(os.environ.get(name))


def _claude_alias_option(
    *,
    value: str,
    default_label: str,
    default_description: str,
    label_env: str | None = None,
    description_env: str | None = None,
) -> ModelOption:
    label = _env_text(label_env) if label_env else None
    description = _env_text(description_env) if description_env else None
    return ModelOption(label or default_label, value, description or default_description)


def _claude_model_options() -> tuple[ModelOption, ...]:
    options = [
        _claude_alias_option(
            value="sonnet",
            default_label="Sonnet",
            default_description="Claude Sonnet alias resolved by Claude Code.",
            label_env="ANTHROPIC_DEFAULT_SONNET_MODEL_NAME",
            description_env="ANTHROPIC_DEFAULT_SONNET_MODEL_DESCRIPTION",
        ),
        _claude_alias_option(
            value="opus",
            default_label="Opus",
            default_description="Claude Opus alias resolved by Claude Code.",
            label_env="ANTHROPIC_DEFAULT_OPUS_MODEL_NAME",
            description_env="ANTHROPIC_DEFAULT_OPUS_MODEL_DESCRIPTION",
        ),
        _claude_alias_option(
            value="haiku",
            default_label="Haiku",
            default_description="Claude Haiku alias resolved by Claude Code.",
            label_env="ANTHROPIC_DEFAULT_HAIKU_MODEL_NAME",
            description_env="ANTHROPIC_DEFAULT_HAIKU_MODEL_DESCRIPTION",
        ),
        ModelOption("Sonnet (1M context)", "sonnet[1m]", "Claude Sonnet with extended context when available."),
        ModelOption("Opus (1M context)", "opus[1m]", "Claude Opus with extended context when available."),
        ModelOption("Opus Plan Mode", "opusplan", "Use Opus in plan mode and Sonnet otherwise."),
    ]
    custom_model = _env_text("ANTHROPIC_CUSTOM_MODEL_OPTION")
    if custom_model:
        options.insert(
            0,
            ModelOption(
                _env_text("ANTHROPIC_CUSTOM_MODEL_OPTION_NAME") or custom_model,
                custom_model,
                _env_text("ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION") or "Custom Claude model option.",
            ),
        )
    return _with_default_model_option("claude", options)


def model_options_for_backend(backend: str) -> tuple[ModelOption, ...]:
    if backend == "codex":
        return _codex_model_options()
    if backend == "claude":
        return _claude_model_options()
    if backend == "gemini":
        return _fallback_model_options("gemini")
    return _fallback_model_options(backend)


def find_model_option(backend: str, value: str | None) -> ModelOption | None:
    normalized = (value or "").strip()
    for option in model_options_for_backend(backend):
        if option.value == value:
            return option
        if normalized and normalized.lower() in {option.label.lower(), (option.value or "").lower()}:
            return option
    return None


def find_permission_option(value: str) -> PermissionOption | None:
    normalized = value.strip().lower()
    for option in PERMISSION_OPTIONS:
        if normalized in {option.key.lower(), option.label.lower()}:
            return option
    return None


def format_model_options(controller: SessionController, selected_index: int = 0) -> str:
    backend = controller.session.backend.value
    active_model = current_model(controller)
    lines = [
        f"Model Picker ({backend})",
        f"Current: {current_model_label(controller)}",
        "",
    ]
    for index, option in enumerate(model_options_for_backend(backend)):
        marker = ">" if index == selected_index else " "
        active = " *" if option.value == active_model else ""
        value = option.value or "default"
        lines.append(f"{marker} {option.label:<20} {value:<32} {option.description}{active}")
    lines.extend(
        [
            "",
            "Enter apply",
            "Up/Down move",
            "Esc cancel",
        ]
    )
    return "\n".join(lines)


def format_permission_options(controller: SessionController, selected_index: int = 0) -> str:
    backend = controller.session.backend.value
    current = current_permission_values(controller)
    lines = [
        f"Permissions Picker ({backend})",
        f"Current: {current_permission_label(controller)}",
        "",
    ]
    for index, option in enumerate(PERMISSION_OPTIONS):
        marker = ">" if index == selected_index else " "
        values = permission_option_values_for_backend(option, backend)
        active = " *" if values == current else ""
        rendered_values = ", ".join(f"{key}={value}" for key, value in values.items())
        lines.append(f"{marker} {option.label:<22} {option.key:<12} {rendered_values:<52} {option.description}{active}")
    lines.extend(
        [
            "",
            "Enter apply",
            "Up/Down move",
            "Esc cancel",
        ]
    )
    return "\n".join(lines)


def apply_model_selection(controller: SessionController, model: str | None) -> str:
    backend = controller.session.backend.value
    if backend == "antigravity":
        available_models = tuple(
            option.value
            for option in model_options_for_backend(backend)
            if option.value is not None
        )
        try:
            selected_model = set_antigravity_model(model, available_models=available_models)
        except ValueError as exc:
            return str(exc)
        controller.attach_backend(build_backend(backend, model=selected_model, **permission_kwargs_for_backend(controller)))
        return f"Model set to {selected_model or 'default'} for {backend}."
    controller.attach_backend(build_backend(backend, model=model, **permission_kwargs_for_backend(controller)))
    return f"Model set to {model or 'default'} for {backend}."


def apply_permission_selection(controller: SessionController, option: PermissionOption) -> str:
    backend = controller.session.backend.value
    kwargs = permission_option_values_for_backend(option, backend)
    controller.attach_backend(build_backend(backend, model=current_model(controller), **kwargs))
    rendered_values = ", ".join(f"{key}={value}" for key, value in kwargs.items())
    return f"Permissions set to {option.label} for {backend}: {rendered_values}."


def colorize_backend_label(text: str, backend: str) -> str:
    color = BACKEND_ANSI.get(backend, "")
    if not color:
        return text
    return f"{ANSI_BOLD}{color}{text}{ANSI_RESET}"


def _divider(label: str) -> str:
    return f"== {label} " + "=" * 52


def _format_output_lines(text: str) -> list[str]:
    output_lines = [line for line in text.splitlines() if line.strip()]
    return output_lines or ["<empty>"]


def format_markdown_lines(text: str) -> list[str]:
    source_lines = text.splitlines() or [text]
    rendered: list[str] = []
    in_code_block = False
    code_language = "text"
    code_buffer: list[str] = []

    def flush_code_block() -> None:
        nonlocal code_buffer, code_language
        rendered.append(f"┌─ code:{code_language}")
        for code_line in code_buffer or [""]:
            rendered.append(f"│ {code_line}")
        rendered.append("└─")
        code_buffer = []
        code_language = "text"

    for raw_line in source_lines:
        stripped = raw_line.rstrip()
        if stripped.startswith("```"):
            if in_code_block:
                flush_code_block()
                in_code_block = False
            else:
                in_code_block = True
                code_language = stripped[3:].strip() or "text"
            continue
        if in_code_block:
            code_buffer.append(stripped)
            continue
        if stripped.startswith("##"):
            rendered.append(stripped.lstrip("# ").upper())
        elif stripped.startswith(("- ", "* ")):
            rendered.append(f"• {stripped[2:]}")
        else:
            rendered.append(stripped)
    if in_code_block:
        flush_code_block()
    return [line for line in rendered if line != ""] or ["<empty>"]


def backend_glyph(backend: BackendName | str) -> str:
    value = backend.value if isinstance(backend, BackendName) else str(backend)
    return BACKEND_GLYPHS.get(value, value[:1].upper() or "?")


def backend_vendor_label(backend: BackendName | str) -> str:
    value = backend.value if isinstance(backend, BackendName) else str(backend)
    return BACKEND_VENDOR_LABELS.get(value, "Backend")


def _first_nonempty_line(text: str, fallback: str = "<empty>") -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return fallback


def _single_line(text: str, *, fallback: str = "<empty>", max_chars: int = 72) -> str:
    normalized = " ".join((text or "").split()) or fallback
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


def _clock_time(value: str | None) -> str:
    if not value:
        return "--:--:--"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return _single_line(value, max_chars=19)
    return parsed.astimezone().strftime("%H:%M:%S")


def _turn_elapsed_label(turn: TurnRecord) -> str | None:
    if not turn.started_at or not turn.completed_at:
        return None
    try:
        started = datetime.fromisoformat(turn.started_at.replace("Z", "+00:00"))
        completed = datetime.fromisoformat(turn.completed_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    elapsed = max(0.0, (completed - started).total_seconds())
    if elapsed >= 60:
        return f"{elapsed / 60:.1f}m"
    return f"{elapsed:.1f}s"


def _session_state_label(controller: SessionController, *, is_busy: bool) -> str:
    if is_busy:
        return "busy · streaming"
    recovery = latest_recovery_status(controller)
    if recovery and "interrupted" in recovery:
        return "interrupted"
    return "ready"


def _session_state_style(controller: SessionController, *, is_busy: bool) -> str:
    state_label = _session_state_label(controller, is_busy=is_busy)
    if state_label.startswith("busy"):
        return "class:state.busy"
    if state_label == "interrupted":
        return "class:state.interrupted"
    return "class:state.ready"


def _active_turn_count(controller: SessionController) -> int:
    return len(_iter_turns(controller))


def _active_task_short_label(controller: SessionController, *, max_chars: int = 34) -> str:
    return _single_line(active_task_label(controller), max_chars=max_chars)


def _iter_turns(controller: SessionController):
    turns = list(controller.session.turns)
    active_turn = getattr(controller, "active_turn", None)
    turn_ids = {getattr(turn, "id", None) for turn in turns}
    if active_turn is not None and getattr(active_turn, "id", None) not in turn_ids:
        turns.append(active_turn)
    return turns


SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def spinner_frame(tick: int) -> str:
    return SPINNER_FRAMES[tick % len(SPINNER_FRAMES)]


def phase_label(status: TurnStatus, output: str, tick: int = 0) -> str:
    spin = spinner_frame(tick)
    if status is TurnStatus.SUBMITTING:
        return f"{spin} launching…"
    if status is TurnStatus.STREAMING and not output.strip():
        return f"{spin} waiting for first token…"
    if status is TurnStatus.STREAMING:
        return f"{spin} streaming…"
    if status is TurnStatus.COMPLETED:
        return f"{spin} finalizing…"
    return f"{spin} working…"


def stalled_message(status: TurnStatus, elapsed_seconds: float) -> str:
    if elapsed_seconds < 8:
        return ""
    if status in {TurnStatus.SUBMITTING, TurnStatus.STREAMING}:
        return f"still waiting for backend response… {elapsed_seconds:.0f}s"
    return ""


def turn_is_busy(turn) -> bool:
    return turn.status in {TurnStatus.SUBMITTING, TurnStatus.STREAMING}


def format_turn_recovery_status(turn: TurnRecord | None) -> str | None:
    if turn is None:
        return None
    transcript_state = turn_transcript_state(turn)
    if transcript_state not in {"failed", "incomplete", "interrupted"}:
        return None
    recovery = turn.metadata.get("recovery") if isinstance(turn.metadata, dict) else None
    terminal_event_seen = recovery.get("terminal_event_seen") if isinstance(recovery, dict) else None
    parts = [transcript_state]
    if terminal_event_seen is False or transcript_state == "interrupted":
        parts.append("no terminal event")
    elif terminal_event_seen is True:
        parts.append("terminal event seen")
    elif transcript_state == "incomplete":
        parts.append("terminal event pending" if turn_is_busy(turn) else "terminal event unknown")
    if turn_has_partial_output(turn):
        parts.append("partial output")
    if turn.error is not None:
        parts.append(f"error={turn.error.kind}")
    return "; ".join(parts)


def latest_recovery_status(controller: SessionController) -> str | None:
    return format_turn_recovery_status(_latest_status_turn(controller))


def progress_message(turn, tick: int = 0) -> str:
    if turn_is_busy(turn):
        message = phase_label(turn.status, turn.output, tick=tick)
        activity = latest_activity_text(turn)
        if activity:
            return f"{message} {activity}"
        if turn.output.strip():
            return f"{message} {len(turn.output)} chars"
        return message
    return f"Last turn: {turn.status.value}"


def transcript_turn_separator(turn_number: int) -> str:
    return f"──── turn {turn_number} " + "─" * 20


def latest_activity_text(turn) -> str:
    for event in reversed(turn.events):
        if event.type == EventType.ACTIVITY.value and event.text.strip():
            return event.text.strip()
    return ""


def latest_summary_lines(controller: SessionController) -> list[str]:
    summaries = getattr(controller.session, "summaries", [])
    if not summaries:
        return []
    summary = summaries[-1]
    lines = [
        "──── latest summary checkpoint " + "─" * 10,
        "",
        f"summary > {summary.kind} • {summary.scope} • {summary.created_at}",
    ]
    for line in format_markdown_lines(summary.text):
        lines.append(f"          {line}")
    return lines


def resume_context_label(controller: SessionController) -> str:
    if getattr(controller, "resume_context_pending", False):
        return "pending"
    config = getattr(controller, "resume_context_config", None)
    if config is not None and getattr(config, "enabled", False):
        return "used"
    return "off"


def resume_context_status_message(controller: SessionController) -> str:
    payload = controller.preview_resume_context() if hasattr(controller, "preview_resume_context") else None
    if payload is None:
        return "No resume context pending"
    summary_id = payload.metadata.get("injected_summary_id") or "<none>"
    turn_count = len(payload.metadata.get("injected_turn_ids", []))
    chars = payload.metadata.get("context_char_count", 0)
    return f"Resume context pending: summary={summary_id} turns={turn_count} chars={chars}"


def format_resume_context_preview(controller: SessionController) -> str:
    payload = controller.preview_resume_context() if hasattr(controller, "preview_resume_context") else None
    if payload is None:
        return "No resume context pending."
    summary_id = payload.metadata.get("injected_summary_id") or "<none>"
    turn_ids = payload.metadata.get("injected_turn_ids", [])
    turns = ", ".join(turn_ids) if turn_ids else "<none>"
    return "\n".join(
        [
            "Resume context preview",
            f"Summary: {summary_id}",
            f"Turns  : {turns}",
            f"Chars  : {payload.metadata.get('context_char_count', 0)}",
            "",
            payload.context_text,
        ]
    )


def task_status_message(controller: SessionController) -> str:
    active_task = controller.active_user_task() if hasattr(controller, "active_user_task") else None
    if active_task is not None:
        title = active_task.title or "<untitled>"
        turns = len(active_task.turn_ids)
        return f"Active task: {title} ({active_task.id}) turns={turns}"
    latest_closed = controller.latest_closed_task() if hasattr(controller, "latest_closed_task") else None
    if latest_closed is not None:
        title = latest_closed.title or "<untitled>"
        turns = len(latest_closed.turn_ids)
        return f"No active task. Latest closed: {title} ({latest_closed.id}) turns={turns}. Prompts default to task-main."
    return "No active task. Prompts default to task-main."


def active_task_label(controller: SessionController) -> str:
    active_task = controller.active_user_task() if hasattr(controller, "active_user_task") else None
    if active_task is not None:
        title = active_task.title or "<untitled>"
        return f"{title} ({active_task.id})"
    latest_closed = controller.latest_closed_task() if hasattr(controller, "latest_closed_task") else None
    if latest_closed is not None and latest_closed.id != "task-main":
        title = latest_closed.title or "<untitled>"
        return f"task-main (default); latest closed: {title} ({latest_closed.id})"
    return "task-main (default)"


def handle_task_command(controller: SessionController, args: str) -> tuple[bool, str]:
    parts = args.split(None, 1)
    if not parts:
        return False, "Usage: /task start <title> | /task status | /task close [note]"
    subcommand = parts[0].lower()
    remainder = parts[1].strip() if len(parts) > 1 else ""
    if subcommand == "start":
        try:
            task = controller.start_task(remainder or None)
        except ValueError as exc:
            return False, str(exc)
        title = task.title or "<untitled>"
        return True, f"Started task: {title} ({task.id})"
    if subcommand == "status":
        return True, task_status_message(controller)
    if subcommand == "close":
        try:
            task = controller.close_task(remainder or None)
        except ValueError as exc:
            return False, str(exc)
        title = task.title or "<untitled>"
        suffix = " with note." if task.closing_note else "."
        return True, f"Closed task: {title} ({task.id}){suffix}"
    return False, "Usage: /task start <title> | /task status | /task close [note]"


def format_product_status(controller: SessionController, *, is_busy: bool = False) -> str:
    summaries = getattr(controller.session, "summaries", [])
    turn_count, last_status, resumable = _session_continuation_state(controller)
    recovery_status = latest_recovery_status(controller) or "none"
    return "\n".join(
        [
            "CCG Status",
            f"Backend : {controller.session.backend.value}",
            f"Model   : {current_model_label(controller)}",
            f"Perms   : {current_permission_label(controller)}",
            f"Session : {controller.session.id}",
            f"Vendor  : {getattr(controller.session, 'vendor_session_id', None) or 'pending'}",
            f"Workspace: {getattr(controller, 'cwd', Path.cwd())}",
            f"State   : {'busy' if is_busy else 'ready'}",
            f"Turns   : {turn_count}",
            f"Summary : {len(summaries)}",
            f"Task    : {active_task_label(controller)}",
            f"Context : {resume_context_label(controller)}",
            f"Route   : {compact_routing_advisory()}",
            f"Last    : {last_status}",
            f"Recovery: {recovery_status}",
            f"Resume  : {'yes' if resumable else 'no'}",
        ]
    )


def latest_assistant_output(controller: SessionController) -> str:
    for turn in reversed(controller.session.turns):
        if turn.output.strip():
            return turn.output
    return ""


def copy_text_to_clipboard(text: str) -> tuple[bool, str]:
    if not text:
        return False, "No assistant output to copy."
    commands = (
        ("wl-copy",),
        ("xclip", "-selection", "clipboard"),
        ("xsel", "--clipboard", "--input"),
    )
    for command in commands:
        executable = shutil.which(command[0])
        if executable is None:
            continue
        try:
            subprocess.run(
                (executable, *command[1:]),
                input=text,
                text=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        return True, f"Copied latest assistant output ({len(text)} chars)."
    return False, "Clipboard tool not found. Latest assistant output is still visible in the transcript."


def format_resume_session_list(controller: SessionController) -> str:
    sessions = [
        session
        for session in controller.store.list_sessions()
        if session.backend == controller.session.backend.value and session.id != controller.session.id
    ]
    if not sessions:
        return f"No saved {controller.session.backend.value} sessions to resume."
    return "\n".join(
        [
            "Resume a session with /resume <session_id>",
            "",
            format_session_list(sessions[:10]),
        ]
    )


def _format_detail_value(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _activity_detail_lines(activity: dict, *, indent: str = "           ") -> list[str]:
    details = activity.get("details")
    if not isinstance(details, dict) or not details:
        return []
    lines: list[str] = []
    for key, value in details.items():
        rendered = _format_detail_value(value)
        if rendered:
            for index, line in enumerate(rendered.splitlines() or [rendered]):
                prefix = f"{indent}{key}: " if index == 0 else indent
                lines.append(f"{prefix}{line}")
    return lines


def recent_activity_lines(turn, limit: int = 5, show_details: bool = False) -> list[str]:
    activity_events = [
        event
        for event in turn.events
        if event.type == EventType.ACTIVITY.value and event.text.strip()
    ]
    lines: list[str] = []
    for event in activity_events[-limit:]:
        activity = event.activity or {}
        status = activity.get("status") if isinstance(activity, dict) else None
        status_suffix = f" [{status}]" if status else ""
        lines.append(f"activity > {event.text.strip()}{status_suffix}")
        if show_details and isinstance(activity, dict):
            lines.extend(_activity_detail_lines(activity))
    return lines


def turn_meta_lines(
    turn,
    turn_number: int,
    tick: int = 0,
    elapsed_seconds: float = 0.0,
    show_activity_details: bool = False,
) -> list[str]:
    status_text = phase_label(turn.status, turn.output, tick=tick) if turn_is_busy(turn) else turn.status.value
    lines = [f"meta   > turn {turn_number} • {turn.backend.value} • {status_text}"]
    stalled = stalled_message(turn.status, elapsed_seconds)
    if stalled:
        lines.append(f"meta   > {stalled}")
    recovery_status = format_turn_recovery_status(turn)
    if recovery_status is not None:
        lines.append(f"recovery> {recovery_status}")
    lines.extend(recent_activity_lines(turn, show_details=show_activity_details))
    resume_context = getattr(turn, "metadata", {}).get("resume_context", {})
    if isinstance(resume_context, dict) and resume_context.get("injected"):
        summary_id = resume_context.get("injected_summary_id") or "<none>"
        turn_count = len(resume_context.get("injected_turn_ids", []))
        chars = resume_context.get("context_char_count", 0)
        lines.append(f"context> resume context injected • summary {summary_id} • turns {turn_count} • {chars} chars")
    if turn.error is not None:
        lines.append(f"error  > {turn.error.message}")
    return lines


def _conversation_lines(
    controller: SessionController,
    use_color: bool = False,
    tick: int = 0,
    elapsed_seconds: float = 0.0,
    show_activity_details: bool = False,
) -> list[str]:
    turns = _iter_turns(controller)
    lines: list[str] = []
    if not turns:
        lines.extend(["No messages yet.", "Send a prompt to start the session."])
    else:
        for index, turn in enumerate(turns):
            turn_number = index + 1
            lines.append(transcript_turn_separator(turn_number))
            lines.append("")
            lines.append(f"You    > {turn.prompt}")
            lines.append("")
            label = f"{turn.backend.value:<6}"
            if use_color:
                label = colorize_backend_label(label, turn.backend.value)
            if turn.output.strip():
                output_lines = format_markdown_lines(turn.output)
            elif turn.status in {TurnStatus.SUBMITTING, TurnStatus.STREAMING}:
                output_lines = [phase_label(turn.status, turn.output, tick=tick + index)]
            else:
                output_lines = ["<empty>"]
            lines.append(f"{label} > {output_lines[0]}")
            for extra_line in output_lines[1:]:
                lines.append(f"         {extra_line}")
            lines.append("")
            lines.extend(
                turn_meta_lines(
                    turn,
                    turn_number=turn_number,
                    tick=tick + index,
                    elapsed_seconds=elapsed_seconds,
                    show_activity_details=show_activity_details,
                )
            )
            lines.append("")
    summary_lines = latest_summary_lines(controller)
    if summary_lines:
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend(summary_lines)
    return lines[:-1] if lines and lines[-1] == "" else lines


def build_header_line(controller: SessionController) -> str:
    cwd_name = Path(getattr(controller, "cwd", Path.cwd())).name or "/"
    backend_badge = f"[{controller.session.backend.value}]"
    return (
        f"CCG TUI {backend_badge}"
        f" | Session: {controller.session.id}"
        f" | Turns: {len(controller.session.turns)}"
        f" | Workspace: {cwd_name}"
    )


def build_sidebar_text(controller: SessionController, is_busy: bool = False, draft_text: str = "") -> str:
    vendor_session = getattr(controller.session, "vendor_session_id", None) or "pending"
    summaries = getattr(controller.session, "summaries", [])
    turn_count, last_status, resumable = _session_continuation_state(controller)
    recovery_status = latest_recovery_status(controller) or "none"
    line_count = len(draft_text.splitlines()) if draft_text else 0
    char_count = len(draft_text)
    lines = [
        "Session Info",
        f"Backend : {controller.session.backend.value}",
        f"Model   : {current_model_label(controller)}",
        f"Perms   : {current_permission_label(controller)}",
        f"Vendor  : {vendor_session}",
        f"Turns   : {turn_count}",
        f"Summary : {len(summaries)}",
        f"Task    : {active_task_label(controller)}",
        f"Context : {resume_context_label(controller)}",
        f"Route   : {compact_routing_advisory()}",
        f"State   : {'busy' if is_busy else 'ready'}",
        f"Last    : {last_status}",
        f"Recovery: {recovery_status}",
        f"Resume  : {'yes' if resumable else 'no'}",
        f"Draft   : {line_count} lines / {char_count} chars",
        "",
        "Controls",
        "Enter   submit prompt",
        "S-Enter newline",
        "C-J     submit fallback",
        "Esc+Ret newline fallback",
        "F2      refresh view",
        "F3      activity details",
        "Esc     quit",
        "",
        "Commands",
        "/history refresh transcript",
        "/details toggle activity details",
        "/context preview resume context",
        "/capabilities route registry",
        "/summarize create Gemini summary",
        "/task    manage task boundary",
        "/quit    exit session",
    ]
    return "\n".join(lines)


def _build_prompt_toolkit_sidebar_fragments(
    controller: SessionController,
    *,
    is_busy: bool = False,
    draft_text: str = "",
    sidebar_groups: dict[str, bool] | None = None,
    toggle_sidebar_group: Callable[[str], None] | None = None,
) -> list[tuple[str, ...]]:
    sidebar_groups = sidebar_groups or {
        "state": True,
        "session": True,
        "composer": True,
        "lineage": False,
        "last activity": False,
        "quick actions": True,
    }
    vendor_session = getattr(controller.session, "vendor_session_id", None) or "pending"
    summaries = getattr(controller.session, "summaries", [])
    turn_count, last_status, resumable = _session_continuation_state(controller)
    line_count = len(draft_text.splitlines()) if draft_text else 0
    char_count = len(draft_text)
    backend = controller.session.backend.value
    permission_values = current_permission_values(controller)
    permission_detail = (
        permission_values.get("sandbox_mode")
        or permission_values.get("permission_mode")
        or permission_values.get("approval_mode")
        or "default"
    )
    permission_option = current_permission_option(controller)
    permission_label = permission_option.key if permission_option is not None else current_permission_label(controller)
    status_label = _session_state_label(controller, is_busy=is_busy)
    status_style = _session_state_style(controller, is_busy=is_busy)
    fragments: list[tuple[str, ...]] = []

    def group(label: str, *, collapsible: bool = True, shortcut: str | None = None) -> None:
        open_group = bool(sidebar_groups.get(label, True))
        arrow = "▾" if open_group else "▸"
        header = f"{arrow} {label.upper()}"
        if shortcut:
            header += f" [{shortcut}]"
        rule_width = max(1, 28 - len(label) - (len(shortcut) + 3 if shortcut else 0))

        if collapsible and toggle_sidebar_group is not None:
            from prompt_toolkit.mouse_events import MouseEventType

            def toggle_group(event, group_label: str = label) -> None:
                if event.event_type != MouseEventType.MOUSE_UP:
                    return
                toggle_sidebar_group(group_label)

            fragments.append(("class:sidebar.group", f"{header} ", toggle_group))
        else:
            fragments.append(("class:sidebar.group", f"{header} "))
        fragments.append(("class:rule", "─" * rule_width))
        fragments.append(("", "\n"))

    def kv(key: str, value: str, *, style: str = "class:sidebar.value") -> None:
        fragments.append(("class:sidebar.key", f"{key:<9} "))
        fragments.append((style, f"{value}\n"))

    group("state")
    kv("Backend :", f"[{backend_glyph(backend)}] {backend}")
    kv("Model   :", _single_line(current_model_label(controller), max_chars=25))
    kv("Perms   :", _single_line(f"{permission_label} · {permission_detail}", max_chars=31))
    kv("status", f"● {status_label}", style=status_style)
    kv("route", "local · advisory only")

    group("session")
    if sidebar_groups.get("session", True):
        kv("ccg id", _single_line(controller.session.id, max_chars=28))
        kv("vendor", _single_line(vendor_session, max_chars=28))
        kv("turns", f"{turn_count:02d} / {turn_count:02d}")
        kv("summary", str(len(summaries)))
        kv("task", _active_task_short_label(controller, max_chars=28))
        kv("last", last_status)
        kv("resume", "yes" if resumable else "no")

    group("composer")
    if sidebar_groups.get("composer", True):
        kv("draft", f"{line_count} lines / {char_count} chars")
        kv("slash", "CCG + backend translations")

    group("lineage", shortcut="F4")
    if sidebar_groups.get("lineage", False):
        kv("kind", "root")
        kv("parent", "—")
        kv("children", "0")
        kv("forks", "0")

    group("last activity", shortcut="F5")
    if sidebar_groups.get("last activity", False):
        kv("00:14", "edit · handoff.py", style="class:tool.run")
        kv("00:09", "grep · excluded_source_ids", style="class:tool.ok")
        kv("00:04", "read · handoff.py:412", style="class:tool.ok")

    group("quick actions", collapsible=False)
    kv("F2", "refresh history")
    kv("F3", "toggle activity details")
    kv("/handoff", "preview handoff")
    kv("/summarize", "checkpoint via gemini")
    kv("/context", "resume context")
    return fragments


def build_status_text(
    controller: SessionController,
    composer_message: str = "Type a prompt. Enter submits. Shift-Enter adds a newline.",
    is_busy: bool = False,
) -> str:
    prefix = "busy" if is_busy else "ready"
    recovery_suffix = ""
    recovery_status = latest_recovery_status(controller)
    if recovery_status is not None:
        recovery_suffix = f" • recovery={recovery_status.replace('; ', ', ')}"
    return (
        f"{composer_message} • {prefix} • Enter submit • Shift-Enter newline"
        f" • Ctrl-J submit • Esc-Enter newline fallback • Esc quit"
        f" • F3 details • backend={controller.session.backend.value}{recovery_suffix}"
    )


def build_transcript_text(
    controller: SessionController,
    tick: int = 0,
    elapsed_seconds: float = 0.0,
    show_activity_details: bool = False,
) -> str:
    return "\n".join(
        _conversation_lines(
            controller,
            tick=tick,
            elapsed_seconds=elapsed_seconds,
            show_activity_details=show_activity_details,
        )
    )


def render_interface_screen(
    controller: SessionController,
    composer_text: str = "",
    tick: int = 0,
    elapsed_seconds: float = 0.0,
    show_activity_details: bool = False,
) -> str:
    lines = [
        build_header_line(controller),
        _divider("Conversation"),
        *(
            _conversation_lines(
                controller,
                use_color=False,
                tick=tick,
                elapsed_seconds=elapsed_seconds,
                show_activity_details=show_activity_details,
            )
        ),
        _divider("Composer"),
        composer_text or "Type a prompt. Enter submits. Shift-Enter adds a newline.",
        _divider("Commands"),
        "/history   refresh conversation view",
        "/details   toggle activity details",
        "/context   preview resume context",
        "/capabilities route registry",
        "/summarize create Gemini summary checkpoint",
        "/task      manage task boundary",
        "/quit      exit session",
        "Enter      submit prompt",
        "Shift-Enter adds a newline",
        "Ctrl-J submits as a fallback",
        "Esc-Enter adds a newline fallback",
        "Ctrl-C     exit session",
    ]
    return "\n".join(lines)


def _draw_screen(
    controller: SessionController,
    print_fn: Callable[[str], None],
    composer_text: str = "",
    show_activity_details: bool = False,
) -> None:
    print_fn(SCREEN_CLEAR + render_interface_screen(controller, composer_text=composer_text, show_activity_details=show_activity_details))


def run_simple_interface(
    controller_factory: Callable[[str], SessionController],
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    initial_backend: str | None = None,
) -> int:
    backend = initial_backend or choose_backend(input_fn=input_fn, print_fn=print_fn)
    controller = controller_factory(backend)
    show_activity_details = False

    def resume_session(session_id: str) -> tuple[bool, str]:
        nonlocal controller
        try:
            session = controller.store.load_session(session_id)
        except FileNotFoundError:
            return False, f"Session not found: {session_id}"
        if session.backend != controller.session.backend:
            return False, f"Session {session_id} uses {session.backend.value}; current backend is {controller.session.backend.value}."
        old_controller = controller
        resume_cwd = Path(session.workspace_cwd) if session.workspace_cwd else controller.cwd
        controller = SessionController.resume(
            adapter=build_backend(session.backend.value),
            store=controller.store,
            cwd=resume_cwd,
            session=session,
            resume_context_config=getattr(controller, "resume_context_config", ResumeContextConfig(enabled=False)),
        )
        old_controller.close()
        return True, f"Resumed session: {session.id}"

    try:
        _draw_screen(controller, print_fn, show_activity_details=show_activity_details)
        while True:
            try:
                prompt = input_fn("ccg> ").strip()
            except EOFError:
                print_fn("")
                return 0
            if not prompt:
                continue
            parsed = parse_slash_command(prompt, controller.session.backend.value)
            if parsed is not None and parsed.action is SlashCommandAction.PRODUCT:
                composer_text = ""
                if parsed.canonical == "/quit":
                    return 0
                if parsed.canonical == "/help":
                    composer_text = format_slash_command_help(backend=controller.session.backend.value)
                elif parsed.canonical == "/clear":
                    current_backend = controller.session.backend.value
                    controller.close()
                    controller = controller_factory(current_backend)
                    composer_text = f"Started fresh {current_backend} session."
                elif parsed.canonical == "/model":
                    if parsed.args:
                        option = find_model_option(controller.session.backend.value, parsed.args)
                        model = option.value if option is not None else parsed.args
                        composer_text = apply_model_selection(controller, model)
                    else:
                        composer_text = format_model_options(controller)
                elif parsed.canonical == "/permissions":
                    if parsed.args:
                        option = find_permission_option(parsed.args)
                        if option is None:
                            composer_text = f"Unknown permission preset: {parsed.args}"
                        else:
                            composer_text = apply_permission_selection(controller, option)
                    else:
                        composer_text = format_permission_options(controller)
                elif parsed.canonical == "/status":
                    composer_text = format_product_status(controller)
                elif parsed.canonical == "/capabilities":
                    record_capability_inspection(controller)
                    composer_text = format_capability_registry(controller)
                elif parsed.canonical == "/copy":
                    _, composer_text = copy_text_to_clipboard(latest_assistant_output(controller))
                elif parsed.canonical == "/resume":
                    if parsed.args:
                        _, composer_text = resume_session(parsed.args)
                    else:
                        composer_text = format_resume_session_list(controller)
                _draw_screen(controller, print_fn, composer_text=composer_text, show_activity_details=show_activity_details)
                continue
            if parsed is not None and parsed.action is SlashCommandAction.LOCAL and parsed.canonical == "/history":
                _draw_screen(controller, print_fn, show_activity_details=show_activity_details)
                continue
            if parsed is not None and parsed.action is SlashCommandAction.LOCAL and parsed.canonical == "/details":
                show_activity_details = not show_activity_details
                _draw_screen(
                    controller,
                    print_fn,
                    composer_text=f"Activity details: {'expanded' if show_activity_details else 'collapsed'}",
                    show_activity_details=show_activity_details,
                )
                continue
            if parsed is not None and parsed.action is SlashCommandAction.LOCAL and parsed.canonical == "/context":
                _draw_screen(
                    controller,
                    print_fn,
                    composer_text=format_resume_context_preview(controller),
                    show_activity_details=show_activity_details,
                )
                continue
            if parsed is not None and parsed.action is SlashCommandAction.LOCAL and parsed.canonical == "/summarize":
                summary_adapter = build_summary_backend()
                try:
                    summary = controller.generate_summary(summary_adapter)
                    composer_text = f"Summary saved: {summary.id}"
                except SummaryGenerationError as exc:
                    composer_text = f"Summary failed: {exc}"
                finally:
                    summary_adapter.close()
                _draw_screen(controller, print_fn, composer_text=composer_text, show_activity_details=show_activity_details)
                continue
            if parsed is not None and parsed.action is SlashCommandAction.LOCAL and parsed.canonical == "/handoff":
                source_permission_values = current_permission_values(controller)
                ok, composer_text = build_handoff_preview(
                    controller.session,
                    parsed.args,
                    source_permission_values=source_permission_values,
                )
                record_controller_handoff_routing_decision(
                    controller,
                    parsed.args,
                    source_permission_values=source_permission_values,
                    user_decision="deferred",
                    final_action="previewed" if ok else "blocked",
                    reason="" if ok else composer_text,
                )
                _draw_screen(
                    controller,
                    print_fn,
                    composer_text=composer_text,
                    show_activity_details=show_activity_details,
                )
                continue
            if parsed is not None and parsed.action is SlashCommandAction.LOCAL and parsed.canonical == "/task":
                _, composer_text = handle_task_command(controller, parsed.args)
                _draw_screen(
                    controller,
                    print_fn,
                    composer_text=composer_text,
                    show_activity_details=show_activity_details,
                )
                continue
            backend_prompt = parsed.backend_prompt if parsed is not None else prompt
            turn = controller.submit_prompt(backend_prompt)
            composer_text = f"Last turn: {turn.status.value}"
            _draw_screen(controller, print_fn, composer_text=composer_text, show_activity_details=show_activity_details)
    finally:
        controller.close()


def _build_prompt_toolkit_style():
    from prompt_toolkit.styles import Style

    return Style.from_dict(
        {
            "chrome": "bg:#22252b #7d838f",
            "chrome.live": "bg:#22252b #d8c47a bold",
            "chrome.path": "bg:#22252b #a8adb6",
            "header": "bg:#1c1f24 #d0d4dc",
            "header.brand": "bg:#252931 #eef0f3 bold",
            "header.label": "bg:#1c1f24 #7d838f",
            "header.value": "bg:#1c1f24 #eef0f3",
            "header.dim": "bg:#1c1f24 #7d838f",
            "header.meta": "bg:#1c1f24 #a8adb6",
            "header.rule": "bg:#1c1f24 #555b66",
            "border": "bg:#16181d #555b66",
            "backend.codex": "bg:#252931 #d0d4dc bold",
            "backend.claude": "bg:#252931 #d0d4dc bold",
            "backend.gemini": "bg:#252931 #d0d4dc bold",
            "state.ready": "bg:#223226 #8dd9ad bold",
            "state.busy": "bg:#332d1d #d8c47a bold",
            "state.interrupted": "bg:#3a261f #e08a72 bold",
            "section": "bg:#1c1f24 #7d838f",
            "rule": "bg:#1c1f24 #555b66",
            "gutter": "bg:#1c1f24 #7d838f",
            "gutter.role": "bg:#1c1f24 #eef0f3 bold",
            "gutter.ccg": "bg:#1c1f24 #d8c47a bold",
            "gutter.err": "bg:#1c1f24 #e08a72 bold",
            "gutter.num": "bg:#1c1f24 #5d646f",
            "role.user": "#eef0f3 bold",
            "role.assistant": "#d0d4dc",
            "role.meta": "#8f96a3",
            "role.error": "#e08a72 bold",
            "note.ccg": "bg:#252931 #d8c47a",
            "note.text": "bg:#252931 #a8adb6",
            "interrupted": "bg:#35241f #e08a72",
            "interrupted.text": "bg:#1c1f24 #a8adb6 italic",
            "tool.row": "bg:#22252b #a8adb6",
            "tool.kind": "bg:#22252b #7d838f",
            "tool.ok": "bg:#22252b #8dd9ad",
            "tool.run": "bg:#22252b #d8c47a",
            "tool.err": "bg:#22252b #e08a72",
            "sidebar": "bg:#1c1f24 #d0d4dc",
            "sidebar.group": "bg:#1c1f24 #7d838f bold",
            "sidebar.key": "bg:#1c1f24 #7d838f",
            "sidebar.value": "bg:#1c1f24 #d0d4dc",
            "composer": "bg:#22252b #eef0f3",
            "composer.prompt": "bg:#1c1f24 #d8c47a bold",
            "composer.hint": "bg:#1c1f24 #7d838f",
            "composer.border": "bg:#1c1f24 #555b66",
            "composer.submit.border": "bg:#252931 #555b66",
            "composer.submit.border.disabled": "bg:#252931 #4d535e",
            "composer.submit.label": "bg:#252931 #eef0f3 bold",
            "composer.submit.label.disabled": "bg:#252931 #7d838f",
            "composer.submit.key": "bg:#252931 #d8c47a bold",
            "composer.submit.key.disabled": "bg:#252931 #5d646f bold",
            "composer.submit.hint": "bg:#252931 #7d838f",
            "composer.submit.hint.disabled": "bg:#252931 #5d646f",
            "footer": "bg:#22252b #7d838f",
            "footer.ready": "bg:#22252b #8dd9ad",
            "footer.busy": "bg:#22252b #d8c47a",
            "footer.interrupted": "bg:#22252b #e08a72",
            "picker": "bg:#1c1f24 #d0d4dc",
            "picker.active": "bg:#2b3038 #d8c47a bold",
            "picker.header": "bg:#22252b #eef0f3 bold",
            "picker.footer": "bg:#22252b #7d838f",
            "tag": "bg:#252931 #a8adb6",
            "tag.ok": "bg:#213029 #8dd9ad",
            "tag.warn": "bg:#332d1d #d8c47a",
            "tag.err": "bg:#3a261f #e08a72",
            "tag.info": "bg:#202b35 #9db5d9",
            "completion-menu": "bg:#1c1f24 #d0d4dc",
            "slash.row": "bg:#1c1f24 #d0d4dc",
            "slash.active": "bg:#2b3038 #d8c47a bold",
        }
    )


SLASH_PALETTE_VISIBLE_ROWS = 7


def _slash_palette_visible_range(
    total: int,
    selected_index: int,
    *,
    visible_rows: int = SLASH_PALETTE_VISIBLE_ROWS,
) -> tuple[int, int]:
    if total <= 0:
        return (0, 0)
    if visible_rows <= 0 or total <= visible_rows:
        return (0, total)
    selected_index = max(0, min(selected_index, total - 1))
    start = selected_index - visible_rows // 2
    start = max(0, min(start, total - visible_rows))
    return (start, start + visible_rows)


def _build_prompt_toolkit_transcript_fragments(
    controller: SessionController,
    tick: int = 0,
    elapsed_seconds: float = 0.0,
    show_activity_details: bool = False,
):
    fragments: list[tuple[str, str]] = []
    turns = _iter_turns(controller)
    gutter_width = 8

    def append_role_line(
        label: str,
        number: int | None,
        text: str,
        *,
        gutter_style: str,
        text_style: str,
    ) -> None:
        number_text = f"{number:02d}" if number is not None else ""
        gutter = f"{label:>{gutter_width}}\n{number_text:>{gutter_width}}"
        first = True
        for raw_line in text.splitlines() or [""]:
            for line in _format_output_lines(raw_line):
                rendered_gutter = gutter if first else " " * gutter_width
                fragments.append((gutter_style, f"{rendered_gutter}  "))
                fragments.append((text_style, f"{line}\n"))
                first = False

    def append_content_line(text: str, *, style: str = "class:role.assistant") -> None:
        fragments.append(("class:gutter", " " * gutter_width + "  "))
        fragments.append((style, f"{text}\n"))

    def append_meta_line(text: str, *, error: bool = False) -> None:
        append_content_line(text, style="class:role.error" if error else "class:role.meta")

    def append_ccg_note(label: str, body: str) -> None:
        fragments.append(("class:gutter.ccg", f"{'CCG':>{gutter_width}}  "))
        fragments.append(("class:note.ccg", "│ "))
        fragments.append(("class:note.text", f"{label} · {body}\n"))

    def append_turn_meta(turn, turn_number: int) -> None:
        if turn_is_busy(turn):
            status_text = phase_label(turn.status, turn.output, tick=tick + turn_number)
        else:
            status_text = turn.status.value
        started = _clock_time(turn.started_at)
        completed = _clock_time(turn.completed_at) if turn.completed_at else "→"
        elapsed = _turn_elapsed_label(turn)
        suffix = f" · {elapsed}" if elapsed else ""
        append_meta_line(f"{started} → {completed} · {turn.id} · {status_text}{suffix}")
        stalled = stalled_message(turn.status, elapsed_seconds)
        if stalled:
            append_meta_line(stalled)
        recovery_status = format_turn_recovery_status(turn)
        if recovery_status is not None:
            append_meta_line(f"recovery · {recovery_status}", error="interrupted" in recovery_status)
        for line in recent_activity_lines(turn, show_details=show_activity_details):
            style = "class:tool.row"
            if "[started]" in line:
                style = "class:tool.run"
            append_content_line(f"✓ {line}", style=style)
        resume_context = getattr(turn, "metadata", {}).get("resume_context", {})
        if isinstance(resume_context, dict) and resume_context.get("injected"):
            summary_id = resume_context.get("injected_summary_id") or "<none>"
            turn_count = len(resume_context.get("injected_turn_ids", []))
            chars = resume_context.get("context_char_count", 0)
            append_ccg_note(
                "/context",
                f"resume context injected · summary {summary_id} · turns {turn_count} · {chars} chars",
            )
        if turn.error is not None:
            append_meta_line(f"error · {turn.error.message}", error=True)

    def append_interrupted_turn(turn, turn_number: int) -> None:
        label = turn.backend.value.upper()
        recovery = turn.metadata.get("recovery") if isinstance(turn.metadata, dict) else {}
        recovery_state = recovery.get("state", "interrupted") if isinstance(recovery, dict) else "interrupted"
        terminal_seen = recovery.get("terminal_event_seen") if isinstance(recovery, dict) else None
        header = (
            "● TURN INTERRUPTED · "
            f"error.kind={turn.error.kind if turn.error else 'interrupted'} · "
            f"recovery.state={recovery_state} · "
            f"terminal_event_seen={str(terminal_seen).lower()}"
        )
        append_role_line(
            label,
            turn_number,
            header,
            gutter_style="class:gutter.err",
            text_style="class:interrupted",
        )
        message = turn.error.message if turn.error is not None else "backend process exited before a terminal event"
        append_content_line(message, style="class:role.error")
        append_content_line("┌─ partial output · non-authoritative ─────────────────────────", style="class:interrupted.text")
        partial_lines = format_markdown_lines(turn.output) if turn.output.strip() else ["<empty>"]
        for line in partial_lines[:8]:
            append_content_line(f"│ {line}", style="class:interrupted.text")
        if len(partial_lines) > 8 or turn_has_partial_output(turn):
            append_content_line("│ [ truncated ]", style="class:role.error")
        append_content_line("└──────────────────────────────────────────────────────────────", style="class:interrupted.text")
        append_content_line("[r] retry turn  [c] inspect partial  [s] summarize-then-resume  [h] handoff", style="class:role.meta")
        append_turn_meta(turn, turn_number)

    if not turns:
        fragments.append(("class:section", " SESSION OPENED "))
        fragments.append(("class:rule", "────────────────────────────────────────────────────────────\n\n"))
        append_ccg_note("ready", "No messages yet. Send a prompt to start the session.")
    else:
        for index, turn in enumerate(turns):
            turn_number = index + 1
            if index == 0:
                fragments.append(("class:section", " SESSION OPENED "))
                fragments.append(("class:rule", f"{_clock_time(turn.started_at):─>54}\n\n"))
            else:
                fragments.append(("class:rule", " " * gutter_width + "  ─────────────────────────────────────────────────────\n"))
            if index < len(turns) - 1:
                append_role_line(
                    "YOU",
                    turn_number,
                    _single_line(turn.prompt, max_chars=92),
                    gutter_style="class:gutter.role",
                    text_style="class:role.user",
                )
                if turn.output.strip():
                    preview = _single_line(_first_nonempty_line(turn.output), max_chars=92)
                elif turn.status in {TurnStatus.SUBMITTING, TurnStatus.STREAMING}:
                    preview = phase_label(turn.status, turn.output, tick=tick + index)
                else:
                    preview = "<empty>"
                append_role_line(
                    turn.backend.value.upper(),
                    turn_number,
                    f"✓ {preview}",
                    gutter_style=f"class:backend.{turn.backend.value}",
                    text_style="class:role.assistant",
                )
                append_meta_line(f"{_clock_time(turn.started_at)} · {turn.id} · {turn.status.value}")
                fragments.append(("", "\n"))
                continue
            append_role_line(
                "YOU",
                turn_number,
                turn.prompt,
                gutter_style="class:gutter.role",
                text_style="class:role.user",
            )
            append_meta_line(f"{_clock_time(turn.started_at)} · {turn.id} · submitted")
            fragments.append(("", "\n"))
            recovery_status = format_turn_recovery_status(turn)
            if recovery_status is not None and "interrupted" in recovery_status:
                append_interrupted_turn(turn, turn_number)
                fragments.append(("", "\n"))
                continue
            if turn.output.strip():
                output_lines = format_markdown_lines(turn.output)
            elif turn.status in {TurnStatus.SUBMITTING, TurnStatus.STREAMING}:
                output_lines = [phase_label(turn.status, turn.output, tick=tick + index)]
            else:
                output_lines = ["<empty>"]
            append_role_line(
                turn.backend.value.upper(),
                turn_number,
                output_lines[0],
                gutter_style=f"class:backend.{turn.backend.value}",
                text_style="class:role.assistant",
            )
            for extra_line in output_lines[1:]:
                append_content_line(extra_line)
            append_turn_meta(turn, turn_number)
            fragments.append(("", "\n"))
    summary_lines = latest_summary_lines(controller)
    if summary_lines:
        fragments.append(("class:rule", " " * gutter_width + "  ─────────────────────────────────────────────────────\n"))
        append_ccg_note("/summarize", _single_line(summary_lines[2] if len(summary_lines) > 2 else "latest summary checkpoint", max_chars=86))
        for line in summary_lines[3:8]:
            append_content_line(line, style="class:role.meta")
    while fragments and fragments[-1] == ("", "\n"):
        fragments.pop()
    return fragments


def run_prompt_toolkit_interface(
    controller_factory: Callable[[str], SessionController],
    initial_backend: str | None = None,
) -> int:
    install_prompt_toolkit_shift_enter_sequences()

    from prompt_toolkit.application import Application, run_in_terminal
    from prompt_toolkit.filters import Condition, completion_is_selected, has_focus
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import FloatContainer, HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.containers import Float as LayoutFloat
    from prompt_toolkit.layout.containers import ConditionalContainer
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.widgets import TextArea

    controller = controller_factory(initial_backend) if initial_backend else None
    state = {
        "busy": False,
        "message": "Type a prompt. Enter submits. Shift-Enter adds a newline." if controller else "Select a backend to start.",
        "tick": 0,
        "active_started_at": None,
        "selected_backend": initial_backend or "codex",
        "show_activity_details": False,
        "sidebar_groups": {
            "state": True,
            "session": True,
            "composer": True,
            "lineage": False,
            "last activity": False,
            "quick actions": True,
        },
        "model_picker": False,
        "model_index": 0,
        "permission_picker": False,
        "permission_index": _default_permission_index(),
        "local_panel": None,
        "slash_palette": False,
    }

    def current_controller() -> SessionController:
        assert controller is not None
        return controller

    def latest_interrupted_turn() -> TurnRecord | None:
        if controller is None:
            return None
        turn = _latest_status_turn(current_controller())
        recovery_status = format_turn_recovery_status(turn)
        if recovery_status is not None and "interrupted" in recovery_status:
            return turn
        return None

    @Condition
    def picker_visible() -> bool:
        return controller is None

    @Condition
    def session_visible() -> bool:
        return controller is not None

    @Condition
    def model_picker_visible() -> bool:
        return controller is not None and bool(state["model_picker"])

    @Condition
    def permission_picker_visible() -> bool:
        return controller is not None and bool(state["permission_picker"])

    @Condition
    def local_panel_visible() -> bool:
        return controller is not None and state["local_panel"] is not None

    def toggle_sidebar_group(group_label: str) -> None:
        groups = state["sidebar_groups"]
        if group_label not in groups:
            return
        if group_label == "quick actions":
            refresh(message="Quick actions stay visible.", busy=state["busy"])
            return
        groups[group_label] = not bool(groups[group_label])
        refresh(
            message=f"{group_label.title()} {'expanded' if groups[group_label] else 'collapsed'}",
            busy=state["busy"],
        )

    @Condition
    def product_picker_hidden() -> bool:
        return not bool(state["model_picker"]) and not bool(state["permission_picker"])

    @Condition
    def interrupted_recovery_visible() -> bool:
        return (
            controller is not None
            and product_picker_hidden()
            and latest_interrupted_turn() is not None
            and not state["busy"]
            and not composer.text.strip()
        )

    @Condition
    def slash_palette_visible() -> bool:
        if controller is None or not session_visible():
            return False
        if not product_picker_hidden():
            return False
        if not state["slash_palette"] or not composer.buffer.document.current_line_before_cursor.startswith("/"):
            return False
        complete_state = composer.buffer.complete_state
        return bool(complete_state and complete_state.completions)

    def _workspace_label(path: Path) -> str:
        home = Path.home()
        try:
            return "~/" + str(path.resolve().relative_to(home.resolve()))
        except ValueError:
            return str(path)

    def _chrome_fragments():
        terminal_size = shutil.get_terminal_size(fallback=(0, 0))
        path = Path(getattr(controller, "cwd", Path.cwd())) if controller is not None else Path.cwd()
        live_style = "class:chrome.live" if controller is None or state["busy"] or _session_state_label(current_controller(), is_busy=False) != "interrupted" else "class:role.error"
        return [
            (live_style, " ● "),
            ("class:chrome", "ccg-tui  "),
            ("class:chrome.path", _workspace_label(path)),
            ("class:chrome", "     tty/0"),
            ("class:chrome", f" · {terminal_size.columns}×{terminal_size.lines} "),
        ]

    def _header_fragments():
        workspace = Path.cwd().name or "/"
        if controller is None:
            return [
                ("class:header.brand", " CCG TUI "),
                ("class:header.dim", "  Select Backend "),
                ("class:header.value", "active backend "),
                ("class:header.dim", f" workspace {workspace} "),
            ]
        active_controller = current_controller()
        backend = active_controller.session.backend.value
        state_label = _session_state_label(active_controller, is_busy=state["busy"])
        return [
            ("class:header.brand", " CCG TUI "),
            ("class:header.dim", "  "),
            ("class:header.value", f"[{backend}] "),
            (f"class:backend.{backend}", f"[{backend_glyph(backend)}]"),
            ("class:header.label", " backend "),
            ("class:header.value", f"{backend}  "),
            ("class:header.dim", "│ "),
            ("class:header.label", "model "),
            ("class:header.value", f"{_single_line(current_model_label(active_controller), max_chars=34)}  "),
            ("class:header.dim", "│ "),
            ("class:header.label", "perms "),
            ("class:header.value", f"{_single_line(current_permission_label(active_controller), max_chars=28)}  "),
            ("class:header.dim", "│ "),
            (_session_state_style(active_controller, is_busy=state["busy"]), f" ● {state_label} "),
        ]

    def _header_meta_fragments():
        if controller is None:
            return [
                ("class:header.meta", " select "),
                ("class:header.value", "backend"),
                ("class:header.dim", " · one backend per session · explicit user action required "),
            ]
        active_controller = current_controller()
        turn_count, last_status, resumable = _session_continuation_state(active_controller)
        return [
            ("class:header.meta", " session "),
            ("class:header.value", _single_line(active_controller.session.id, max_chars=28)),
            ("class:header.dim", " · vendor "),
            ("class:header.value", _single_line(getattr(active_controller.session, "vendor_session_id", None) or "pending", max_chars=22)),
            ("class:header.dim", " · workspace "),
            ("class:header.value", _single_line(Path(active_controller.cwd).name or "/", max_chars=22)),
            ("class:header.dim", " · task "),
            ("class:header.value", _active_task_short_label(active_controller, max_chars=30)),
            ("class:header.dim", " · turns "),
            ("class:header.value", f"{turn_count:02d}/{turn_count:02d}"),
            ("class:header.dim", " · last "),
            ("class:header.value", last_status),
            ("class:header.dim", " · resume "),
            ("class:header.value", "yes" if resumable else "no"),
            ("class:header.dim", " "),
        ]

    def _picker_fragments():
        fragments: list[tuple[str, str]] = [
            ("class:section", "Select Backend\n"),
            ("class:role.meta", "One backend per session. Authentication is vendor-native; CCG never auto-switches.\n\n"),
        ]
        for backend in BACKEND_CHOICES:
            active = state["selected_backend"] == backend
            marker = "›" if active else " "
            index = BACKEND_CHOICES.index(backend) + 1
            row_style = "class:picker.active" if active else "class:picker"
            fragments.append((row_style, f" {marker} {index}. {backend:<8} [{backend_glyph(backend)}] "))
            fragments.append(("class:role.meta", f"{backend_vendor_label(backend)} · cli auth: vendor-native · "))
            fragments.append(("class:tag.ok", " authenticated "))
            fragments.append(("class:tag", " explicit "))
            fragments.append(("", "\n"))
        fragments.append(("", "\n"))
        fragments.append(("class:role.meta", "1 2 3 direct · ↑↓ navigate · ↵ start · esc cancel · no automatic switching\n"))
        return fragments[:-1]

    def _model_picker_fragments():
        if controller is None:
            return [("class:role.meta", "")]
        fragments: list[tuple[str, str]] = []
        active_controller = current_controller()
        backend = active_controller.session.backend.value
        active_model = current_model(active_controller)
        options = model_options_for_backend(backend)
        fragments.extend(
            [
                ("class:section", f"Model Picker · {backend}\n"),
                ("class:role.meta", "Current: "),
                ("class:header.value", f"{current_model_label(active_controller)}"),
                ("class:role.meta", " · /model <value> applies without opening this picker\n\n"),
            ]
        )
        for index, option in enumerate(options):
            selected = index == state["model_index"]
            current = option.value == active_model
            row_style = "class:picker.active" if selected else "class:picker"
            marker = "●" if current else "○"
            value = option.value or "default"
            fragments.append((row_style, f" {marker} {index + 1:02d} {option.label:<24} "))
            fragments.append(("class:role.meta", f"{value:<28} "))
            if current:
                fragments.append(("class:tag.ok", " active "))
            fragments.append(("class:role.meta", f" {_single_line(option.description, max_chars=70)}\n"))
        fragments.append(("", "\n"))
        fragments.append(("class:role.meta", "↑↓ navigate · ↵ apply · esc cancel\n"))
        return fragments

    def _permission_picker_fragments():
        if controller is None:
            return [("class:role.meta", "")]
        fragments: list[tuple[str, str]] = []
        active_controller = current_controller()
        backend = active_controller.session.backend.value
        current = current_permission_values(active_controller)
        fragments.extend(
            [
                ("class:section", "Permissions Picker\n"),
                ("class:role.meta", f"Active backend: {backend} · current: {current_permission_label(active_controller)}\n"),
                ("class:role.meta", "Full backend mapping is shown inline before applying.\n\n"),
            ]
        )
        for index, option in enumerate(PERMISSION_OPTIONS):
            selected = index == state["permission_index"]
            values = permission_option_values_for_backend(option, backend)
            active = values == current
            row_style = "class:picker.active" if selected else "class:picker"
            marker = "●" if active else "○"
            risk_style = "class:tag.err" if option.key == "full-access" else "class:tag.warn" if option.key == "auto-edit" else "class:tag.ok" if active else "class:tag"
            fragments.append((row_style, f" {marker} {index + 1}. {option.label:<22} "))
            fragments.append((risk_style, f" {option.key} "))
            fragments.append(("class:role.meta", f" {_single_line(option.description, max_chars=76)}\n"))
            for mapped_backend in BACKEND_CHOICES:
                mapped = permission_values_for_backend(option.key, mapped_backend)
                rendered = ", ".join(f"{key}={value}" for key, value in mapped.items())
                fragments.append(("class:gutter", f"    {mapped_backend:<6} "))
                fragments.append(("class:role.meta", f"{rendered}\n"))
        fragments.append(("", "\n"))
        fragments.append(("class:tag.warn", " widening permissions is never automatic "))
        fragments.append(("class:role.meta", " · ↑↓ navigate · ↵ apply · esc cancel\n"))
        return fragments

    def _slash_palette_fragments():
        if controller is None:
            return [("class:role.meta", "")]
        if not state["slash_palette"] or not composer.buffer.document.current_line_before_cursor.startswith("/"):
            return [("class:role.meta", "")]
        complete_state = composer.buffer.complete_state
        if complete_state is None or not complete_state.completions:
            return [("class:role.meta", "")]
        backend = current_controller().session.backend.value
        commands = ordered_slash_commands_for_palette(tuple(complete_state.completions), backend=backend)
        selected_index = max(0, min(complete_state.complete_index or 0, len(commands) - 1))
        visible_start, visible_end = _slash_palette_visible_range(len(commands), selected_index)
        terminal_width = shutil.get_terminal_size(fallback=(100, 32)).columns
        body_width = max(48, terminal_width - 10)
        description_width = max(20, body_width - 42)
        fragments: list[tuple[str, str]] = []
        for index, command in enumerate(commands[visible_start:visible_end], start=visible_start):
            command_name = getattr(command, "name", None) or getattr(command, "text")
            command_description = getattr(command, "description", None)
            if command_description is None:
                command_description = getattr(command, "display_meta_text", "")
            group = slash_command_palette_group(command_name, backend=backend)
            marker = "›" if index == selected_index else " "
            row_style = "class:slash.active" if index == selected_index else "class:slash.row"
            description = _single_line(str(command_description), max_chars=description_width)
            line = f" {marker} {index + 1:02d} {group:<11} {command_name:<18} {description}"
            fragments.append((row_style, _single_line(line, max_chars=body_width) + "\n"))
        return fragments

    def _block_bar(fraction: float, width: int = 24) -> str:
        filled = max(0, min(width, int(round(width * fraction))))
        return "█" * filled + "░" * (width - filled)

    def _panel_header(command: str, title: str, subtitle: str, *, tag: str = "local") -> list[tuple[str, str]]:
        return [
            ("class:section", f"{title}\n"),
            ("class:role.meta", f"{command} · {subtitle} · "),
            ("class:tag.ok" if tag == "ok" else "class:tag.warn" if tag == "warn" else "class:tag", f" {tag} "),
            ("", "\n"),
            ("class:rule", "─" * 88 + "\n"),
        ]

    def _plain_panel_fragments(
        *,
        command: str,
        title: str,
        subtitle: str,
        lines: list[str],
        tag: str = "local",
        footer: str | None = None,
    ) -> list[tuple[str, str]]:
        fragments = _panel_header(command, title, subtitle, tag=tag)
        for line in lines:
            fragments.append(("class:role.meta" if line.startswith(("  ", "·", "│", "└", "├", "┌")) else "class:role.assistant", f"{line}\n"))
        if footer:
            fragments.append(("class:rule", "─" * 88 + "\n"))
            fragments.append(("class:role.meta", footer + "\n"))
        return fragments

    def _panel_trim(text: str, width: int) -> str:
        if width <= 0:
            return ""
        if len(text) <= width:
            return text
        if width <= 3:
            return text[:width]
        return text[: width - 3].rstrip() + "..."

    def _panel_pad(text: str, width: int) -> str:
        trimmed = _panel_trim(text, width)
        return trimmed + (" " * (width - len(trimmed)))

    def _panel_box(title: str, body_lines: list[str], *, width: int) -> list[str]:
        inner_width = max(0, width - 2)
        header = f" {title} "
        header = _panel_trim(header, inner_width)
        left_fill = max(0, (inner_width - len(header)) // 2)
        right_fill = max(0, inner_width - len(header) - left_fill)
        lines = [
            f"┌{'─' * left_fill}{header}{'─' * right_fill}┐",
        ]
        for line in body_lines:
            lines.append(f"│{_panel_pad(line, inner_width)}│")
        lines.append(f"└{'─' * inner_width}┘")
        return lines

    def _panel_join_columns(left_lines: list[str], right_lines: list[str], *, left_width: int, right_width: int, gap: int = 4) -> list[str]:
        total_lines = max(len(left_lines), len(right_lines))
        combined: list[str] = []
        spacer = " " * gap
        for index in range(total_lines):
            left = left_lines[index] if index < len(left_lines) else ""
            right = right_lines[index] if index < len(right_lines) else ""
            combined.append(f"{_panel_pad(left, left_width)}{spacer}{_panel_pad(right, right_width)}")
        return combined

    def _panel_kv(label: str, value: str, *, label_width: int = 15) -> str:
        return f"{label:<{label_width}} {value}"

    def _capability_status_cell(status: str) -> str:
        return f"{status:<8}"

    def _compatibility_tag(compatibility) -> str:
        if compatibility.widens_permissions:
            return "wider"
        if compatibility.source_state.level > compatibility.target_state.level:
            return "narrower"
        return "equivalent"

    def _format_permission_values(values: dict[str, str]) -> str:
        return ", ".join(f"{key}={value}" for key, value in values.items()) or "<none>"

    def _build_capability_matrix_lines() -> list[str]:
        rows = [
            ("long-context refactor", {"codex": "fit", "claude": "fit", "gemini": "partial", "antigravity": "partial"}, "vendor limits differ; handoff remains manual"),
            ("web search grounded", {"codex": "no", "claude": "partial", "gemini": "fit", "antigravity": "fit"}, "google-backed CLIs fit grounded search"),
            ("rapid in-loop edits", {"codex": "fit", "claude": "partial", "gemini": "partial", "antigravity": "partial"}, "codex tool latency is lowest in-loop"),
            ("plan-then-execute", {"codex": "partial", "claude": "fit", "gemini": "partial", "antigravity": "fit"}, "antigravity shares the newer google agent path"),
            ("large diff review", {"codex": "partial", "claude": "fit", "gemini": "fit", "antigravity": "fit"}, "google and claude paths fit broad context"),
            ("shell workspace", {"codex": "fit", "claude": "fit", "gemini": "fit", "antigravity": "fit"}, "permission widening stays manual"),
        ]
        header = "task area                 codex    claude   gemini   antig"
        lines = [header, " " * len(header)]
        for area, statuses, note in rows:
            row = (
                f"{area:<24} "
                f"{_capability_status_cell(statuses['codex'])} "
                f"{_capability_status_cell(statuses['claude'])} "
                f"{_capability_status_cell(statuses['gemini'])} "
                f"{_capability_status_cell(statuses['antigravity'])}"
            )
            lines.append(row.rstrip())
            lines.append(f"note · {note}")
        return lines

    def _capabilities_panel_fragments(active_controller: SessionController) -> list[tuple[str, str]]:
        source_backend = active_controller.session.backend
        source_values = current_permission_values(active_controller)
        fragments = _panel_header(
            "/capabilities",
            "Routing Capability Registry",
            "advisory routing matrix · local audit entry",
            tag="local",
        )
        fragments.append(("class:role.meta", "fit / partial / no reflect routing fit, not capability depth\n"))
        fragments.append(("class:rule", "─" * 88 + "\n"))
        left_box = _panel_box("capability matrix", _build_capability_matrix_lines(), width=52)
        right_lines: list[str] = []
        for target in BackendName:
            compatibility = compare_permission_compatibility(source_backend, source_values, target)
            tag = _compatibility_tag(compatibility)
            values = _format_permission_values(compatibility.target_state.values)
            right_lines.extend(
                [
                    f"{target.value} compatibility",
                    f"{source_backend.value} ({compatibility.source_state.label}) → {target.value} ({compatibility.target_state.label})",
                    f"{tag} · {values}",
                    compatibility.reason,
                    "",
                ]
            )
        right_box = _panel_box("permission compatibility", right_lines, width=32)
        for line in _panel_join_columns(left_box, right_box, left_width=52, right_width=32, gap=4):
            fragments.append(("class:role.meta", line + "\n"))
        fragments.append(("class:rule", "─" * 88 + "\n"))
        fragments.append(("class:role.meta", "this view records a routing audit entry but never switches anything\n"))
        return fragments

    def _handoff_panel_fragments(args: str, *, ok: bool, preview: str, status_line: str) -> list[tuple[str, str]]:
        tag = "local" if ok else "warn"
        fragments = _panel_header(
            "/handoff",
            "Handoff preview:",
            "cross-backend packet · does not submit",
            tag=tag,
        )
        if ok:
            preview_data = _build_handoff_preview_data(
                current_controller().session,
                args,
                source_permission_values=current_permission_values(current_controller()),
            )
            assert preview_data.packet is not None
            packet = preview_data.packet
            audit = packet.metadata.get("audit", {})
            audit_turns = audit.get("turns", {}) if isinstance(audit, dict) else {}
            selected_context = packet.metadata.get("selected_context", {})
            source_turn_ids = packet.metadata.get("source_turn_ids", [])
            source_summary_id = packet.metadata.get("source_summary_id") or "<none>"
            source_scope = packet.metadata.get("source_scope") or "session"
            target_backend = packet.metadata.get("target_backend") or "<unknown>"
            target_model = packet.metadata.get("target_model") or "<default>"
            max_context_chars = packet.metadata.get("max_context_chars", 0)
            context_char_count = packet.metadata.get("context_char_count", 0)
            source_backend = packet.metadata.get("source_backend") or current_controller().session.backend.value
            permission_values = current_permission_values(current_controller())
            compatibility_state = compare_permission_compatibility(source_backend, permission_values, target_backend)
            compatibility = format_permission_compatibility_line(
                source_backend,
                permission_values,
                str(target_backend),
            )

            source_lines = _panel_box(
                "source",
                [
                    f"[{backend_glyph(source_backend)}] {source_backend}",
                    f"session {packet.metadata.get('source_session_id') or '<unknown>'}",
                    f"scope   {source_scope}",
                    f"summary {source_summary_id}",
                    f"turns   {len(source_turn_ids)}",
                ],
                width=42,
            )
            target_lines = _panel_box(
                "target",
                [
                    f"[{backend_glyph(target_backend)}] {target_backend}",
                    f"model   {target_model}",
                    f"status  {_compatibility_tag(compatibility_state)}",
                    f"compat  {compatibility}",
                    "execute required for fork",
                ],
                width=42,
            )
            fragments.append(("class:role.meta", status_line + "\n"))
            fragments.append(("class:role.meta", f"args: {args or '<missing>'}\n"))
            fragments.append(("class:rule", "─" * 88 + "\n"))
            for line in _panel_join_columns(source_lines, target_lines, left_width=42, right_width=42, gap=4):
                fragments.append(("class:role.meta", line + "\n"))
            fragments.append(("class:rule", "─" * 88 + "\n"))

            context_lines: list[str] = [
                f"budget {context_char_count:,} / {max_context_chars:,} chars",
                f"source {packet.metadata.get('source_session_id') or '<unknown>'} · {source_backend}",
                f"target {target_backend} · {target_model}",
                f"summary {source_summary_id}",
                f"selected {', '.join(selected_context.get('source_turn_ids', [])) or '<none>'}",
                "",
            ]
            for line in packet.context_text.splitlines()[:8]:
                context_lines.append(line)
            if len(packet.context_text.splitlines()) > 8:
                context_lines.append(f"[ truncated · +{len(packet.context_text.splitlines()) - 8} lines ]")

            audit_lines = _panel_box(
                "audit selection trace",
                [
                    _panel_kv("included", ", ".join(audit_turns.get("included_source_ids", [])) or "<none>", label_width=12),
                    _panel_kv("selected", ", ".join(audit_turns.get("selected_before_recent_limit_source_ids", [])) or "<none>", label_width=12),
                    _panel_kv("pre-context", ", ".join(audit_turns.get("selected_before_context_limit_source_ids", [])) or "<none>", label_width=12),
                    _panel_kv("excluded", ", ".join(
                        f"{item.get('source_id', '<unknown>')}={item.get('reason', 'unknown')}"
                        for item in audit_turns.get("excluded_source_ids", [])
                        if isinstance(item, dict)
                    ) or "<none>", label_width=12),
                    _panel_kv("recent cap", str(audit.get("limits", {}).get("recent_turn_limit", packet.metadata.get("recent_turn_limit", "<default>"))), label_width=12),
                    _panel_kv("context cap", f"{max_context_chars:,} chars", label_width=12),
                    _panel_kv("truncation", f"recent={audit.get('truncation', {}).get('dropped_for_recent_limit_count', 0)} context={audit.get('truncation', {}).get('dropped_for_context_limit_count', 0)}", label_width=12),
                ],
                width=42,
            )
            lineage_lines = _panel_box(
                "lineage · on execute",
                [
                    _panel_kv("kind", "handoff", label_width=10),
                    _panel_kv("parent", packet.metadata.get("source_session_id") or "<unknown>", label_width=10),
                    _panel_kv("forked_from", source_turn_ids[-1] if source_turn_ids else "<none>", label_width=10),
                    _panel_kv("relationships", "parent · handoff", label_width=10),
                ],
                width=42,
            )
            action_lines = _panel_box(
                "action stack",
                [
                    "--handoff-execute · start new target session",
                    "export packet to runtime/handoffs/…",
                    "copy backend prompt",
                    "",
                    "↵ execute   e export   c copy",
                ],
                width=42,
            )
            left_box = _panel_box("rendered context preview", context_lines, width=42)
            right_stack = audit_lines + [" " * 42] + lineage_lines + [" " * 42] + action_lines
            for line in _panel_join_columns(left_box, right_stack, left_width=42, right_width=42, gap=4):
                fragments.append(("class:role.meta", line + "\n"))
        else:
            fragments.append(("class:role.error", preview + "\n"))
            fragments.append(("class:role.meta", f"args: {args or '<missing>'}\n"))
        fragments.append(("class:rule", "─" * 88 + "\n"))
        fragments.append(("class:tag.warn", " execution requires explicit confirmation · no auto-rotate "))
        fragments.append(("class:role.meta", " · use --handoff-execute outside the preview path\n"))
        return fragments

    def _context_panel_fragments(active_controller: SessionController) -> list[tuple[str, str]]:
        payload = active_controller.preview_resume_context() if hasattr(active_controller, "preview_resume_context") else None
        if payload is None:
            return _plain_panel_fragments(
                command="/context",
                title="Resume context preview",
                subtitle="no pending resume context",
                lines=["No resume context pending.", "visible-prompt · unchanged"],
            )
        metadata = payload.metadata
        turn_ids = metadata.get("injected_turn_ids", [])
        lines = [
            f"mode        auto",
            f"turns       {len(turn_ids)} / {len(turn_ids)}",
            f"summary id  {metadata.get('injected_summary_id') or '<none>'}",
            f"size        {metadata.get('context_char_count', len(payload.context_text))} chars",
            "injected as single JSON-escaped line",
            "visible-prompt (unchanged)",
            "",
            "preview",
            *payload.context_text.splitlines()[:20],
        ]
        if len(payload.context_text.splitlines()) > 20:
            lines.append("[ truncated ]")
        return _plain_panel_fragments(
            command="/context",
            title="Resume context preview",
            subtitle="pending resume context",
            lines=lines,
        )

    def _summarize_panel_fragments(*, phase: str, detail: str, summary_id: str | None = None) -> list[tuple[str, str]]:
        fraction = 1.0 if summary_id else 0.62 if phase == "running" else 0.0
        tag = "ok" if summary_id else "local" if phase == "running" else "warn"
        lines = [
            f"{_block_bar(fraction)} {int(fraction * 100):>3}% · gemini-2.5-flash",
            f"00:00 scan · loading transcript",
            f"00:01 scan · filtering task/session scope",
            f"00:02 call · gemini summary backend",
            f"00:08 {phase} · {detail}",
        ]
        if summary_id:
            lines.append(f"summary saved · {summary_id}")
        return _plain_panel_fragments(
            command="/summarize",
            title="Summary checkpoint",
            subtitle="Gemini-backed progress panel",
            lines=lines,
            tag=tag,
            footer="UI stays responsive · checkpoint appends to the source transcript on completion",
        )

    def _interrupted_partial_panel_fragments(turn: TurnRecord) -> list[tuple[str, str]]:
        recovery_status = format_turn_recovery_status(turn) or "interrupted; no terminal event"
        message = turn.error.message if turn.error is not None else "backend process exited before a terminal event"
        partial_lines = format_markdown_lines(turn.output) if turn.output.strip() else ["<empty>"]
        fragments = _panel_header(
            "[c]",
            "Partial output preview",
            "interrupted turn · local only",
            tag="warn",
        )
        fragments.append(("class:role.error", f"{message}\n"))
        fragments.append(("class:role.meta", f"turn      {turn.id}\n"))
        fragments.append(("class:role.meta", f"recovery  {recovery_status}\n"))
        fragments.append(("class:rule", "─" * 88 + "\n"))
        fragments.append(("class:interrupted.text", "┌─ partial output · non-authoritative ─────────────────────────\n"))
        for line in partial_lines[:10]:
            fragments.append(("class:interrupted.text", f"│ {line}\n"))
        if len(partial_lines) > 10 or turn_has_partial_output(turn):
            fragments.append(("class:role.error", "│ [ truncated ]\n"))
        fragments.append(("class:interrupted.text", "└──────────────────────────────────────────────────────────────\n"))
        fragments.append(("class:role.meta", "press [r] to retry, [s] to summarize-then-resume, or [h] for handoff\n"))
        return fragments

    def _summary_then_resume_panel_fragments(summary_id: str) -> list[tuple[str, str]]:
        fragments = _summarize_panel_fragments(
            phase="completed",
            detail="checkpoint persisted",
            summary_id=summary_id,
        )
        fragments.append(("", "\n"))
        fragments.extend(_context_panel_fragments(current_controller()))
        return fragments

    def _current_panel_fragments() -> list[tuple[str, str]]:
        panel = state["local_panel"]
        return list(panel) if isinstance(panel, list) else []

    transcript_control = FormattedTextControl(
        lambda: (
            _current_panel_fragments()
            + ([("", "\n\n")] if _current_panel_fragments() else [])
            + _build_prompt_toolkit_transcript_fragments(
                current_controller(),
                tick=state["tick"],
                elapsed_seconds=(time.monotonic() - state["active_started_at"]) if state["active_started_at"] else 0.0,
                show_activity_details=state["show_activity_details"],
            )
        )
        if controller is not None
        else [("class:role.meta", "")],
        focusable=False,
    )
    transcript_window = Window(content=transcript_control, wrap_lines=True, always_hide_cursor=True)
    sidebar_window = Window(
        content=FormattedTextControl(
            lambda: _build_prompt_toolkit_sidebar_fragments(
                current_controller(),
                is_busy=state["busy"],
                draft_text=composer.text,
                sidebar_groups=state["sidebar_groups"],
                toggle_sidebar_group=toggle_sidebar_group,
            )
            if controller is not None
            else [("class:sidebar", "")]
        ),
        width=40,
        wrap_lines=True,
    )
    picker_window = Window(content=FormattedTextControl(_picker_fragments), wrap_lines=True, always_hide_cursor=True)
    model_picker_window = Window(content=FormattedTextControl(_model_picker_fragments), wrap_lines=True, always_hide_cursor=True)
    permission_picker_window = Window(content=FormattedTextControl(_permission_picker_fragments), wrap_lines=True, always_hide_cursor=True)
    composer = TextArea(
        height=4,
        prompt=lambda: [("class:composer.prompt", f"{spinner_frame(state['tick']) if state['busy'] else '›'} ")],
        multiline=True,
        wrap_lines=True,
        history=InMemoryHistory(),
        completer=SlashCommandCompleter(lambda: current_controller().session.backend.value if controller is not None else None),
        complete_while_typing=True,
        style="class:composer",
        scrollbar=False,
        accept_handler=lambda buffer: False,
    )

    def _composer_submit_cluster_fragments():
        is_busy = state["busy"]
        is_disabled = controller is None or is_busy
        label = "sending…" if is_busy else "submit"
        key_label = "↵"
        inner_width = 14
        content_width = len(label) + 1 + len(key_label)
        padding = max(0, inner_width - content_width)
        left_padding = padding // 2
        right_padding = padding - left_padding
        border_style = "class:composer.submit.border.disabled" if is_disabled else "class:composer.submit.border"
        label_style = "class:composer.submit.label.disabled" if is_disabled else "class:composer.submit.label"
        key_style = "class:composer.submit.key.disabled" if is_disabled else "class:composer.submit.key"
        hint_style = "class:composer.submit.hint.disabled" if is_disabled else "class:composer.submit.hint"
        return [
            (border_style, "╭" + "─" * inner_width + "╮\n"),
            (border_style, "│"),
            (label_style, " " * left_padding + label),
            (key_style, f" {key_label}"),
            (label_style, " " * right_padding),
            (border_style, "│\n"),
            (border_style, "╰" + "─" * inner_width + "╯\n"),
            (hint_style, " enter to send "),
        ]

    submit_cluster = Window(
        width=16,
        height=4,
        content=FormattedTextControl(_composer_submit_cluster_fragments),
        wrap_lines=False,
        always_hide_cursor=True,
        dont_extend_width=True,
        dont_extend_height=True,
    )

    def _section_fragments(label: str, *, extra: str = ""):
        return [
            ("class:section", f" {label.upper()} "),
            ("class:rule", "─" * 24),
            ("class:section", f" {extra}" if extra else ""),
        ]

    conversation_bar = Window(
        height=1,
        content=FormattedTextControl(
            lambda: _section_fragments(
                "Conversation",
                extra="F2 refresh · F3 details" if controller is not None else "",
            )
        ),
    )
    sidebar_bar = Window(height=1, content=FormattedTextControl(lambda: _section_fragments("Sidebar")))
    composer_bar = Window(
        height=1,
        content=FormattedTextControl(
            lambda: [
                ("class:section", " COMPOSER "),
                ("class:rule", "─" * 24),
                ("class:composer.hint", " / slash commands · Enter submit · Shift+Enter newline "),
            ]
        ),
    )
    composer_hints = Window(
        height=1,
        content=FormattedTextControl(
            lambda: [
                ("class:composer.hint", f"  / commands · {len(composer.text.splitlines()) if composer.text else 0} lines · {len(composer.text)} ch"),
                ("class:composer.hint", f" · {'will queue' if state['busy'] else 'submits to ' + current_controller().session.backend.value if controller is not None else 'select backend'}"),
                ("class:composer.hint", " · F3 details · F2 refresh · Esc cancel "),
            ]
        ),
    )
    slash_palette_window = Window(
        content=FormattedTextControl(_slash_palette_fragments),
        wrap_lines=False,
        always_hide_cursor=True,
        dont_extend_height=True,
        style="class:completion-menu",
    )

    def _footer_fragments():
        if controller is None:
            return [("class:footer", f" {state['message']} · ↑↓ navigate · 1 2 3 direct · ↵ start · esc cancel ")]
        active_controller = current_controller()
        state_label = _session_state_label(active_controller, is_busy=state["busy"])
        state_style = "class:footer.busy" if state["busy"] else "class:footer.interrupted" if state_label == "interrupted" else "class:footer.ready"
        return [
            (state_style, f" ● {state_label} "),
            ("class:footer", f" {state['message']} "),
            ("class:footer", "        ↵ submit · ⇧↵ newline · F2 refresh · F3 details · esc cancel · ^c exit "),
            ("class:footer", f" via [{backend_glyph(active_controller.session.backend)}] {active_controller.session.backend.value} · ccg-tui 0.1.0 "),
        ]

    status_bar = Window(height=1, content=FormattedTextControl(_footer_fragments))
    chrome_bar = Window(height=1, content=FormattedTextControl(_chrome_fragments))
    header_bar = Window(height=1, content=FormattedTextControl(_header_fragments))
    header_meta_bar = Window(height=1, content=FormattedTextControl(_header_meta_fragments))

    def _rule_row(left: str, fill: str, right: str, *, style: str = "class:border"):
        return VSplit(
            [
                Window(width=1, height=1, char=left, style=style),
                Window(height=1, char=fill, style=style),
                Window(width=1, height=1, char=right, style=style),
            ]
        )

    def _boxed_surface(title: str, subtitle: str, body_window: Window, footer: str):
        title_bar = Window(
            height=1,
            content=FormattedTextControl(
                lambda: [
                    ("class:picker.header", f" {title} "),
                    ("class:role.meta", f"{subtitle} "),
                ]
            ),
        )
        footer_bar = Window(
            height=1,
            content=FormattedTextControl(lambda: [("class:picker.footer", f" {footer} ")]),
        )
        return HSplit(
            [
                _rule_row("┌", "─", "┐"),
                VSplit([Window(width=1, char="│", style="class:border"), title_bar, Window(width=1, char="│", style="class:border")]),
                _rule_row("├", "─", "┤"),
                VSplit([Window(width=1, char="│", style="class:border"), body_window, Window(width=1, char="│", style="class:border")]),
                _rule_row("├", "─", "┤"),
                VSplit([Window(width=1, char="│", style="class:border"), footer_bar, Window(width=1, char="│", style="class:border")]),
                _rule_row("└", "─", "┘"),
            ]
        )

    composer_box = HSplit(
        [
            composer_bar,
            _rule_row("┌", "─", "┐", style="class:composer.border"),
            VSplit(
                [
                    Window(width=1, char="│", style="class:composer.border"),
                    composer,
                    Window(width=2, char=" ", style="class:composer"),
                    submit_cluster,
                    Window(width=1, char="│", style="class:composer.border"),
                ]
            ),
            _rule_row("└", "─", "┘", style="class:composer.border"),
            composer_hints,
        ]
    )

    def refresh(message: str | None = None, busy: bool | None = None) -> None:
        if message is not None:
            state["message"] = message
        if busy is not None:
            state["busy"] = busy
            if busy and state["active_started_at"] is None:
                state["active_started_at"] = time.monotonic()
            if not busy:
                state["active_started_at"] = None
        state["tick"] += 1
        if controller is not None:
            elapsed_seconds = (time.monotonic() - state["active_started_at"]) if state["active_started_at"] else 0.0
            if state["local_panel"] is not None:
                transcript_window.vertical_scroll = 0
            else:
                transcript_fragments = _build_prompt_toolkit_transcript_fragments(
                    current_controller(),
                    tick=state["tick"],
                    elapsed_seconds=elapsed_seconds,
                    show_activity_details=state["show_activity_details"],
                )
                transcript_line_count = sum(text.count("\n") for _, text in transcript_fragments) + 1
                transcript_window.vertical_scroll = max(
                    0,
                    transcript_line_count
                    - 6,
                )
            sidebar_fragments = _build_prompt_toolkit_sidebar_fragments(
                current_controller(),
                is_busy=state["busy"],
                draft_text=composer.text,
                sidebar_groups=state["sidebar_groups"],
                toggle_sidebar_group=toggle_sidebar_group,
            )
            sidebar_line_count = sum(fragment[1].count("\n") for fragment in sidebar_fragments) + 1
            sidebar_window.vertical_scroll = 10_000 if sidebar_line_count > 0 else 0
        app.invalidate()

    def select_backend(backend: str) -> None:
        nonlocal controller
        state["selected_backend"] = backend
        controller = controller_factory(backend)
        state["message"] = "Type a prompt. Enter submits. Shift-Enter adds a newline."
        state["busy"] = False
        state["active_started_at"] = None
        state["model_picker"] = False
        state["model_index"] = 0
        state["permission_picker"] = False
        state["permission_index"] = _default_permission_index()
        state["local_panel"] = None
        composer.buffer.text = ""
        app.layout.focus(composer)
        refresh(message=f"Started {backend} session", busy=False)

    def clear_current_session() -> None:
        nonlocal controller
        current = current_controller()
        backend = current.session.backend.value
        current.close()
        controller = controller_factory(backend)
        state["selected_backend"] = backend
        state["model_picker"] = False
        state["model_index"] = 0
        state["permission_picker"] = False
        state["permission_index"] = _default_permission_index()
        state["local_panel"] = None
        composer.buffer.text = ""
        app.layout.focus(composer)
        refresh(message=f"Started fresh {backend} session", busy=False)

    def resume_session(session_id: str) -> tuple[bool, str]:
        nonlocal controller
        current = current_controller()
        try:
            session = current.store.load_session(session_id)
        except FileNotFoundError:
            return False, f"Session not found: {session_id}"
        if session.backend != current.session.backend:
            return False, f"Session {session_id} uses {session.backend.value}; current backend is {current.session.backend.value}."
        resume_cwd = Path(session.workspace_cwd) if session.workspace_cwd else current.cwd
        controller = SessionController.resume(
            adapter=build_backend(session.backend.value),
            store=current.store,
            cwd=resume_cwd,
            session=session,
            resume_context_config=getattr(current, "resume_context_config", ResumeContextConfig(enabled=False)),
        )
        current.close()
        state["selected_backend"] = session.backend.value
        state["model_picker"] = False
        state["model_index"] = 0
        state["permission_picker"] = False
        state["permission_index"] = _default_permission_index()
        state["local_panel"] = None
        app.layout.focus(composer)
        return True, f"Resumed session: {session.id}"

    def open_model_picker() -> None:
        backend = current_controller().session.backend.value
        active_model = current_model(current_controller())
        options = model_options_for_backend(backend)
        state["model_index"] = next(
            (index for index, option in enumerate(options) if option.value == active_model),
            0,
        )
        state["model_picker"] = True
        state["permission_picker"] = False
        composer.buffer.text = ""
        app.layout.focus(model_picker_window)
        refresh(message=f"Choose model for {backend}", busy=False)

    def close_model_picker(message: str = "Model selection cancelled.") -> None:
        state["model_picker"] = False
        composer.buffer.text = ""
        app.layout.focus(composer)
        refresh(message=message, busy=False)

    def move_model_picker(offset: int) -> None:
        options = model_options_for_backend(current_controller().session.backend.value)
        state["model_index"] = (state["model_index"] + offset) % len(options)
        refresh()

    def apply_selected_model() -> None:
        option = model_options_for_backend(current_controller().session.backend.value)[state["model_index"]]
        message = apply_model_selection(current_controller(), option.value)
        close_model_picker(message)

    def open_permission_picker() -> None:
        backend = current_controller().session.backend.value
        current = current_permission_option(current_controller())
        state["permission_index"] = next(
            (index for index, option in enumerate(PERMISSION_OPTIONS) if option == current),
            _default_permission_index(),
        )
        state["permission_picker"] = True
        state["model_picker"] = False
        composer.buffer.text = ""
        app.layout.focus(permission_picker_window)
        refresh(message=f"Choose permissions for {backend}", busy=False)

    def close_permission_picker(message: str = "Permissions selection cancelled.") -> None:
        state["permission_picker"] = False
        composer.buffer.text = ""
        app.layout.focus(composer)
        refresh(message=message, busy=False)

    def move_permission_picker(offset: int) -> None:
        state["permission_index"] = (state["permission_index"] + offset) % len(PERMISSION_OPTIONS)
        refresh()

    def apply_selected_permissions() -> None:
        option = PERMISSION_OPTIONS[state["permission_index"]]
        message = apply_permission_selection(current_controller(), option)
        close_permission_picker(message)

    def close_local_panel(message: str = "Closed local command panel.") -> None:
        state["local_panel"] = None
        app.layout.focus(composer)
        refresh(message=message, busy=state["busy"])

    def start_backend_interactive(prompt: str) -> None:
        current = current_controller()
        composer.buffer.text = ""
        refresh(message=f"Handing terminal to backend for {prompt}. Press Ctrl-G to return to CCG.", busy=True)

        async def foreground_task() -> None:
            try:
                await run_in_terminal(
                    lambda: current.adapter.run_interactive_terminal(prompt, current.cwd),
                    render_cli_done=False,
                )
                refresh(message="Returned to CCG.", busy=False)
            except Exception as exc:  # pragma: no cover
                refresh(message=f"Interactive mode failed: {exc}", busy=False)

        app.create_background_task(foreground_task())

    def worker(text: str) -> None:
        try:
            current = current_controller()
            current.submit_prompt(
                text,
                on_update=lambda current_turn: refresh(
                    message=progress_message(current_turn, tick=state["tick"]),
                    busy=turn_is_busy(current_turn),
                ),
            )
            refresh(message="Ready for next prompt", busy=False)
        except Exception as exc:  # pragma: no cover
            refresh(message=f"Error: {exc}", busy=False)

    def summary_worker() -> None:
        summary_adapter = build_summary_backend()
        try:
            summary = current_controller().generate_summary(summary_adapter)
            state["local_panel"] = _summarize_panel_fragments(
                phase="completed",
                detail="checkpoint persisted",
                summary_id=summary.id,
            )
            refresh(message=f"Summary saved: {summary.id}", busy=False)
        except Exception as exc:  # pragma: no cover
            state["local_panel"] = _summarize_panel_fragments(
                phase="failed",
                detail=str(exc),
            )
            refresh(message=f"Summary failed: {exc}", busy=False)
        finally:
            summary_adapter.close()

    def summarize_then_resume_worker() -> None:
        summary_adapter = build_summary_backend()
        try:
            summary = current_controller().generate_summary(summary_adapter)
            state["local_panel"] = _summary_then_resume_panel_fragments(summary.id)
            refresh(message=f"Summary saved: {summary.id} · resume context visible", busy=False)
        except Exception as exc:  # pragma: no cover
            state["local_panel"] = _summarize_panel_fragments(
                phase="failed",
                detail=str(exc),
            )
            refresh(message=f"Summary failed: {exc}", busy=False)
        finally:
            summary_adapter.close()

    def retry_interrupted_turn() -> None:
        turn = latest_interrupted_turn()
        if turn is None:
            refresh(message="No interrupted turn to retry", busy=state["busy"])
            return
        state["local_panel"] = None
        composer.buffer.text = ""
        refresh(message=f"Retrying interrupted turn {turn.id}…", busy=True)
        threading.Thread(target=worker, args=(turn.prompt,), daemon=True).start()

    def inspect_interrupted_partial() -> None:
        turn = latest_interrupted_turn()
        if turn is None:
            refresh(message="No interrupted turn to inspect", busy=state["busy"])
            return
        state["local_panel"] = _interrupted_partial_panel_fragments(turn)
        composer.buffer.text = ""
        app.layout.focus(composer)
        refresh(message="Interrupted partial output displayed locally", busy=False)

    def summarize_then_resume_interrupted_turn() -> None:
        turn = latest_interrupted_turn()
        if turn is None:
            refresh(message="No interrupted turn to summarize", busy=state["busy"])
            return
        state["local_panel"] = _summarize_panel_fragments(
            phase="running",
            detail="checkpoint draft for interrupted recovery",
        )
        composer.buffer.text = ""
        refresh(message="Generating summary checkpoint for interrupted recovery…", busy=True)
        threading.Thread(target=summarize_then_resume_worker, daemon=True).start()

    def prefill_handoff_preview_from_interrupted_turn() -> None:
        turn = latest_interrupted_turn()
        if turn is None:
            refresh(message="No interrupted turn to hand off", busy=state["busy"])
            return
        state["local_panel"] = None
        composer.buffer.text = "/handoff "
        app.layout.focus(composer)
        refresh(message=f"Handoff preview prefilled for interrupted turn {turn.id}", busy=False)

    def submit_current_buffer() -> None:
        if controller is None:
            select_backend(state["selected_backend"])
            return
        state["slash_palette"] = False
        text = composer.text.strip()
        if not text:
            return
        parsed = parse_slash_command(text, current_controller().session.backend.value)
        if parsed is not None and parsed.action is SlashCommandAction.INTERACTIVE_BACKEND:
            if state["busy"]:
                refresh(message="Wait for the current turn to finish", busy=True)
                return
            composer.buffer.text = ""
            start_backend_interactive(parsed.backend_prompt or text)
            return
        if parsed is not None and parsed.action is SlashCommandAction.PRODUCT:
            composer.buffer.text = ""
            if parsed.canonical != "/capabilities":
                state["local_panel"] = None
            if parsed.canonical == "/quit":
                app.exit(result=0)
                return
            if parsed.canonical == "/help":
                refresh(message="Type / to open slash command suggestions; keep typing to filter.", busy=state["busy"])
                return
            if parsed.canonical == "/clear":
                if state["busy"]:
                    refresh(message="Wait for the current turn to finish", busy=True)
                    return
                clear_current_session()
                return
            if parsed.canonical == "/model":
                if state["busy"]:
                    refresh(message="Wait for the current turn to finish", busy=True)
                    return
                if parsed.args:
                    option = find_model_option(current_controller().session.backend.value, parsed.args)
                    model = option.value if option is not None else parsed.args
                    refresh(message=apply_model_selection(current_controller(), model), busy=False)
                else:
                    open_model_picker()
                return
            if parsed.canonical == "/permissions":
                if state["busy"]:
                    refresh(message="Wait for the current turn to finish", busy=True)
                    return
                if parsed.args:
                    option = find_permission_option(parsed.args)
                    if option is None:
                        refresh(message=f"Unknown permission preset: {parsed.args}", busy=False)
                    else:
                        refresh(message=apply_permission_selection(current_controller(), option), busy=False)
                else:
                    open_permission_picker()
                return
            if parsed.canonical == "/status":
                refresh(message=format_product_status(current_controller(), is_busy=state["busy"]).replace("\n", " • "), busy=state["busy"])
                return
            if parsed.canonical == "/capabilities":
                record_capability_inspection(current_controller())
                state["local_panel"] = _capabilities_panel_fragments(current_controller())
                refresh(message="Routing Capability Registry displayed. Advisory only; no backend switch.", busy=state["busy"])
                return
            if parsed.canonical == "/copy":
                _, message = copy_text_to_clipboard(latest_assistant_output(current_controller()))
                refresh(message=message, busy=state["busy"])
                return
            if parsed.canonical == "/resume":
                if state["busy"]:
                    refresh(message="Wait for the current turn to finish", busy=True)
                    return
                if parsed.args:
                    _, message = resume_session(parsed.args)
                else:
                    message = format_resume_session_list(current_controller()).replace("\n", " • ")
                refresh(message=message, busy=False)
                return
        if parsed is not None and parsed.action is SlashCommandAction.LOCAL and parsed.canonical == "/history":
            composer.buffer.text = ""
            state["local_panel"] = None
            refresh(message="Conversation refreshed", busy=state["busy"])
            return
        if parsed is not None and parsed.action is SlashCommandAction.LOCAL and parsed.canonical == "/details":
            composer.buffer.text = ""
            state["show_activity_details"] = not state["show_activity_details"]
            refresh(
                message=f"Activity details {'expanded' if state['show_activity_details'] else 'collapsed'}",
                busy=state["busy"],
            )
            return
        if parsed is not None and parsed.action is SlashCommandAction.LOCAL and parsed.canonical == "/context":
            composer.buffer.text = ""
            state["local_panel"] = _context_panel_fragments(current_controller())
            refresh(message=resume_context_status_message(current_controller()), busy=state["busy"])
            return
        if parsed is not None and parsed.action is SlashCommandAction.LOCAL and parsed.canonical == "/summarize":
            composer.buffer.text = ""
            if state["busy"]:
                refresh(message="Wait for the current turn to finish", busy=True)
                return
            state["local_panel"] = _summarize_panel_fragments(
                phase="running",
                detail="streaming summary draft",
            )
            refresh(message="Generating Gemini summary checkpoint…", busy=True)
            threading.Thread(target=summary_worker, daemon=True).start()
            return
        if parsed is not None and parsed.action is SlashCommandAction.LOCAL and parsed.canonical == "/handoff":
            composer.buffer.text = ""
            source_permission_values = current_permission_values(current_controller())
            ok, preview = build_handoff_preview(
                current_controller().session,
                parsed.args,
                source_permission_values=source_permission_values,
            )
            record_controller_handoff_routing_decision(
                current_controller(),
                parsed.args,
                source_permission_values=source_permission_values,
                user_decision="deferred",
                final_action="previewed" if ok else "blocked",
                reason="" if ok else preview,
            )
            handoff_status = handoff_status_message(
                current_controller().session,
                parsed.args,
                source_permission_values=source_permission_values,
            )
            state["local_panel"] = _handoff_panel_fragments(
                parsed.args,
                ok=ok,
                preview=preview,
                status_line=handoff_status,
            )
            refresh(message=handoff_status, busy=state["busy"])
            return
        if parsed is not None and parsed.action is SlashCommandAction.LOCAL and parsed.canonical == "/task":
            composer.buffer.text = ""
            state["local_panel"] = None
            if state["busy"] and parsed.args.strip().lower() != "status":
                refresh(message="Wait for the current turn to finish", busy=True)
                return
            _, message = handle_task_command(current_controller(), parsed.args)
            refresh(message=message, busy=state["busy"])
            return
        if state["busy"]:
            refresh(message="Wait for the current turn to finish", busy=True)
            return
        composer.buffer.text = ""
        state["local_panel"] = None
        refresh(message=f"Sending to {current_controller().session.backend.value}…", busy=True)
        backend_prompt = parsed.backend_prompt if parsed is not None else text
        threading.Thread(target=worker, args=(backend_prompt,), daemon=True).start()

    def move_picker(offset: int) -> None:
        current_index = BACKEND_CHOICES.index(state["selected_backend"])
        state["selected_backend"] = BACKEND_CHOICES[(current_index + offset) % len(BACKEND_CHOICES)]
        refresh(message=f"Selected {state['selected_backend']}. Press Enter to continue.", busy=False)

    kb = KeyBindings()

    @kb.add("c-c", filter=product_picker_hidden)
    def _(event) -> None:
        event.app.exit(result=0)

    @kb.add("escape", filter=picker_visible)
    def _(event) -> None:
        event.app.exit(result=0)

    @kb.add("escape", filter=session_visible & product_picker_hidden & slash_palette_visible)
    def _(event) -> None:
        state["slash_palette"] = False
        composer.buffer.cancel_completion()
        refresh(message="Slash palette closed", busy=state["busy"])

    @kb.add("escape", filter=session_visible & product_picker_hidden & local_panel_visible & ~slash_palette_visible)
    def _(event) -> None:
        close_local_panel()

    @kb.add("escape", filter=session_visible & product_picker_hidden & ~local_panel_visible & ~slash_palette_visible)
    def _(event) -> None:
        event.app.exit(result=0)

    @kb.add("f2", filter=product_picker_hidden)
    def _(event) -> None:
        refresh(message="Conversation refreshed" if controller is not None else "Backend list refreshed", busy=state["busy"])

    @kb.add("f3", filter=session_visible & product_picker_hidden)
    def _(event) -> None:
        state["show_activity_details"] = not state["show_activity_details"]
        refresh(
            message=f"Activity details {'expanded' if state['show_activity_details'] else 'collapsed'}",
            busy=state["busy"],
        )

    @kb.add("r", filter=interrupted_recovery_visible)
    def _(event) -> None:
        retry_interrupted_turn()

    @kb.add("c", filter=interrupted_recovery_visible)
    def _(event) -> None:
        inspect_interrupted_partial()

    @kb.add("s", filter=interrupted_recovery_visible)
    def _(event) -> None:
        summarize_then_resume_interrupted_turn()

    @kb.add("h", filter=interrupted_recovery_visible)
    def _(event) -> None:
        prefill_handoff_preview_from_interrupted_turn()

    @kb.add("f4", filter=session_visible & product_picker_hidden)
    def _(event) -> None:
        toggle_sidebar_group("lineage")

    @kb.add("f5", filter=session_visible & product_picker_hidden)
    def _(event) -> None:
        toggle_sidebar_group("last activity")

    @kb.add("up", filter=picker_visible)
    def _(event) -> None:
        move_picker(-1)

    @kb.add("down", filter=picker_visible)
    def _(event) -> None:
        move_picker(1)

    @kb.add("1", filter=picker_visible)
    def _(event) -> None:
        select_backend("codex")

    @kb.add("2", filter=picker_visible)
    def _(event) -> None:
        select_backend("claude")

    @kb.add("3", filter=picker_visible)
    def _(event) -> None:
        select_backend("gemini")

    @kb.add("enter", filter=picker_visible)
    def _(event) -> None:
        select_backend(state["selected_backend"])

    @kb.add("up", filter=model_picker_visible)
    def _(event) -> None:
        move_model_picker(-1)

    @kb.add("down", filter=model_picker_visible)
    def _(event) -> None:
        move_model_picker(1)

    @kb.add("enter", filter=model_picker_visible)
    def _(event) -> None:
        apply_selected_model()

    @kb.add("escape", filter=model_picker_visible)
    def _(event) -> None:
        close_model_picker()

    @kb.add("up", filter=permission_picker_visible)
    def _(event) -> None:
        move_permission_picker(-1)

    @kb.add("down", filter=permission_picker_visible)
    def _(event) -> None:
        move_permission_picker(1)

    @kb.add("enter", filter=permission_picker_visible)
    def _(event) -> None:
        apply_selected_permissions()

    @kb.add("escape", filter=permission_picker_visible)
    def _(event) -> None:
        close_permission_picker()

    @kb.add("/", filter=has_focus(composer) & session_visible & product_picker_hidden)
    def _(event) -> None:
        event.current_buffer.insert_text("/")
        state["slash_palette"] = True
        if event.current_buffer.document.current_line_before_cursor == "/":
            event.current_buffer.start_completion(select_first=False)

    @kb.add("up", filter=has_focus(composer) & session_visible & product_picker_hidden & slash_palette_visible)
    def _(event) -> None:
        event.current_buffer.complete_previous()
        event.app.invalidate()

    @kb.add("down", filter=has_focus(composer) & session_visible & product_picker_hidden & slash_palette_visible)
    def _(event) -> None:
        event.current_buffer.complete_next()
        event.app.invalidate()

    @kb.add("enter", filter=has_focus(composer) & session_visible & product_picker_hidden & ~completion_is_selected)
    def _(event) -> None:
        if is_prompt_toolkit_shift_enter_event(event):
            event.current_buffer.insert_text("\n")
        else:
            submit_current_buffer()

    @kb.add("escape", "enter", filter=has_focus(composer) & session_visible & product_picker_hidden)
    def _(event) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("c-j", filter=has_focus(composer) & session_visible & product_picker_hidden)
    def _(event) -> None:
        submit_current_buffer()

    session_content = HSplit(
        [
            chrome_bar,
            header_bar,
            header_meta_bar,
            Window(height=1, char="─", style="class:header.rule"),
            ConditionalContainer(
                _boxed_surface(
                    "SELECT · ACTIVE BACKEND",
                    "vendor-native auth · explicit session start",
                    picker_window,
                    "1 2 3 direct · ↑↓ navigate · ↵ start · esc cancel",
                ),
                filter=picker_visible,
            ),
            ConditionalContainer(
                _boxed_surface(
                    "SELECT · MODEL",
                    "/model <value> skips this picker",
                    model_picker_window,
                    "↑↓ navigate · ↵ apply · esc cancel",
                ),
                filter=model_picker_visible,
            ),
            ConditionalContainer(
                _boxed_surface(
                    "SELECT · PERMISSIONS",
                    "backend mapping visible before applying",
                    permission_picker_window,
                    "↑↓ navigate · ↵ apply · esc cancel · widening permissions is never automatic",
                ),
                filter=permission_picker_visible,
            ),
            ConditionalContainer(
                VSplit(
                    [
                        HSplit([conversation_bar, transcript_window]),
                        Window(width=1, char="│", style="class:rule"),
                        HSplit([sidebar_bar, sidebar_window], width=40),
                    ]
                ),
                filter=session_visible & product_picker_hidden,
            ),
            ConditionalContainer(
                composer_box,
                filter=session_visible & product_picker_hidden,
            ),
            status_bar,
        ]
    )

    root = FloatContainer(
        content=HSplit(
            [
                _rule_row("┌", "─", "┐"),
                VSplit(
                    [
                        Window(width=1, char="│", style="class:border"),
                        session_content,
                        Window(width=1, char="│", style="class:border"),
                    ]
                ),
                _rule_row("└", "─", "┘"),
            ]
        ),
        floats=[
            LayoutFloat(
                left=2,
                right=2,
                bottom=10,
                height=13,
                content=ConditionalContainer(
                    _boxed_surface(
                        "SELECT · SLASH COMMANDS",
                        "local · product · backend · passthrough",
                        slash_palette_window,
                        "↑↓ select · ↵ insert · esc close",
                    ),
                    filter=slash_palette_visible,
                ),
                z_index=2,
            )
        ],
    )
    app = Application(
        layout=Layout(root, focused_element=composer if controller is not None else picker_window),
        key_bindings=kb,
        full_screen=True,
        # Keep terminal mouse reporting disabled so users can drag-select
        # rendered transcript/sidebar text with their terminal and copy it.
        mouse_support=False,
        style=_build_prompt_toolkit_style(),
    )

    stop_spinner = threading.Event()

    def spinner_loop() -> None:
        while not stop_spinner.wait(0.12):
            if state["busy"]:
                refresh()

    threading.Thread(target=spinner_loop, daemon=True).start()
    refresh()
    try:
        return int(app.run() or 0)
    finally:
        stop_spinner.set()
        if controller is not None:
            controller.close()


def run_interface(
    controller_factory: Callable[[str], SessionController],
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    initial_backend: str | None = None,
    use_fullscreen: bool | None = None,
) -> int:
    if use_fullscreen is None:
        use_fullscreen = input_fn is input and print_fn is print and sys.stdin.isatty() and sys.stdout.isatty()
    if use_fullscreen:
        return run_prompt_toolkit_interface(controller_factory=controller_factory, initial_backend=initial_backend)
    return run_simple_interface(
        controller_factory=controller_factory,
        input_fn=input_fn,
        print_fn=print_fn,
        initial_backend=initial_backend,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cwd = Path.cwd()
    store = TranscriptStore(cwd / args.transcript_dir)
    controller_factory = default_controller_factory(args.transcript_dir, cwd)
    if args.list_sessions:
        print(format_session_list(store.list_sessions()))
        return 0
    if args.summarize_session:
        session = store.load_session(args.summarize_session)
        summary_adapter = build_summary_backend(args.summary_backend)
        try:
            summary = generate_and_persist_summary(
                session,
                adapter=summary_adapter,
                cwd=cwd,
                save_session=store.save_session,
                scope=args.summary_scope,
                task_id=args.summary_task_id,
            )
            print(format_summary_record(summary))
            return 0
        finally:
            summary_adapter.close()
    if args.handoff_session:
        if not args.target_backend:
            print("--target-backend is required with --handoff-session", file=sys.stderr)
            return 2
        target_backend = normalize_backend_choice(args.target_backend)
        if target_backend is None:
            print(f"Unsupported target backend: {args.target_backend}", file=sys.stderr)
            return 2
        try:
            session = store.load_session(args.handoff_session)
        except FileNotFoundError:
            print(f"Session not found: {args.handoff_session}", file=sys.stderr)
            return 2
        handoff_scope, handoff_task_id = default_handoff_scope(session)
        if args.handoff_task_id:
            handoff_scope = "task"
            handoff_task_id = args.handoff_task_id
        handoff_turn_ids = split_csv_values(args.handoff_turn_id)
        handoff_statuses = split_csv_values(args.handoff_status)
        source_permission_values = None
        packet = build_handoff_packet(
            session,
            target_backend=BackendName(target_backend),
            target_model=args.target_model,
            user_goal=args.handoff_goal,
            scope=handoff_scope,
            task_id=handoff_task_id,
            turn_ids=handoff_turn_ids or None,
            statuses=handoff_statuses or None,
            recent_turn_limit=args.handoff_recent,
        )
        if args.handoff_execute:
            if not args.handoff_goal.strip():
                print("--handoff-goal is required with --handoff-execute", file=sys.stderr)
                return 2
            controller, turn = execute_handoff_packet(
                adapter=build_backend(target_backend, model=args.target_model),
                store=store,
                cwd=Path(session.workspace_cwd) if session.workspace_cwd else cwd,
                source_session=session,
                packet=packet,
                user_goal=args.handoff_goal,
                confirmation_method="--handoff-execute",
                source_permission_values=source_permission_values,
            )
            try:
                print(format_handoff_execution_confirmation(packet, confirmation_method="--handoff-execute"))
                print(f"Session : {controller.session.id}")
                print(format_turn_summary(turn))
                return 0
            finally:
                controller.close()
        persist_session_routing_decision(
            store,
            session,
            build_packet_routing_decision(
                session,
                packet,
                source_permission_values=source_permission_values,
                target_permission_values=default_permission_values_for_backend(target_backend),
                user_decision="deferred",
                final_action="preview_exported" if args.handoff_output else "previewed",
                reason="manual handoff packet preview generated",
                metadata={
                    "confirmation_method": "--handoff-session",
                    "permission_source": "unavailable_for_loaded_session",
                },
            ),
        )
        rendered = format_handoff_packet(packet, source_permission_values=source_permission_values)
        if args.handoff_output:
            output_path = Path(args.handoff_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n")
            print(f"Handoff packet written: {output_path}")
        else:
            print(rendered)
        return 0
    if args.resume_session:
        try:
            session = store.load_session(args.resume_session)
        except FileNotFoundError:
            print(f"Session not found: {args.resume_session}", file=sys.stderr)
            return 2
        backend = args.backend or session.backend.value
        try:
            if session.turns and backend != session.backend.value:
                raise ValueError(
                    "One backend per session is enforced for local resume; "
                    f"session {session.id} uses {session.backend.value}, requested {backend}"
                )
            resume_context_config = ResumeContextConfig(
                enabled=args.resume_context == "auto",
                recent_turn_limit=args.resume_context_turns,
            )
            controller_factory = resume_controller_factory(
                store,
                session,
                cwd,
                resume_context_config=resume_context_config,
            )
            if args.prompt:
                controller = controller_factory(backend)
                try:
                    print(format_turn_summary(controller.submit_prompt(args.prompt)))
                    return 0
                finally:
                    controller.close()
            return run_interface(
                controller_factory=controller_factory,
                initial_backend=backend,
                use_fullscreen=False if args.simple_ui else None,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args.prompt:
        backend = args.backend or "codex"
        controller = controller_factory(backend)
        try:
            print(format_turn_summary(controller.submit_prompt(args.prompt)))
            return 0
        finally:
            controller.close()
    return run_interface(
        controller_factory=controller_factory,
        initial_backend=args.backend,
        use_fullscreen=False if args.simple_ui else None,
    )
