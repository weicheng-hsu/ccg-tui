from __future__ import annotations

import re
import time
import json
from pathlib import Path
from uuid import uuid4

from ccg_tui.backends.base import BackendAdapter, BackendEvent, BackendSession, EventType, NormalizedError, activity_event, text_delta
from ccg_tui.backends.pty_transport import JsonlTail, PtyProcess
from ccg_tui.models import BackendName

_CLAUDE_PASTED_TEXT_MARKER = "[Pasted text #"


def _one_line_text(text: str) -> str:
    return " ".join(text.split())


def _format_tool_args(args: object) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    parts = []
    for key, value in args.items():
        if isinstance(value, str):
            rendered = value
        else:
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        parts.append(f"{key}={rendered}")
    return _one_line_text(" ".join(parts))


def _claude_project_slug(cwd: Path) -> str:
    return str(Path(cwd).resolve()).replace("/", "-")


def _claude_transcript_path(cwd: Path, session_id: str) -> Path:
    return Path.home() / ".claude" / "projects" / _claude_project_slug(cwd) / f"{session_id}.jsonl"


def _extract_claude_text(record: dict) -> str:
    if record.get("type") != "assistant":
        return ""
    content = record.get("message", {}).get("content", [])
    texts = [item.get("text", "") for item in content if item.get("type") == "text" and item.get("text")]
    return "".join(texts)


def _extract_claude_activity(record: dict) -> str:
    activity = _extract_claude_activity_payload(record)
    return activity.get("backend_label", "") if activity else ""


def _extract_claude_activity_payload(record: dict) -> dict:
    if record.get("type") != "assistant":
        return {}
    content = record.get("message", {}).get("content", [])
    tool_descriptions = []
    tool_calls = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "tool_use":
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            name = "tool"
        detail = _format_tool_args(item.get("input"))
        tool_descriptions.append(f"{name}" + (f" {detail}" if detail else ""))
        tool_calls.append(
            {
                "name": name,
                "input": item.get("input") if isinstance(item.get("input"), dict) else {},
                "id": item.get("id"),
            }
        )
    if not tool_descriptions:
        return {}
    return {
        "kind": "tool_started",
        "title": "Claude tool use",
        "backend_label": "tools: " + "; ".join(tool_descriptions),
        "status": "started",
        "details": {"tool_calls": tool_calls},
    }


def _latest_terminal_title(snapshot: str) -> str:
    matches = re.findall(r"\x1b]0;([^\x07\x1b]*)(?:\x07|\x1b\\)", snapshot[-4096:])
    return matches[-1] if matches else ""


def _strip_terminal_controls(text: str) -> str:
    without_osc = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", text)
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", without_osc)


def _claude_exit_plan_mode_prompt_visible(snapshot: str) -> bool:
    normalized = " ".join(_strip_terminal_controls(snapshot[-4096:]).split()).lower()
    compact = re.sub(r"\s+", "", normalized)
    has_prompt = "exit plan mode?" in normalized or "exitplanmode?" in compact
    has_yes = "1. yes" in normalized or "1.yes" in compact
    has_no = "2. no" in normalized or "2.no" in compact
    return has_prompt and has_yes and has_no


def _claude_ui_ready(snapshot: str) -> bool:
    has_prompt = "plan mode" in snapshot and ("❯" in snapshot or "> " in snapshot)
    return has_prompt or _latest_terminal_title(snapshot).startswith("✳ ")


def _snapshot_has_new_claude_prompt_marker(snapshot: str, baseline: str, prompt: str) -> bool:
    if prompt and snapshot.count(prompt) > baseline.count(prompt):
        return True
    if snapshot.count(_CLAUDE_PASTED_TEXT_MARKER) > baseline.count(_CLAUDE_PASTED_TEXT_MARKER):
        return True
    return False


def _claude_composer_still_has_prompt(snapshot: str, baseline: str, prompt: str) -> bool:
    tail = snapshot[-4096:]
    if "❯" not in tail and "> " not in tail:
        return False
    return _snapshot_has_new_claude_prompt_marker(snapshot, baseline, prompt)


def _permission_flags(permission_mode: str) -> list[str]:
    normalized = permission_mode.strip() or "default"
    if normalized == "dangerously-skip-permissions":
        return ["--dangerously-skip-permissions"]
    return ["--permission-mode", normalized]


def _claude_turn_complete(
    *,
    emitted_text: str,
    saw_end_turn: bool,
    snapshot: str,
    transport_idle: float,
    transcript_idle: float | None,
) -> bool:
    if not emitted_text or not saw_end_turn or transcript_idle is None:
        return False
    return _claude_ui_ready(snapshot) and transport_idle >= 0.3 and transcript_idle >= 0.3


class ClaudePtySession(BackendSession):
    prompt_reflect_timeout = 10.0
    submit_start_timeout = 3.0

    def __init__(self, cwd: Path, *, model: str | None = None, permission_mode: str = "default") -> None:
        self.cwd = Path(cwd)
        self.permission_mode = permission_mode
        self.session_id = str(uuid4())
        self.transcript_path = _claude_transcript_path(self.cwd, self.session_id)
        self.tail = JsonlTail(self.transcript_path)
        self._session_emitted = False
        self._declined_exit_plan_mode = False
        self.transport = PtyProcess(
            self.build_start_command(model=model, permission_mode=permission_mode, session_id=self.session_id),
            self.cwd,
            env={"TERM": "xterm-256color"},
        )
        self._wait_until_ready(timeout=30.0)
        self.tail.seek_to_end()

    @staticmethod
    def build_start_command(
        *,
        model: str | None = None,
        permission_mode: str = "default",
        session_id: str | None = None,
    ) -> list[str]:
        resolved_session_id = session_id or str(uuid4())
        command = ["claude", "--session-id", resolved_session_id, *_permission_flags(permission_mode)]
        if model:
            command.extend(["--model", model])
        return command

    def _wait_until_ready(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.transport.is_running():
                return
            snapshot = self.transport.snapshot()
            if "Quick safety check" in snapshot:
                self.transport.send("\r")
                time.sleep(0.2)
                continue
            has_prompt = "❯" in snapshot or "> " in snapshot
            if "plan mode" in snapshot and has_prompt and self.transport.idle_for() >= 0.5:
                return
            if "Enter to confirm" not in snapshot and self.transport.idle_for() >= 1.0 and snapshot:
                return
            time.sleep(0.05)

    def _wait_until_prompt_reflected(self, prompt: str, *, baseline: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _snapshot_has_new_claude_prompt_marker(self.transport.snapshot(), baseline, prompt):
                return
            if not self.transport.is_running():
                return
            time.sleep(0.05)

    def _wait_until_turn_started(self, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.transcript_path.exists() and self.transcript_path.stat().st_size > 0:
                return True
            if not self.transport.is_running():
                return True
            time.sleep(0.05)
        return self.transcript_path.exists() and self.transcript_path.stat().st_size > 0

    def _submit_prompt(self, prompt: str) -> None:
        self._wait_until_ready(timeout=15.0)
        baseline_snapshot = self.transport.snapshot()
        self.transport.send(prompt)
        self._wait_until_prompt_reflected(
            prompt,
            baseline=baseline_snapshot,
            timeout=self.prompt_reflect_timeout,
        )
        self.transport.send("\r")
        if self._wait_until_turn_started(timeout=self.submit_start_timeout):
            return
        if _claude_composer_still_has_prompt(self.transport.snapshot(), baseline_snapshot, prompt):
            self.transport.send("\r")

    def _decline_exit_plan_mode_if_prompted(self) -> bool:
        if self._declined_exit_plan_mode:
            return False
        if not _claude_exit_plan_mode_prompt_visible(self.transport.snapshot()):
            return False
        self.transport.send("2\r")
        self._declined_exit_plan_mode = True
        return True

    def run(self, prompt: str):
        if not self._session_emitted:
            self._session_emitted = True
            yield BackendEvent(type=EventType.SESSION_STARTED, session_id=self.session_id)
        self._submit_prompt(prompt)
        output_started = False
        emitted_text = ""
        saw_end_turn = False
        last_transcript_change_at: float | None = None
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            for record in self.tail.read_new_records():
                last_transcript_change_at = time.monotonic()
                activity = _extract_claude_activity(record)
                if activity:
                    yield activity_event(raw=record, **_extract_claude_activity_payload(record))
                text = _extract_claude_text(record)
                if text:
                    delta = text_delta(emitted_text, text)
                    if not delta:
                        continue
                    if not output_started:
                        output_started = True
                        yield BackendEvent(type=EventType.OUTPUT_STARTED, raw=record)
                    emitted_text = text
                    yield BackendEvent(type=EventType.OUTPUT_DELTA, text=delta, raw=record)
                if record.get("type") == "assistant" and record.get("message", {}).get("stop_reason") == "end_turn":
                    saw_end_turn = True
                if record.get("type") == "result" and record.get("is_error"):
                    yield BackendEvent(
                        type=EventType.BACKEND_FAILED,
                        error=NormalizedError(kind="backend_error", message=record.get("result", "Claude request failed")),
                        raw=record,
                    )
                    return
            if self._decline_exit_plan_mode_if_prompted():
                yield activity_event(
                    kind="plan_mode_confirmation",
                    title="Claude plan-mode confirmation",
                    backend_label="declined exit plan mode",
                    status="finished",
                )
            transcript_idle = None if last_transcript_change_at is None else time.monotonic() - last_transcript_change_at
            if _claude_turn_complete(
                emitted_text=emitted_text,
                saw_end_turn=saw_end_turn,
                snapshot=self.transport.snapshot(),
                transport_idle=self.transport.idle_for(),
                transcript_idle=transcript_idle,
            ):
                yield BackendEvent(type=EventType.BACKEND_SUCCEEDED)
                return
            exit_code = self.transport.exit_code()
            if exit_code is not None:
                message = self.transport.snapshot().strip() or f"claude exited with code {exit_code}"
                yield BackendEvent(
                    type=EventType.BACKEND_FAILED,
                    error=NormalizedError(kind="process_exit", message=message, exit_code=exit_code),
                )
                return
            time.sleep(0.1)
        yield BackendEvent(
            type=EventType.BACKEND_FAILED,
            error=NormalizedError(kind="timeout_error", message="Timed out waiting for Claude response"),
        )

    def close(self) -> None:
        self.transport.close()

    def start_interactive(self, prompt: str) -> None:
        self._submit_prompt(prompt)

    def send_interactive_input(self, text: str) -> None:
        self.transport.send(text)

    def interactive_snapshot(self) -> str:
        return self.transport.snapshot()

    def run_interactive_terminal(self, prompt: str) -> None:
        self.start_interactive(prompt)
        self.transport.run_foreground_until_ctrl_g()


class ClaudeAdapter(BackendAdapter):
    name = BackendName.CLAUDE

    def __init__(self, *, model: str | None = None, permission_mode: str = "default") -> None:
        super().__init__()
        self.model = model
        self.permission_mode = permission_mode

    def build_start_command(self, cwd: Path, session_id: str | None = None) -> list[str]:
        return ClaudePtySession.build_start_command(
            model=self.model,
            permission_mode=self.permission_mode,
            session_id=session_id,
        )

    def build_command(self, prompt: str, cwd: Path) -> list[str]:
        command = [
            "claude",
            "-p",
            "--verbose",
            "--output-format",
            "stream-json",
            *_permission_flags(self.permission_mode),
        ]
        if self.model:
            command.extend(["--model", self.model])
        command.append(prompt)
        return command

    def parse_stdout_line(self, line: str) -> list[BackendEvent]:
        payload = self.parse_json(line)
        if payload is None:
            return []
        event_type = payload.get("type")
        if event_type == "system" and payload.get("subtype") == "init":
            return [BackendEvent(type=EventType.SESSION_STARTED, session_id=payload.get("session_id"), raw=payload)]
        if event_type == "assistant":
            events = []
            activity = _extract_claude_activity_payload(payload)
            if activity:
                events.append(activity_event(raw=payload, **activity))
            content = payload.get("message", {}).get("content", [])
            texts = [item.get("text", "") for item in content if item.get("type") == "text" and item.get("text")]
            merged = "".join(texts)
            if merged:
                events.extend(
                    [
                        BackendEvent(type=EventType.OUTPUT_STARTED, raw=payload),
                        BackendEvent(type=EventType.OUTPUT_DELTA, text=merged, raw=payload),
                    ]
                )
            return events
        if event_type == "result" and payload.get("subtype") == "success":
            return [BackendEvent(type=EventType.BACKEND_SUCCEEDED, raw=payload)]
        if event_type == "result" and payload.get("is_error"):
            return [
                BackendEvent(
                    type=EventType.BACKEND_FAILED,
                    error=NormalizedError(kind="backend_error", message=payload.get("result", "Claude request failed")),
                    raw=payload,
                )
            ]
        return []

    def open_session(self, cwd: Path) -> BackendSession:
        return ClaudePtySession(cwd, model=self.model, permission_mode=self.permission_mode)
