from __future__ import annotations

import json
import time
from pathlib import Path

from ccg_tui.backends.base import BackendAdapter, BackendEvent, BackendSession, EventType, NormalizedError, activity_event, text_delta
from ccg_tui.backends.pty_transport import JsonlTail, PtyProcess
from ccg_tui.models import BackendName

_CODEX_PASTED_CONTENT_MARKER = "[Pasted Content"


def _one_line_text(text: str) -> str:
    return " ".join(text.split())


def _format_tool_args(arguments: object) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return _one_line_text(arguments)
        if not isinstance(parsed, dict):
            return _one_line_text(arguments)
        arguments = parsed
    if not isinstance(arguments, dict) or not arguments:
        return ""
    parts = []
    for key, value in arguments.items():
        if isinstance(value, str):
            rendered = value
        else:
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        parts.append(f"{key}={rendered}")
    return _one_line_text(" ".join(parts))


def _normalize_tool_args(arguments: object) -> dict:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"arguments": arguments}
        return parsed if isinstance(parsed, dict) else {"arguments": arguments}
    return arguments if isinstance(arguments, dict) else {}


def _codex_sessions_root() -> Path:
    return Path.home() / ".codex" / "sessions"


def _list_codex_session_files() -> list[Path]:
    root = _codex_sessions_root()
    if not root.exists():
        return []
    return list(root.rglob("rollout-*.jsonl"))


def _read_codex_session_meta(path: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    with path.open("rb") as handle:
        for _ in range(8):
            line = handle.readline()
            if not line:
                break
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if payload.get("type") != "session_meta":
                continue
            metadata = payload.get("payload", {})
            return metadata.get("id"), metadata.get("cwd")
    return None, None


def _extract_codex_message(record: dict) -> tuple[str, str | None]:
    if record.get("type") != "response_item":
        return "", None
    payload = record.get("payload", {})
    if payload.get("type") != "message" or payload.get("role") != "assistant":
        return "", None
    texts = [
        item.get("text", "")
        for item in payload.get("content", [])
        if item.get("type") == "output_text" and item.get("text")
    ]
    return "".join(texts), payload.get("phase")


def _extract_codex_activity(record: dict) -> str:
    activity = _extract_codex_activity_payload(record)
    return activity.get("backend_label", "") if activity else ""


def _extract_codex_activity_payload(record: dict) -> dict:
    if record.get("type") == "event_msg":
        payload = record.get("payload", {})
        payload_type = payload.get("type")
        if payload_type == "agent_message" and payload.get("phase") != "final_answer":
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                text = _one_line_text(message)
                return {
                    "kind": "progress",
                    "title": "Codex progress",
                    "backend_label": text,
                    "status": None,
                    "details": {"message": message, "phase": payload.get("phase")},
                }
        if payload_type == "agent_reasoning":
            text = payload.get("text")
            if isinstance(text, str) and text.strip():
                return {
                    "kind": "progress",
                    "title": "Codex reasoning",
                    "backend_label": f"reasoning: {_one_line_text(text)}",
                    "status": None,
                    "details": {"text": text},
                }
        if payload_type in {"task_started", "task_complete"}:
            return {
                "kind": "progress",
                "title": payload_type.replace("_", " "),
                "backend_label": payload_type.replace("_", " "),
                "status": "started" if payload_type == "task_started" else "finished",
                "details": dict(payload),
            }
        return {}
    if record.get("type") != "response_item":
        return {}
    payload = record.get("payload", {})
    item_type = payload.get("type")
    if item_type == "function_call":
        name = payload.get("name") or "tool"
        summary = _format_tool_args(payload.get("arguments"))
        return {
            "kind": "tool_started",
            "title": str(name),
            "backend_label": f"tool: {name}" + (f" {summary}" if summary else ""),
            "status": "started",
            "details": {
                "name": name,
                "arguments": _normalize_tool_args(payload.get("arguments")),
                "call_id": payload.get("call_id"),
            },
        }
    if item_type == "function_call_output":
        return {
            "kind": "tool_output",
            "title": "tool output",
            "backend_label": f"tool output: {payload.get('call_id', 'completed')}",
            "status": "finished",
            "details": {"call_id": payload.get("call_id"), "output": payload.get("output")},
        }
    if item_type == "reasoning":
        summaries = payload.get("summary", [])
        texts = [
            item.get("text", "")
            for item in summaries
            if isinstance(item, dict) and item.get("type") == "summary_text" and item.get("text")
        ]
        if texts:
            text = " ".join(texts)
            return {
                "kind": "progress",
                "title": "Codex reasoning",
                "backend_label": f"reasoning: {_one_line_text(text)}",
                "status": None,
                "details": {"summary": summaries},
            }
    return {}


def _extract_codex_task_complete_text(record: dict) -> str:
    if record.get("type") != "event_msg":
        return ""
    payload = record.get("payload", {})
    if payload.get("type") != "task_complete":
        return ""
    message = payload.get("last_agent_message")
    return message if isinstance(message, str) else ""


def _is_codex_task_complete(record: dict) -> bool:
    return record.get("type") == "event_msg" and record.get("payload", {}).get("type") == "task_complete"


def _extract_codex_error(record: dict) -> NormalizedError | None:
    if record.get("type") != "event_msg":
        return None
    payload = record.get("payload", {})
    if payload.get("type") != "error":
        return None
    message = payload.get("message")
    if not isinstance(message, str) or not message:
        return None
    error_kind = payload.get("codex_error_info")
    details = {}
    if isinstance(error_kind, str) and error_kind:
        details["codex_error_info"] = error_kind
    return NormalizedError(
        kind=error_kind if isinstance(error_kind, str) and error_kind else "backend_error",
        message=message,
        details=details,
    )


def _codex_ui_ready(snapshot: str) -> bool:
    return "›" in snapshot and (
        "Use /skills to list available skills" in snapshot
        or "Find and fix a bug in @filename" in snapshot
        or "New /fast" in snapshot
    )


def _codex_turn_complete(
    *,
    emitted_text: str,
    saw_task_complete: bool,
    snapshot: str,
    transport_idle: float,
    rollout_idle: float | None,
) -> bool:
    if not emitted_text:
        return False
    if saw_task_complete:
        return True
    if rollout_idle is None:
        return False
    return _codex_ui_ready(snapshot) and transport_idle >= 0.3 and rollout_idle >= 0.5


def _snapshot_has_new_prompt_marker(snapshot: str, baseline: str, prompt: str) -> bool:
    if prompt and snapshot.count(prompt) > baseline.count(prompt):
        return True
    if snapshot.count(_CODEX_PASTED_CONTENT_MARKER) > baseline.count(_CODEX_PASTED_CONTENT_MARKER):
        return True
    return False


def _codex_turn_start_visible(snapshot: str) -> bool:
    return "Working" in snapshot or "Approaching rate limits" in snapshot


def _codex_composer_still_has_prompt(snapshot: str, baseline: str, prompt: str) -> bool:
    tail = snapshot[-4096:]
    if "›" not in tail:
        return False
    return _snapshot_has_new_prompt_marker(snapshot, baseline, prompt)


class CodexPtySession(BackendSession):
    prompt_reflect_timeout = 10.0
    submit_start_timeout = 3.0

    def __init__(
        self,
        cwd: Path,
        *,
        model: str | None = None,
        approval_policy: str = "on-request",
        sandbox_mode: str = "workspace-write",
    ) -> None:
        self.cwd = Path(cwd)
        self._known_files = {str(path) for path in _list_codex_session_files()}
        self.session_path: Path | None = None
        self.session_id: str | None = None
        self.tail: JsonlTail | None = None
        self._session_emitted = False
        self._launch_started_at = time.time()
        self.transport = PtyProcess(
            self.build_start_command(
                model=model,
                approval_policy=approval_policy,
                sandbox_mode=sandbox_mode,
            ),
            self.cwd,
            env={"TERM": "xterm-256color"},
        )
        self._prepare_startup()

    @staticmethod
    def build_start_command(
        *,
        model: str | None = None,
        approval_policy: str = "on-request",
        sandbox_mode: str = "workspace-write",
    ) -> list[str]:
        command = ["codex", "--no-alt-screen", "-a", approval_policy, "-s", sandbox_mode]
        if model:
            command.extend(["--model", model])
        return command

    def _prepare_startup(self) -> None:
        deadline = time.monotonic() + 30
        confirmed = False
        while time.monotonic() < deadline:
            if not self.transport.is_running():
                return
            snapshot = self.transport.snapshot()
            if not confirmed and "Continue anyway? [y/N]:" in snapshot:
                self.transport.send("y\r")
                confirmed = True
                time.sleep(0.2)
                continue
            has_prompt = (
                "›" in snapshot
                and (
                    "Use /skills to list available skills" in snapshot
                    or "Find and fix a bug in @filename" in snapshot
                    or "New /fast" in snapshot
                )
            )
            if has_prompt and self.transport.idle_for() >= 0.3:
                return
            if confirmed and "Refusing to start" in snapshot:
                continue
            time.sleep(0.05)

    def _wait_until_prompt_reflected(self, prompt: str, *, baseline: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _snapshot_has_new_prompt_marker(self.transport.snapshot(), baseline, prompt):
                return
            if not self.transport.is_running():
                return
            time.sleep(0.05)

    def _wait_until_turn_started(self, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _codex_turn_start_visible(self.transport.snapshot()):
                return True
            if self.tail is None:
                self._locate_session_file(timeout=0.1)
            if self.session_path is not None:
                return True
            if not self.transport.is_running():
                return True
            time.sleep(0.05)
        return _codex_turn_start_visible(self.transport.snapshot()) or self.session_path is not None

    def _submit_prompt(self, prompt: str) -> None:
        self._prepare_startup()
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
        if _codex_composer_still_has_prompt(self.transport.snapshot(), baseline_snapshot, prompt):
            self.transport.send("\r")

    def _locate_session_file(self, *, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            candidates = sorted(
                _list_codex_session_files(),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            for path in candidates:
                if str(path) in self._known_files:
                    continue
                session_id, session_cwd = _read_codex_session_meta(path)
                if session_id and session_cwd == str(self.cwd):
                    self.session_path = path
                    self.session_id = session_id
                    self.tail = JsonlTail(path)
                    return
            if not self.transport.is_running():
                return
            time.sleep(0.1)

    def run(self, prompt: str):
        self._submit_prompt(prompt)
        if self.tail is None:
            self._locate_session_file(timeout=10.0)
        if self.session_id and not self._session_emitted:
            self._session_emitted = True
            yield BackendEvent(type=EventType.SESSION_STARTED, session_id=self.session_id)
        output_started = False
        emitted_text = ""
        saw_task_complete = False
        last_rollout_change_at: float | None = None
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if self.tail is None:
                self._locate_session_file(timeout=0.5)
                if self.session_id and not self._session_emitted:
                    self._session_emitted = True
                    yield BackendEvent(type=EventType.SESSION_STARTED, session_id=self.session_id)
            if self.tail is not None:
                for record in self.tail.read_new_records():
                    last_rollout_change_at = time.monotonic()
                    activity = _extract_codex_activity(record)
                    if activity:
                        yield activity_event(raw=record, **_extract_codex_activity_payload(record))
                    error = _extract_codex_error(record)
                    if error is not None:
                        yield BackendEvent(type=EventType.BACKEND_FAILED, error=error, raw=record)
                        return
                    if _is_codex_task_complete(record):
                        saw_task_complete = True
                    task_complete_text = _extract_codex_task_complete_text(record)
                    if task_complete_text:
                        delta = text_delta(emitted_text, task_complete_text)
                        if delta:
                            if not output_started:
                                output_started = True
                                yield BackendEvent(type=EventType.OUTPUT_STARTED, raw=record)
                            emitted_text = task_complete_text
                            yield BackendEvent(type=EventType.OUTPUT_DELTA, text=delta, raw=record)
                        if emitted_text:
                            yield BackendEvent(type=EventType.BACKEND_SUCCEEDED, raw=record)
                            return
                    text, phase = _extract_codex_message(record)
                    if not text or phase == "commentary":
                        continue
                    delta = text_delta(emitted_text, text)
                    if not delta:
                        continue
                    if phase == "final_answer":
                        if not output_started:
                            output_started = True
                            yield BackendEvent(type=EventType.OUTPUT_STARTED, raw=record)
                        emitted_text = text
                        yield BackendEvent(type=EventType.OUTPUT_DELTA, text=delta, raw=record)
                        yield BackendEvent(type=EventType.BACKEND_SUCCEEDED, raw=record)
                        return
                    if phase is None:
                        if not output_started:
                            output_started = True
                            yield BackendEvent(type=EventType.OUTPUT_STARTED, raw=record)
                        emitted_text = text
                        yield BackendEvent(type=EventType.OUTPUT_DELTA, text=delta, raw=record)
            rollout_idle = None if last_rollout_change_at is None else time.monotonic() - last_rollout_change_at
            if _codex_turn_complete(
                emitted_text=emitted_text,
                saw_task_complete=saw_task_complete,
                snapshot=self.transport.snapshot(),
                transport_idle=self.transport.idle_for(),
                rollout_idle=rollout_idle,
            ):
                yield BackendEvent(type=EventType.BACKEND_SUCCEEDED)
                return
            exit_code = self.transport.exit_code()
            if exit_code is not None:
                message = self.transport.snapshot().strip() or f"codex exited with code {exit_code}"
                yield BackendEvent(
                    type=EventType.BACKEND_FAILED,
                    error=NormalizedError(kind="process_exit", message=message, exit_code=exit_code),
                )
                return
            time.sleep(0.1)
        yield BackendEvent(
            type=EventType.BACKEND_FAILED,
            error=NormalizedError(kind="timeout_error", message="Timed out waiting for Codex response"),
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


class CodexAdapter(BackendAdapter):
    name = BackendName.CODEX

    def __init__(
        self,
        *,
        model: str | None = None,
        approval_policy: str = "on-request",
        sandbox_mode: str = "workspace-write",
    ) -> None:
        super().__init__()
        self.model = model
        self.approval_policy = approval_policy
        self.sandbox_mode = sandbox_mode

    def build_start_command(self, cwd: Path) -> list[str]:
        return CodexPtySession.build_start_command(
            model=self.model,
            approval_policy=self.approval_policy,
            sandbox_mode=self.sandbox_mode,
        )

    def build_command(self, prompt: str, cwd: Path) -> list[str]:
        command = ["codex", "exec", "--skip-git-repo-check", "--json"]
        if self.approval_policy == "never" and self.sandbox_mode == "danger-full-access":
            command.append("--dangerously-bypass-approvals-and-sandbox")
        if self.model:
            command.extend(["--model", self.model])
        command.append(prompt)
        return command

    def parse_stdout_line(self, line: str) -> list[BackendEvent]:
        payload = self.parse_json(line)
        if payload is None:
            return []
        event_type = payload.get("type")
        if event_type == "thread.started":
            return [BackendEvent(type=EventType.SESSION_STARTED, session_id=payload.get("thread_id"), raw=payload)]
        if event_type in {"item.started", "item.completed"}:
            item = payload.get("item", {})
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type and item_type != "agent_message":
                    label = str(item.get("name") or item.get("command") or item_type)
                    detail = _format_tool_args(item.get("arguments") or item.get("args") or item.get("params"))
                    return [
                        activity_event(
                            kind="tool",
                            title=_one_line_text(label),
                            backend_label=f"tool: {_one_line_text(label)}" + (f" {detail}" if detail else ""),
                            status="finished" if event_type == "item.completed" else "started",
                            details={
                                "item_type": item_type,
                                "arguments": _normalize_tool_args(item.get("arguments") or item.get("args") or item.get("params")),
                                "item": item,
                            },
                            raw=payload,
                        )
                    ]
        if event_type == "item.completed":
            item = payload.get("item", {})
            if item.get("type") == "agent_message":
                text = item.get("text", "")
                if text:
                    return [
                        BackendEvent(type=EventType.OUTPUT_STARTED, raw=payload),
                        BackendEvent(type=EventType.OUTPUT_DELTA, text=text, raw=payload),
                    ]
        if event_type == "turn.completed":
            return [BackendEvent(type=EventType.BACKEND_SUCCEEDED, raw=payload)]
        if event_type == "turn.failed":
            return [
                BackendEvent(
                    type=EventType.BACKEND_FAILED,
                    error=NormalizedError(kind="backend_error", message=payload.get("message", "Codex turn failed")),
                    raw=payload,
                )
            ]
        return []

    def open_session(self, cwd: Path) -> BackendSession:
        return CodexPtySession(
            cwd,
            model=self.model,
            approval_policy=self.approval_policy,
            sandbox_mode=self.sandbox_mode,
        )
