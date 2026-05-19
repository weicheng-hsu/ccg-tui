from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ccg_tui.backends.base import BackendAdapter, BackendEvent, BackendSession, EventType, NormalizedError, activity_event, text_delta
from ccg_tui.backends.pty_transport import PtyProcess
from ccg_tui.models import BackendName

GEMINI_ENTER_SEQUENCE = "\x1b[13u"


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
    return " ".join(parts)


def _gemini_projects_registry_path() -> Path:
    return Path.home() / ".gemini" / "projects.json"


def _gemini_project_slug(cwd: Path) -> str:
    registry_path = _gemini_projects_registry_path()
    resolved = str(Path(cwd).resolve())
    if registry_path.exists():
        try:
            payload = json.loads(registry_path.read_text())
        except json.JSONDecodeError:
            payload = {}
        projects = payload.get("projects", {}) if isinstance(payload, dict) else {}
        slug = projects.get(resolved)
        if isinstance(slug, str) and slug:
            return slug
    return Path(cwd).resolve().name


def _gemini_chat_dir(cwd: Path) -> Path:
    return Path.home() / ".gemini" / "tmp" / _gemini_project_slug(cwd) / "chats"


def _load_gemini_chat(path: Path) -> dict | None:
    if not path.exists():
        return None
    if path.suffix == ".jsonl":
        conversation: dict = {"messages": []}
        try:
            lines = path.read_text().splitlines()
        except UnicodeDecodeError:
            return None
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if "$set" in record and isinstance(record["$set"], dict):
                conversation.update(record["$set"])
                continue
            if record.get("kind") == "main" and "sessionId" in record:
                conversation.update(record)
                continue
            if record.get("type") in {"user", "gemini"}:
                conversation["messages"].append(record)
        return conversation
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _iter_gemini_chat_files(chat_dir: Path):
    yield from chat_dir.glob("session-*.json")
    yield from chat_dir.glob("session-*.jsonl")


def _gemini_user_text(message: dict) -> str:
    if message.get("type") != "user":
        return ""
    content = message.get("content", [])
    if not isinstance(content, list):
        return ""
    parts = [item.get("text", "") for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)]
    return "".join(parts)


def _gemini_turn_text(messages: list[dict], previous_count: int) -> str:
    parts = [
        message.get("content", "")
        for message in messages[previous_count:]
        if isinstance(message, dict)
        and message.get("type") == "gemini"
        and isinstance(message.get("content"), str)
        and message.get("content")
    ]
    return "\n\n".join(parts)


def _gemini_turn_has_tool_calls(messages: list[dict], previous_count: int) -> bool:
    return any(
        isinstance(message, dict)
        and message.get("type") == "gemini"
        and bool(message.get("toolCalls"))
        for message in messages[previous_count:]
    )


def _gemini_tool_activity(message: dict) -> str:
    activity = _gemini_tool_activity_payload(message)
    return activity.get("backend_label", "") if activity else ""


def _gemini_tool_activity_payload(message: dict) -> dict:
    tool_calls = message.get("toolCalls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return {}
    descriptions: list[str] = []
    normalized_calls: list[dict] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("displayName") or call.get("name") or call.get("functionName") or call.get("tool")
        if not isinstance(name, str) or not name:
            name = "tool"
        detail = call.get("description")
        if not isinstance(detail, str) or not detail:
            detail = _format_tool_args(call.get("args"))
        rendered = _one_line_text(f"{name} {detail}".strip())
        if rendered:
            descriptions.append(rendered)
        normalized_calls.append(
            {
                "name": call.get("name") or name,
                "display_name": call.get("displayName"),
                "description": call.get("description"),
                "args": call.get("args") if isinstance(call.get("args"), dict) else {},
                "status": call.get("status"),
                "result": call.get("result"),
            }
        )
    if not descriptions:
        descriptions.append(f"{len(tool_calls)} call(s)")
    return {
        "kind": "tool",
        "title": "Gemini tool call",
        "backend_label": "tools: " + "; ".join(descriptions),
        "status": "finished" if all(call.get("status") == "success" for call in normalized_calls) else "started",
        "details": {"tool_calls": normalized_calls},
    }


def _gemini_tool_name_from_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    parts = value.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0]
    return value


def _gemini_direct_tool_activity_payload(payload: dict) -> dict:
    event_type = payload.get("type")
    name = (
        payload.get("name")
        or payload.get("tool_name")
        or payload.get("functionName")
        or payload.get("tool")
        or _gemini_tool_name_from_id(payload.get("tool_id") or payload.get("id") or payload.get("tool_call_id"))
    )
    if not isinstance(name, str) or not name:
        name = "tool"
    args = payload.get("args") or payload.get("input") or payload.get("parameters")
    detail = _format_tool_args(args)
    if event_type == "tool_result":
        return {
            "kind": "tool_output",
            "title": "Gemini tool result",
            "backend_label": f"tool result: {_one_line_text(name)}",
            "status": "finished",
            "details": {
                "name": name,
                "id": payload.get("id") or payload.get("tool_id") or payload.get("tool_call_id"),
                "result": payload.get("result") or payload.get("output") or payload.get("content"),
            },
        }
    return {
        "kind": "tool_started",
        "title": "Gemini tool use",
        "backend_label": f"tools: {_one_line_text(name)}" + (f" {detail}" if detail else ""),
        "status": "started",
        "details": {
            "name": name,
            "id": payload.get("id") or payload.get("tool_id") or payload.get("tool_call_id"),
            "args": args if isinstance(args, dict) else {},
        },
    }


def _gemini_latest_turn_message(messages: list[dict], previous_count: int) -> dict | None:
    for message in reversed(messages[previous_count:]):
        if isinstance(message, dict) and message.get("type") == "gemini":
            return message
    return None


def _latest_terminal_title(snapshot: str) -> str:
    matches = re.findall(r"\x1b]0;([^\x07\x1b]*)(?:\x07|\x1b\\)", snapshot[-4096:])
    return matches[-1] if matches else ""


def _gemini_ui_ready(snapshot: str) -> bool:
    return _latest_terminal_title(snapshot).startswith("◇  Ready")


class GeminiPtySession(BackendSession):
    def __init__(self, cwd: Path, *, model: str | None = None, approval_mode: str = "default") -> None:
        self.cwd = Path(cwd)
        self.chat_dir = _gemini_chat_dir(self.cwd)
        self.chat_dir.mkdir(parents=True, exist_ok=True)
        self._known_chat_files = {str(path) for path in _iter_gemini_chat_files(self.chat_dir)}
        self._launch_started_at = time.time()
        self._chat_file: Path | None = None
        self._session_id: str | None = None
        self._session_emitted = False
        self.transport = PtyProcess(
            self.build_start_command(model=model, approval_mode=approval_mode),
            self.cwd,
            env={"TERM": "dumb"},
        )
        self._wait_until_ready(timeout=30.0)

    @staticmethod
    def build_start_command(*, model: str | None = None, approval_mode: str = "default") -> list[str]:
        command = ["gemini", "--approval-mode", approval_mode]
        if model:
            command.extend(["--model", model])
        return command

    def _snapshot_chat_counts(self) -> dict[Path, int]:
        counts: dict[Path, int] = {}
        for path in _iter_gemini_chat_files(self.chat_dir):
            conversation = _load_gemini_chat(path)
            if conversation is None:
                continue
            messages = conversation.get("messages", [])
            if isinstance(messages, list):
                counts[path] = len(messages)
        return counts

    def _wait_until_ready(self, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.transport.is_running():
                return
            snapshot = self.transport.snapshot()
            has_prompt = "Type your message" in snapshot and "? for shortcuts" in snapshot
            if (has_prompt or _gemini_ui_ready(snapshot)) and self.transport.idle_for() >= 0.3:
                return
            time.sleep(0.05)

    def _wait_until_prompt_echoed(self, prompt: str, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if prompt in self.transport.snapshot():
                return
            if not self.transport.is_running():
                return
            time.sleep(0.05)

    def _submit_prompt(self, prompt: str) -> None:
        self._wait_until_ready(timeout=15.0)
        self.transport.send(prompt)
        self._wait_until_prompt_echoed(prompt, timeout=10.0)
        self.transport.send(GEMINI_ENTER_SEQUENCE)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            snapshot = self.transport.snapshot()
            if "Thinking..." in snapshot:
                return
            if prompt not in snapshot and ("Type your message" in snapshot or _gemini_ui_ready(snapshot)):
                return
            time.sleep(0.05)
        if prompt in self.transport.snapshot():
            self.transport.send(GEMINI_ENTER_SEQUENCE)

    def _candidate_chat_files(self) -> list[Path]:
        if self._chat_file is not None:
            return [self._chat_file]
        paths = sorted(
            _iter_gemini_chat_files(self.chat_dir),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        fresh_paths = [path for path in paths if str(path) not in self._known_chat_files]
        if fresh_paths:
            return fresh_paths
        return [path for path in paths if path.stat().st_mtime >= self._launch_started_at - 1.0]

    def _bind_chat_file(self, path: Path, session_id: str | None) -> BackendEvent | None:
        self._chat_file = path
        if not isinstance(session_id, str) or not session_id or self._session_emitted:
            return None
        self._session_id = session_id
        self._session_emitted = True
        return BackendEvent(type=EventType.SESSION_STARTED, session_id=session_id)

    def _conversation_matches_prompt(self, messages: list, prompt: str, previous_count: int) -> bool:
        recent_messages = messages[previous_count:] if previous_count < len(messages) else messages
        for message in reversed(recent_messages):
            if not isinstance(message, dict):
                continue
            if _gemini_user_text(message) == prompt:
                return True
        return False

    def run(self, prompt: str):
        baseline_counts = self._snapshot_chat_counts()
        self._submit_prompt(prompt)
        emitted_text = ""
        output_started = False
        last_update_marker = ""
        last_change_at: float | None = None
        turn_has_tool_calls = False
        latest_message_has_tool_calls = False
        emitted_activity_markers: set[tuple[Path, int, str]] = set()
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            candidate_paths = self._candidate_chat_files()
            for path in candidate_paths:
                conversation = _load_gemini_chat(path)
                if conversation is None:
                    continue
                messages = conversation.get("messages", [])
                if not isinstance(messages, list):
                    continue
                previous_count = baseline_counts.get(path, 0)
                session_event: BackendEvent | None = None
                if self._chat_file is None and self._conversation_matches_prompt(messages, prompt, previous_count):
                    session_event = self._bind_chat_file(path, conversation.get("sessionId"))
                if session_event is not None:
                    yield session_event
                update_marker = conversation.get("lastUpdated")
                if isinstance(update_marker, str) and update_marker and update_marker != last_update_marker:
                    last_update_marker = update_marker
                    last_change_at = time.monotonic()
                if len(messages) > previous_count:
                    turn_has_tool_calls = _gemini_turn_has_tool_calls(messages, previous_count)
                    for message_index, message in enumerate(messages[previous_count:], start=previous_count):
                        if not isinstance(message, dict) or message.get("type") != "gemini":
                            continue
                        activity = _gemini_tool_activity(message)
                        if not activity:
                            continue
                        marker = (path, message_index, activity)
                        if marker in emitted_activity_markers:
                            continue
                        emitted_activity_markers.add(marker)
                        payload = _gemini_tool_activity_payload(message)
                        yield activity_event(raw=message, **payload)
                    latest_message = _gemini_latest_turn_message(messages, previous_count)
                    latest_message_has_tool_calls = bool(latest_message and latest_message.get("toolCalls"))
                    assistant_text = _gemini_turn_text(messages, previous_count)
                    delta = text_delta(emitted_text, assistant_text)
                    if delta:
                        if self._chat_file is None:
                            session_event = self._bind_chat_file(path, conversation.get("sessionId"))
                            if session_event is not None:
                                yield session_event
                        if not output_started:
                            output_started = True
                            yield BackendEvent(type=EventType.OUTPUT_STARTED, raw=conversation)
                        emitted_text = assistant_text
                        yield BackendEvent(type=EventType.OUTPUT_DELTA, text=delta, raw=conversation)
            if (
                emitted_text
                and last_change_at is not None
                and self.transport.idle_for() >= 0.3
                and _gemini_ui_ready(self.transport.snapshot())
                and not latest_message_has_tool_calls
                and (not turn_has_tool_calls or time.monotonic() - last_change_at >= 0.5)
            ):
                yield BackendEvent(type=EventType.BACKEND_SUCCEEDED)
                return
            exit_code = self.transport.exit_code()
            if exit_code is not None:
                message = self.transport.snapshot().strip() or f"gemini exited with code {exit_code}"
                yield BackendEvent(
                    type=EventType.BACKEND_FAILED,
                    error=NormalizedError(kind="process_exit", message=message, exit_code=exit_code),
                )
                return
            time.sleep(0.1)
        yield BackendEvent(
            type=EventType.BACKEND_FAILED,
            error=NormalizedError(kind="timeout_error", message="Timed out waiting for Gemini response"),
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


class GeminiAdapter(BackendAdapter):
    name = BackendName.GEMINI

    def __init__(self, *, model: str | None = None, approval_mode: str = "default") -> None:
        super().__init__()
        self.model = model
        self.approval_mode = approval_mode

    def build_start_command(self, cwd: Path) -> list[str]:
        return GeminiPtySession.build_start_command(model=self.model, approval_mode=self.approval_mode)

    def build_command(self, prompt: str, cwd: Path) -> list[str]:
        command = ["gemini", "-p", prompt, "-o", "stream-json", "--approval-mode", self.approval_mode]
        if self.model:
            command.extend(["--model", self.model])
        return command

    def parse_stdout_line(self, line: str) -> list[BackendEvent]:
        payload = self.parse_json(line)
        if payload is None:
            if "auth method" in line.lower() or "api_key" in line.lower():
                return [
                    BackendEvent(
                        type=EventType.BACKEND_FAILED,
                        error=NormalizedError(kind="auth_error", message=line.strip()),
                    )
                ]
            return []
        event_type = payload.get("type")
        if event_type == "init":
            session_id = payload.get("session_id")
            if session_id:
                return [BackendEvent(type=EventType.SESSION_STARTED, session_id=session_id, raw=payload)]
            return []
        if event_type == "message" and payload.get("role") == "assistant":
            events = []
            activity = _gemini_tool_activity_payload(payload)
            if activity:
                events.append(activity_event(raw=payload, **activity))
            text = payload.get("content")
            if isinstance(text, str) and text:
                events.extend(
                    [
                        BackendEvent(type=EventType.OUTPUT_STARTED, raw=payload),
                        BackendEvent(type=EventType.OUTPUT_DELTA, text=text, raw=payload),
                    ]
                )
            if events:
                return events
        if event_type in {"tool_use", "tool_result"}:
            return [activity_event(raw=payload, **_gemini_direct_tool_activity_payload(payload))]
        if event_type == "result" and payload.get("status") == "success":
            return [BackendEvent(type=EventType.BACKEND_SUCCEEDED, raw=payload)]
        if payload.get("is_error") or event_type == "error" or payload.get("status") == "error":
            return [
                BackendEvent(
                    type=EventType.BACKEND_FAILED,
                    error=NormalizedError(kind="backend_error", message=payload.get("message", "Gemini request failed")),
                    raw=payload,
                )
            ]
        return []

    def open_session(self, cwd: Path) -> BackendSession:
        return GeminiPtySession(cwd, model=self.model, approval_mode=self.approval_mode)
