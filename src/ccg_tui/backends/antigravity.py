from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Iterable
from pathlib import Path

from ccg_tui.backends.base import BackendAdapter, BackendEvent, EventType, NormalizedError, activity_event
from ccg_tui.backends.pty_transport import PtyProcess
from ccg_tui.models import BackendName


ANTIGRAVITY_EXECUTABLE = "agy"
ANTIGRAVITY_DEFAULT_PRINT_TIMEOUT = "5m0s"
ANTIGRAVITY_MODEL_OPTIONS_ENV = "CCG_TUI_ANTIGRAVITY_MODEL_OPTIONS"
ANTIGRAVITY_PERMISSION_MODES = frozenset(
    {
        "default",
        "sandbox",
        "proceed-in-sandbox",
        "dangerously-skip-permissions",
    }
)
_ANTIGRAVITY_MODEL_OPTIONS_CACHE: tuple[str, ...] | None = None
_TERMINAL_CONTROL_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07]*(?:\x07|\x1b\\)"
    r"|\x1b[PX^_].*?\x1b\\"
    r"|\x1b[@-Z\\-_]",
    re.DOTALL,
)
_MODEL_ROW_PREFIX_RE = re.compile(r"^[\s>●○*•·\-|│╭╰╮╯─━┄┆┊▀▄█▌▐]+")
_MODEL_ROW_SUFFIX_RE = re.compile(r"\s*\(current\)\s*$", re.IGNORECASE)
_MODEL_UI_TEXT_PREFIXES = (
    "/",
    "?",
    "enter ",
    "esc ",
    "keyboard",
    "model picker",
    "select ",
    "switch model",
    "up/down",
    "welcome",
)


def _one_line_text(text: str) -> str:
    return " ".join(text.split())


def _unique_text(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = _one_line_text(value)
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return tuple(unique)


def _strip_terminal_controls(text: str) -> str:
    return _TERMINAL_CONTROL_RE.sub("", text)


def _antigravity_model_options_from_env() -> tuple[str, ...] | None:
    raw = os.environ.get(ANTIGRAVITY_MODEL_OPTIONS_ENV)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, list):
        return _unique_text(str(item) for item in payload)
    return _unique_text(raw.replace(",", "\n").splitlines())


def parse_antigravity_model_options(text: str) -> tuple[str, ...]:
    candidates: list[str] = []
    for raw_line in _strip_terminal_controls(text).replace("\r", "\n").splitlines():
        row = raw_line.rstrip()
        left = row.lstrip()
        if not left:
            continue
        if not left.startswith((">", "●", "○")) and not row.startswith("  "):
            continue
        line = _MODEL_ROW_PREFIX_RE.sub("", raw_line)
        line = _MODEL_ROW_SUFFIX_RE.sub("", line).strip()
        line = _one_line_text(line)
        if not line:
            continue
        lowered = line.casefold()
        if lowered.startswith(_MODEL_UI_TEXT_PREFIXES):
            continue
        if " for shortcuts" in lowered or "handing terminal" in lowered:
            continue
        if "navigate" in lowered and ("select" in lowered or "complete" in lowered):
            continue
        if lowered.endswith("more") and ("↓" in line or "↑" in line):
            continue
        if len(line) > 100:
            continue
        if not any(char.isalnum() for char in line):
            continue
        candidates.append(line)
    return _unique_text(candidates)


def discover_antigravity_model_options(
    *,
    cwd: Path | None = None,
    executable: str = ANTIGRAVITY_EXECUTABLE,
    timeout: float = 8.0,
    max_steps: int = 80,
    step_delay: float = 0.03,
) -> tuple[str, ...]:
    if shutil.which(executable) is None:
        return ()
    transport = PtyProcess(
        [executable],
        cwd or Path.cwd(),
        env={"TERM": "xterm-256color", "AGY_CLI_HIDE_ACCOUNT_INFO": "1"},
        max_buffer_chars=500_000,
    )
    try:
        _wait_until_transport_ready(transport, timeout=min(timeout, 4.0))
        if not transport.is_running():
            return ()
        baseline = len(transport.snapshot())
        transport.send("/model\r")
        _wait_until_model_picker_has_options(transport, timeout=min(timeout, 4.0))
        snapshot = transport.snapshot()
        snapshots = [snapshot[baseline:] if len(snapshot) >= baseline else snapshot]
        for _ in range(max_steps):
            transport.send("\x1b[B")
            time.sleep(step_delay)
            snapshot = transport.snapshot()
            snapshots.append(snapshot[baseline:] if len(snapshot) >= baseline else snapshot)
        return parse_antigravity_model_options("\n".join(snapshots))
    finally:
        try:
            transport.send("\x1b")
            transport.send("\x04")
        except (OSError, RuntimeError):
            pass
        transport.close()


def antigravity_model_options(
    *,
    cwd: Path | None = None,
    executable: str = ANTIGRAVITY_EXECUTABLE,
    refresh: bool = False,
) -> tuple[str, ...]:
    env_options = _antigravity_model_options_from_env()
    if env_options is not None:
        return env_options
    global _ANTIGRAVITY_MODEL_OPTIONS_CACHE
    if _ANTIGRAVITY_MODEL_OPTIONS_CACHE is not None and not refresh:
        return _ANTIGRAVITY_MODEL_OPTIONS_CACHE
    _ANTIGRAVITY_MODEL_OPTIONS_CACHE = discover_antigravity_model_options(cwd=cwd, executable=executable)
    return _ANTIGRAVITY_MODEL_OPTIONS_CACHE


def _wait_until_transport_ready(transport: PtyProcess, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not transport.is_running():
            return
        snapshot = transport.snapshot()
        if snapshot and transport.idle_for() >= 0.25:
            return
        time.sleep(0.05)


def _wait_until_model_picker_has_options(transport: PtyProcess, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not transport.is_running():
            return
        if parse_antigravity_model_options(transport.snapshot()):
            return
        time.sleep(0.05)


def _antigravity_error_kind(message: str) -> str:
    normalized = message.lower()
    if any(token in normalized for token in ("auth", "oauth", "sign in", "signed in", "login", "credential")):
        return "auth_error"
    if "timed out" in normalized or "timeout" in normalized or "deadline" in normalized:
        return "timeout_error"
    return "process_exit"


def _permission_flags(permission_mode: str) -> list[str]:
    normalized = permission_mode.strip() or "default"
    if normalized not in ANTIGRAVITY_PERMISSION_MODES:
        raise ValueError(
            "Unsupported Antigravity permission mode: "
            f"{permission_mode}; expected one of {', '.join(sorted(ANTIGRAVITY_PERMISSION_MODES))}"
        )
    if normalized == "dangerously-skip-permissions":
        return ["--dangerously-skip-permissions"]
    if normalized in {"sandbox", "proceed-in-sandbox"}:
        return ["--sandbox"]
    return []


def antigravity_settings_path() -> Path:
    return Path.home() / ".gemini" / "antigravity-cli" / "settings.json"


def read_antigravity_settings(path: Path | None = None) -> dict:
    settings_path = Path(path) if path is not None else antigravity_settings_path()
    if not settings_path.exists():
        return {}
    try:
        payload = json.loads(settings_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Antigravity settings are not valid JSON: {settings_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Antigravity settings must be a JSON object: {settings_path}")
    return payload


def current_antigravity_model(path: Path | None = None) -> str | None:
    model = read_antigravity_settings(path).get("model")
    return model if isinstance(model, str) and model else None


def set_antigravity_model(
    model: str | None,
    path: Path | None = None,
    *,
    available_models: Iterable[str] | None = None,
) -> str | None:
    if model is None:
        return None
    normalized = model.strip()
    allowed_models = _unique_text(available_models if available_models is not None else antigravity_model_options())
    if not allowed_models:
        raise ValueError(
            "Could not load Antigravity model options from `agy /model`; "
            "verify that the local `agy` CLI is installed and authenticated."
        )
    if normalized not in allowed_models:
        allowed = ", ".join(allowed_models)
        raise ValueError(f"Unsupported Antigravity model: {model}; expected one of {allowed}")
    settings_path = Path(path) if path is not None else antigravity_settings_path()
    settings = read_antigravity_settings(settings_path)
    settings["model"] = normalized
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = settings_path.with_suffix(settings_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    temp_path.replace(settings_path)
    return normalized


def _antigravity_activity_payload(payload: dict) -> dict:
    source = str(payload.get("source") or "").upper()
    event_type = str(payload.get("type") or "").upper()
    if "TOOL" not in source and "TOOL" not in event_type and not payload.get("tool"):
        return {}
    name = (
        payload.get("name")
        or payload.get("tool_name")
        or payload.get("tool")
        or payload.get("title")
        or payload.get("command")
        or "tool"
    )
    rendered_name = _one_line_text(str(name))
    status = str(payload.get("status") or "").lower()
    finished = status in {"done", "success", "finished", "complete", "completed"} or "RESULT" in event_type
    label_prefix = "tool result" if finished and "RESULT" in event_type else "tool"
    detail = payload.get("description") or payload.get("content") or payload.get("command") or ""
    rendered_detail = _one_line_text(str(detail)) if detail else ""
    label = f"{label_prefix}: {rendered_name}" + (f" {rendered_detail}" if rendered_detail else "")
    return {
        "kind": "tool_output" if finished and "RESULT" in event_type else "tool_started",
        "title": "Antigravity tool",
        "backend_label": label,
        "status": "finished" if finished else "started",
        "details": {
            "name": rendered_name,
            "status": payload.get("status"),
            "input": payload.get("input") or payload.get("args") or payload.get("parameters"),
            "output": payload.get("output") or payload.get("result") or payload.get("content"),
        },
    }


class AntigravityAdapter(BackendAdapter):
    name = BackendName.ANTIGRAVITY

    def __init__(
        self,
        *,
        model: str | None = None,
        permission_mode: str = "default",
        print_timeout: str = ANTIGRAVITY_DEFAULT_PRINT_TIMEOUT,
        executable: str = ANTIGRAVITY_EXECUTABLE,
    ) -> None:
        super().__init__()
        self.model = model
        self.permission_mode = permission_mode
        self.print_timeout = print_timeout
        self.executable = executable

    def build_start_command(self, cwd: Path) -> list[str]:
        return [self.executable, *_permission_flags(self.permission_mode)]

    def build_prompt_interactive_command(self, prompt: str, cwd: Path) -> list[str]:
        return [
            self.executable,
            *_permission_flags(self.permission_mode),
            "--prompt-interactive",
            prompt,
        ]

    def build_command(self, prompt: str, cwd: Path) -> list[str]:
        command = [
            self.executable,
            *_permission_flags(self.permission_mode),
            "--print",
            prompt,
        ]
        if self.print_timeout:
            command.extend(["--print-timeout", self.print_timeout])
        return command

    def run(self, prompt: str, cwd: Path):
        if shutil.which(self.executable) is None:
            yield BackendEvent(
                type=EventType.BACKEND_FAILED,
                error=NormalizedError(
                    kind="backend_error",
                    message=(
                        "Antigravity CLI executable not found. "
                        f"Install the official `agy` launcher or add it to PATH: {self.executable}"
                    ),
                    details={"executable": self.executable},
                ),
            )
            return
        yield from super().run(prompt, cwd)

    def parse_stdout_line(self, line: str) -> list[BackendEvent]:
        payload = self.parse_json(line)
        if payload is not None:
            return self._parse_json_payload(payload)
        if not line.strip():
            return []
        return [
            BackendEvent(type=EventType.OUTPUT_STARTED),
            BackendEvent(type=EventType.OUTPUT_DELTA, text=f"{line}\n"),
        ]

    def _parse_json_payload(self, payload: dict) -> list[BackendEvent]:
        source = payload.get("source")
        event_type = payload.get("type")
        status = payload.get("status")
        content = payload.get("content")
        activity = _antigravity_activity_payload(payload)
        if activity:
            return [activity_event(raw=payload, **activity)]
        if source == "MODEL" and isinstance(content, str) and content:
            return [
                BackendEvent(type=EventType.OUTPUT_STARTED, raw=payload),
                BackendEvent(type=EventType.OUTPUT_DELTA, text=content, raw=payload),
            ]
        if status == "ERROR" or event_type == "ERROR":
            message = payload.get("message") or payload.get("error") or content or "Antigravity request failed"
            return [
                BackendEvent(
                    type=EventType.BACKEND_FAILED,
                    error=NormalizedError(kind="backend_error", message=str(message)),
                    raw=payload,
                )
            ]
        return []

    def completion_events(self, exit_code: int, stderr: str, saw_output: bool) -> list[BackendEvent]:
        stderr_text = stderr.strip()
        if exit_code == 0 and saw_output:
            return [BackendEvent(type=EventType.BACKEND_SUCCEEDED)]
        if exit_code == 0:
            return [
                BackendEvent(
                    type=EventType.BACKEND_FAILED,
                    error=NormalizedError(
                        kind="backend_error",
                        message=(
                            "Antigravity CLI produced no stdout in print mode. "
                            "Run `agy --print` directly to verify the local CLI contract."
                        ),
                        details={"reason": "empty_stdout", "command": "agy --print"},
                    ),
                )
            ]
        message = stderr_text or f"Antigravity CLI exited with code {exit_code}"
        return [
            BackendEvent(
                type=EventType.BACKEND_FAILED,
                error=NormalizedError(
                    kind=_antigravity_error_kind(message),
                    message=message,
                    exit_code=exit_code,
                ),
            )
        ]

    def run_interactive_terminal(self, prompt: str, cwd: Path) -> None:
        if shutil.which(self.executable) is None:
            raise FileNotFoundError(f"Antigravity CLI executable not found: {self.executable}")
        transport = PtyProcess(
            self.build_start_command(cwd),
            cwd,
            env={"TERM": "xterm-256color", "AGY_CLI_HIDE_ACCOUNT_INFO": "1"},
        )
        try:
            self._send_interactive_prompt(transport, prompt)
            transport.run_foreground_until_ctrl_g()
        finally:
            transport.close()

    def _send_interactive_prompt(self, transport: PtyProcess, prompt: str) -> None:
        self._wait_until_ready(transport, timeout=30.0)
        if not prompt:
            return
        transport.send(prompt)
        transport.send("\r")
        transport.wait_for_quiet(idle_for=0.5, timeout=5.0)

    def _wait_until_ready(self, transport: PtyProcess, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not transport.is_running():
                return
            snapshot = transport.snapshot()
            if ">" in snapshot and ("? for shortcuts" in snapshot or "Gemini" in snapshot) and transport.idle_for() >= 0.3:
                return
            time.sleep(0.05)


__all__ = [
    "ANTIGRAVITY_DEFAULT_PRINT_TIMEOUT",
    "ANTIGRAVITY_EXECUTABLE",
    "ANTIGRAVITY_MODEL_OPTIONS_ENV",
    "ANTIGRAVITY_PERMISSION_MODES",
    "AntigravityAdapter",
    "antigravity_model_options",
    "antigravity_settings_path",
    "current_antigravity_model",
    "discover_antigravity_model_options",
    "parse_antigravity_model_options",
    "read_antigravity_settings",
    "set_antigravity_model",
]
