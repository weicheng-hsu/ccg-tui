import os
from collections.abc import Iterable
from pathlib import Path

from ccg_tui.models import BackendEvent, BackendName, EventType

from .antigravity import AntigravityAdapter
from .base import BackendAdapter
from .claude import ClaudeAdapter
from .codex import CodexAdapter
from .gemini import GeminiAdapter

_UNSET = object()
_BACKEND_NAMES = {backend.value for backend in BackendName}


class FakeBackendAdapter(BackendAdapter):
    def __init__(
        self,
        name: str,
        *,
        model: str | None | object = _UNSET,
        approval_policy: str | object = _UNSET,
        sandbox_mode: str | object = _UNSET,
        permission_mode: str | object = _UNSET,
        approval_mode: str | object = _UNSET,
    ) -> None:
        super().__init__()
        self.name = BackendName(name)
        self.model = None if model is _UNSET else model
        self.approval_policy = "on-request" if approval_policy is _UNSET else approval_policy
        self.sandbox_mode = "workspace-write" if sandbox_mode is _UNSET else sandbox_mode
        self.permission_mode = "default" if permission_mode is _UNSET else permission_mode
        self.approval_mode = "default" if approval_mode is _UNSET else approval_mode

    def build_command(self, prompt: str, cwd: Path) -> list[str]:
        return []

    def parse_stdout_line(self, line: str) -> list[BackendEvent]:
        return []

    def run(self, prompt: str, cwd: Path) -> Iterable[BackendEvent]:
        activity_prefix = os.environ.get("CCG_TUI_FAKE_ACTIVITY_PREFIX", "fake activity:")
        activity_title = os.environ.get("CCG_TUI_FAKE_ACTIVITY_TITLE", "Fake backend activity")
        reply_prefix = os.environ.get("CCG_TUI_FAKE_REPLY_PREFIX", "fake reply to")
        activity_text = f"{activity_prefix} {prompt}".strip()
        reply_text = f"{reply_prefix} {prompt}".strip()
        yield BackendEvent(type=EventType.SESSION_STARTED, session_id=f"fake-{self.name.value}-session")
        yield BackendEvent(
            type=EventType.ACTIVITY,
            text=activity_text,
            activity={
                "kind": "progress",
                "title": activity_title,
                "backend_label": activity_text,
                "status": "finished",
                "details": {"prompt": prompt},
            },
        )
        yield BackendEvent(type=EventType.OUTPUT_STARTED)
        yield BackendEvent(type=EventType.OUTPUT_DELTA, text=reply_text)
        yield BackendEvent(type=EventType.BACKEND_SUCCEEDED)


def build_backend(
    name: str,
    *,
    model: str | None | object = _UNSET,
    approval_policy: str | object = _UNSET,
    sandbox_mode: str | object = _UNSET,
    permission_mode: str | object = _UNSET,
    approval_mode: str | object = _UNSET,
) -> BackendAdapter:
    normalized = name.strip().lower()
    if os.environ.get("CCG_TUI_FAKE_BACKEND") == "1" and normalized in _BACKEND_NAMES:
        return FakeBackendAdapter(
            normalized,
            model=model,
            approval_policy=approval_policy,
            sandbox_mode=sandbox_mode,
            permission_mode=permission_mode,
            approval_mode=approval_mode,
        )
    if normalized == "codex":
        kwargs = {}
        if model is not _UNSET:
            kwargs["model"] = model
        if approval_policy is not _UNSET:
            kwargs["approval_policy"] = approval_policy
        if sandbox_mode is not _UNSET:
            kwargs["sandbox_mode"] = sandbox_mode
        return CodexAdapter(**kwargs)
    if normalized == "claude":
        kwargs = {}
        if model is not _UNSET:
            kwargs["model"] = model
        if permission_mode is not _UNSET:
            kwargs["permission_mode"] = permission_mode
        return ClaudeAdapter(**kwargs)
    if normalized == "gemini":
        kwargs = {}
        if model is not _UNSET:
            kwargs["model"] = model
        if approval_mode is not _UNSET:
            kwargs["approval_mode"] = approval_mode
        return GeminiAdapter(**kwargs)
    if normalized == "antigravity":
        kwargs = {}
        if model is not _UNSET:
            kwargs["model"] = model
        if permission_mode is not _UNSET:
            kwargs["permission_mode"] = permission_mode
        return AntigravityAdapter(**kwargs)
    raise ValueError(f"Unsupported backend: {name}")


__all__ = [
    "AntigravityAdapter",
    "BackendAdapter",
    "CodexAdapter",
    "ClaudeAdapter",
    "GeminiAdapter",
    "FakeBackendAdapter",
    "build_backend",
]
