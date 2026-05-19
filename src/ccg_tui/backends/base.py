from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from ccg_tui.models import BackendEvent, BackendName, EventType, NormalizedError


def activity_event(
    *,
    kind: str,
    title: str,
    backend_label: str | None = None,
    status: str | None = None,
    details: dict | None = None,
    raw: dict | None = None,
) -> BackendEvent:
    activity = {
        "kind": kind,
        "title": title,
        "backend_label": backend_label or title,
        "status": status,
        "details": details or {},
    }
    return BackendEvent(type=EventType.ACTIVITY, text=activity["backend_label"], activity=activity, raw=raw)


class BackendSession(ABC):
    @abstractmethod
    def run(self, prompt: str) -> Iterable[BackendEvent]:
        raise NotImplementedError

    def start_interactive(self, prompt: str) -> None:
        raise RuntimeError("Backend session does not support interactive passthrough")

    def send_interactive_input(self, text: str) -> None:
        raise RuntimeError("Backend session does not support interactive passthrough")

    def interactive_snapshot(self) -> str:
        return ""

    def run_interactive_terminal(self, prompt: str) -> None:
        self.start_interactive(prompt)

    def close(self) -> None:
        return None


class OneShotBackendSession(BackendSession):
    def __init__(self, adapter: "BackendAdapter", cwd: Path) -> None:
        self.adapter = adapter
        self.cwd = Path(cwd)

    def run(self, prompt: str) -> Iterable[BackendEvent]:
        process = subprocess.Popen(
            self.adapter.build_command(prompt, self.cwd),
            cwd=str(self.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        saw_output = False
        assert process.stdout is not None
        for line in process.stdout:
            for event in self.adapter.parse_stdout_line(line.rstrip("\n")):
                if event.type is EventType.OUTPUT_DELTA and event.text:
                    saw_output = True
                yield event
        stderr = ""
        if process.stderr is not None:
            stderr = process.stderr.read()
        exit_code = process.wait()
        for event in self.adapter.completion_events(exit_code=exit_code, stderr=stderr, saw_output=saw_output):
            yield event


def text_delta(previous: str, current: str) -> str:
    if not current or current == previous:
        return ""
    if current.startswith(previous):
        return current[len(previous) :]
    return current


class BackendAdapter(ABC):
    name: BackendName

    def __init__(self) -> None:
        self._session: BackendSession | None = None

    @abstractmethod
    def build_command(self, prompt: str, cwd: Path) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def parse_stdout_line(self, line: str) -> list[BackendEvent]:
        raise NotImplementedError

    def completion_events(self, exit_code: int, stderr: str, saw_output: bool) -> list[BackendEvent]:
        if exit_code == 0:
            return []
        return [
            BackendEvent(
                type=EventType.BACKEND_FAILED,
                error=NormalizedError(
                    kind="process_exit",
                    message=stderr.strip() or f"{self.name.value} exited with code {exit_code}",
                    exit_code=exit_code,
                ),
            )
        ]

    def parse_json(self, line: str) -> dict | None:
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def open_session(self, cwd: Path) -> BackendSession:
        return OneShotBackendSession(self, cwd)

    def run(self, prompt: str, cwd: Path) -> Iterable[BackendEvent]:
        if self._session is None:
            self._session = self.open_session(cwd)
        yield from self._session.run(prompt)

    def start_interactive(self, prompt: str, cwd: Path) -> None:
        if self._session is None:
            self._session = self.open_session(cwd)
        self._session.start_interactive(prompt)

    def send_interactive_input(self, text: str) -> None:
        if self._session is None:
            raise RuntimeError("No backend session is active")
        self._session.send_interactive_input(text)

    def interactive_snapshot(self) -> str:
        if self._session is None:
            return ""
        return self._session.interactive_snapshot()

    def run_interactive_terminal(self, prompt: str, cwd: Path) -> None:
        if self._session is None:
            self._session = self.open_session(cwd)
        self._session.run_interactive_terminal(prompt)

    def close(self) -> None:
        if self._session is None:
            return
        self._session.close()
        self._session = None
