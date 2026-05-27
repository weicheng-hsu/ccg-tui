from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Sequence

import pexpect
import pyte


class TuiProcess:
    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        artifact_dir: Path,
        cols: int = 100,
        rows: int = 32,
        env: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self.command = list(command)
        self.cwd = Path(cwd)
        self.artifact_dir = Path(artifact_dir)
        self.transcript_dir: Path | None = None
        self.timeout = timeout
        self.raw_output = ""
        self.screen = pyte.Screen(cols, rows)
        self.stream = pyte.Stream(self.screen)
        child_env = os.environ.copy()
        child_env.update(
            {
                "TERM": "xterm-256color",
                "COLORTERM": "truecolor",
                "COLUMNS": str(cols),
                "LINES": str(rows),
            }
        )
        if env:
            child_env.update(env)
        self.child = pexpect.spawn(
            command[0],
            list(command[1:]),
            cwd=str(self.cwd),
            env=child_env,
            dimensions=(rows, cols),
            encoding="utf-8",
            codec_errors="ignore",
            timeout=timeout,
        )

    @classmethod
    def spawn_ccg(
        cls,
        *,
        cwd: Path,
        artifact_dir: Path,
        args: Sequence[str] = (),
        transcript_dir: Path,
        cols: int = 100,
        rows: int = 32,
    ) -> "TuiProcess":
        if shutil.which("uv") is None:
            raise RuntimeError("uv is required to run fullscreen TUI tests")
        relative_transcript_dir = os.path.relpath(transcript_dir, cwd)
        process = cls(
            [
                "uv",
                "run",
                "ccg-tui",
                "--transcript-dir",
                relative_transcript_dir,
                *args,
            ],
            cwd=cwd,
            artifact_dir=artifact_dir,
            cols=cols,
            rows=rows,
            env={
                "CCG_TUI_FAKE_BACKEND": "1",
                "CCG_TUI_CODEX_MODEL_OPTIONS": json.dumps(
                    [{"value": "gpt-5.5", "label": "GPT-5.5", "description": "Test Codex model."}]
                ),
                "CCG_TUI_CLAUDE_MODEL_OPTIONS": json.dumps(
                    [{"value": "sonnet", "label": "Sonnet", "description": "Test Claude model."}]
                ),
                "CCG_TUI_GEMINI_MODEL_OPTIONS": json.dumps(
                    [{"value": "gemini-2.5-flash", "label": "Gemini 2.5 Flash", "description": "Test Gemini model."}]
                ),
                "CCG_TUI_ANTIGRAVITY_MODEL_OPTIONS": "\n".join(
                    [
                        "Gemini 3.5 Flash (Medium)",
                        "Claude Sonnet 4.6 (Thinking)",
                    ]
                ),
            },
        )
        process.transcript_dir = Path(transcript_dir)
        return process

    def type(self, text: str, *, delay: float = 0.003) -> None:
        for char in text:
            self.child.send(char)
            if delay:
                time.sleep(delay)
            self._drain(timeout=0.01)

    def enter(self) -> None:
        self.child.send("\r")
        self._drain(timeout=0.05)

    def press(self, key: str) -> None:
        sequences = {
            "up": "\x1b[A",
            "down": "\x1b[B",
            "left": "\x1b[D",
            "right": "\x1b[C",
            "escape": "\x1b",
            "f2": "\x1bOQ",
            "f3": "\x1bOR",
            "ctrl-c": "\x03",
        }
        try:
            sequence = sequences[key.lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported key: {key}") from exc
        self.child.send(sequence)
        self._drain(timeout=0.05)

    def expect_text(self, text: str, *, timeout: float | None = None) -> None:
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        needle = text.lower()
        while time.monotonic() < deadline:
            self._drain(timeout=0.05)
            if needle in self.screen_text().lower():
                return
            if not self.child.isalive():
                break
            time.sleep(0.05)
        artifact = self.write_artifacts("expect-text-timeout")
        raise AssertionError(
            f"Timed out waiting for {text!r} in TUI screen. "
            f"Screen artifact: {artifact['screen']}"
        )

    def expect_no_text(self, text: str, *, settle_for: float = 0.2) -> None:
        deadline = time.monotonic() + settle_for
        while time.monotonic() < deadline:
            self._drain(timeout=0.05)
            time.sleep(0.02)
        if text.lower() in self.screen_text().lower():
            artifact = self.write_artifacts("unexpected-text")
            raise AssertionError(
                f"Found unexpected {text!r} in TUI screen. "
                f"Screen artifact: {artifact['screen']}"
            )

    def expect_exit(self, expected_code: int = 0, *, timeout: float | None = None) -> None:
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while time.monotonic() < deadline:
            self._drain(timeout=0.05)
            if not self.child.isalive():
                exit_code = self.child.exitstatus
                if exit_code is None:
                    exit_code = 128 + (self.child.signalstatus or 0)
                assert exit_code == expected_code
                return
            time.sleep(0.05)
        artifact = self.write_artifacts("exit-timeout")
        raise AssertionError(
            f"Timed out waiting for TUI exit. Screen artifact: {artifact['screen']}"
        )

    def screen_text(self) -> str:
        return "\n".join(self.screen.display)

    def write_artifacts(self, label: str) -> dict[str, Path]:
        self._drain(timeout=0.05)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        screen_path = self.artifact_dir / f"{label}.screen.txt"
        raw_path = self.artifact_dir / f"{label}.raw.log"
        screen_path.write_text(self.screen_text(), encoding="utf-8")
        raw_path.write_text(self.raw_output, encoding="utf-8")
        return {"screen": screen_path, "raw": raw_path}

    def close(self) -> None:
        self._drain(timeout=0.05)
        if self.child.isalive():
            self.child.sendcontrol("c")
            time.sleep(0.1)
            self._drain(timeout=0.05)
        if self.child.isalive():
            self.child.terminate(force=True)

    def _drain(self, *, timeout: float) -> None:
        while True:
            try:
                chunk = self.child.read_nonblocking(size=4096, timeout=timeout)
            except pexpect.TIMEOUT:
                return
            except pexpect.EOF:
                return
            if not chunk:
                return
            self.raw_output += chunk
            self.stream.feed(chunk)
            timeout = 0
