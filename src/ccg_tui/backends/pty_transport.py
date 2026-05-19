from __future__ import annotations

import errno
import fcntl
import json
import os
import pty
import select
import struct
import subprocess
import sys
import threading
import time
import termios
import tty
from pathlib import Path
from typing import Any, Mapping


class PtyProcess:
    def __init__(
        self,
        command: list[str],
        cwd: Path,
        *,
        env: Mapping[str, str] | None = None,
        max_buffer_chars: int = 65_536,
    ) -> None:
        self.command = list(command)
        self.cwd = Path(cwd)
        self.max_buffer_chars = max_buffer_chars
        self._buffer = ""
        self._query_buffer = ""
        self._lock = threading.Lock()
        self._closed = False
        self._foreground = threading.Event()
        self._last_output_at = time.monotonic()
        self._master_fd, slave_fd = pty.openpty()
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
        child_env = os.environ.copy()
        if env is not None:
            child_env.update(env)

        def _preexec() -> None:
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        self.process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd),
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=child_env,
            close_fds=True,
            preexec_fn=_preexec,
        )
        os.close(slave_fd)
        os.set_blocking(self._master_fd, False)
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _reader_loop(self) -> None:
        while not self._closed:
            if self._foreground.is_set():
                time.sleep(0.05)
                continue
            if self.process.poll() is not None:
                self._drain_once()
                return
            self._drain_once(timeout=0.1)

    def _drain_once(self, timeout: float = 0.0) -> None:
        try:
            ready, _, _ = select.select([self._master_fd], [], [], timeout)
        except (OSError, ValueError):
            return
        if not ready:
            return
        while True:
            try:
                chunk = os.read(self._master_fd, 4096)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno == errno.EIO:
                    return
                raise
            if not chunk:
                return
            self._append_output(chunk)
            self._respond_to_terminal_queries()

    def _append_output(self, chunk: bytes) -> str:
        decoded = chunk.decode("utf-8", errors="ignore")
        with self._lock:
            self._buffer = (self._buffer + decoded)[-self.max_buffer_chars :]
            self._query_buffer = (self._query_buffer + decoded)[-512:]
            self._last_output_at = time.monotonic()
        return decoded

    def _respond_to_terminal_queries(self) -> None:
        queries = {
            "\x1b[6n": "\x1b[1;1R",
            "\x1b[c": "\x1b[?1;2c",
            "\x1b]10;?\x1b\\": "\x1b]10;rgb:ffff/ffff/ffff\x1b\\",
            "\x1b]11;?\x1b\\": "\x1b]11;rgb:0000/0000/0000\x1b\\",
            "\x1b[?u": "\x1b[?1u",
        }
        pending: list[str] = []
        with self._lock:
            for query, response in queries.items():
                while query in self._query_buffer:
                    self._query_buffer = self._query_buffer.replace(query, "", 1)
                    pending.append(response)
        for response in pending:
            try:
                os.write(self._master_fd, response.encode("utf-8"))
            except OSError:
                return

    def snapshot(self) -> str:
        with self._lock:
            return self._buffer

    def idle_for(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_output_at

    def wait_for_quiet(self, *, idle_for: float = 0.6, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        saw_output = bool(self.snapshot())
        while time.monotonic() < deadline:
            if self.snapshot():
                saw_output = True
            if saw_output and self.idle_for() >= idle_for:
                return True
            if self.process.poll() is not None and self.idle_for() >= idle_for:
                return saw_output
            time.sleep(0.05)
        return saw_output and self.idle_for() >= idle_for

    def send(self, text: str) -> None:
        if self._closed:
            raise RuntimeError("PTY is already closed")
        os.write(self._master_fd, text.encode("utf-8"))

    def run_foreground_until_ctrl_g(self) -> None:
        if self._closed:
            raise RuntimeError("PTY is already closed")
        stdin_fd = sys.stdin.fileno()
        stdout_fd = sys.stdout.fileno()
        if not os.isatty(stdin_fd) or not os.isatty(stdout_fd):
            return
        self._foreground.set()
        time.sleep(0.08)
        original_attrs = termios.tcgetattr(stdin_fd)
        try:
            tty.setraw(stdin_fd)
            os.write(stdout_fd, b"\x1b[2J\x1b[H")
            snapshot = self.snapshot()
            if snapshot:
                os.write(stdout_fd, snapshot.encode("utf-8", errors="ignore"))
            while self.is_running():
                try:
                    readable, _, _ = select.select([stdin_fd, self._master_fd], [], [], 0.1)
                except (OSError, ValueError):
                    return
                if stdin_fd in readable:
                    data = os.read(stdin_fd, 4096)
                    if not data:
                        return
                    ctrl_g_index = data.find(b"\x07")
                    if ctrl_g_index >= 0:
                        prefix = data[:ctrl_g_index]
                        if prefix:
                            os.write(self._master_fd, prefix)
                        return
                    os.write(self._master_fd, data)
                if self._master_fd in readable:
                    while True:
                        try:
                            chunk = os.read(self._master_fd, 4096)
                        except BlockingIOError:
                            break
                        except OSError as exc:
                            if exc.errno == errno.EIO:
                                return
                            raise
                        if not chunk:
                            return
                        self._append_output(chunk)
                        os.write(stdout_fd, chunk)
                        self._respond_to_terminal_queries()
        finally:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, original_attrs)
            self._foreground.clear()

    def is_running(self) -> bool:
        return self.process.poll() is None

    def exit_code(self) -> int | None:
        return self.process.poll()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        try:
            os.close(self._master_fd)
        except OSError:
            pass


class JsonlTail:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._offset = 0
        self._partial = b""

    def seek_to_end(self) -> None:
        if self.path.exists():
            self._offset = self.path.stat().st_size
        else:
            self._offset = 0
        self._partial = b""

    def read_new_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset = handle.tell()
        if not chunk:
            return []
        data = self._partial + chunk
        lines = data.split(b"\n")
        self._partial = lines.pop()
        records: list[dict[str, Any]] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records
