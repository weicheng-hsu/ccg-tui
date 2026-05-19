from __future__ import annotations

import os
from pathlib import Path

import pytest

from .tui_harness import TuiProcess


@pytest.fixture
def tui_app(tmp_path: Path):
    if os.name == "nt":
        pytest.skip("pexpect-based fullscreen TUI tests require a POSIX PTY")

    apps: list[TuiProcess] = []

    def spawn(*args: str, cols: int = 100, rows: int = 32) -> TuiProcess:
        app = TuiProcess.spawn_ccg(
            cwd=Path(__file__).resolve().parents[2],
            artifact_dir=tmp_path / "artifacts",
            transcript_dir=tmp_path / "transcripts",
            args=args,
            cols=cols,
            rows=rows,
        )
        apps.append(app)
        return app

    yield spawn

    for app in apps:
        app.close()
