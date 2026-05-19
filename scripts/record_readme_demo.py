#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

import pexpect
import pyte


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GIF = REPO_ROOT / "docs" / "assets" / "ccg-tui-demo.gif"
DEFAULT_TRANSCRIPT_DIR = REPO_ROOT / "runtime" / "readme-demo-transcripts"

COLS = 120
ROWS = 34
FPS = 6
WIDTH = 1280
HEIGHT = 720
GIF_WIDTH = 800
FONT_SIZE = 15
LINE_HEIGHT = 20
TEXT_X = 32
TEXT_Y = 36
FONT_FAMILY = "DejaVu Sans Mono"
BACKGROUND = "#071018"
FOREGROUND = "#dbe7f3"


class DemoRecorder:
    def __init__(self, *, transcript_dir: Path) -> None:
        self.screen = pyte.Screen(COLS, ROWS)
        self.stream = pyte.Stream(self.screen)
        self.frames: list[list[str]] = []
        child_env = os.environ.copy()
        child_env.update(
            {
                "TERM": "xterm-256color",
                "COLORTERM": "truecolor",
                "COLUMNS": str(COLS),
                "LINES": str(ROWS),
                "CCG_TUI_FAKE_BACKEND": "1",
                "CCG_TUI_FAKE_ACTIVITY_PREFIX": "backend activity:",
                "CCG_TUI_FAKE_ACTIVITY_TITLE": "Backend activity",
                "CCG_TUI_FAKE_REPLY_PREFIX": "demo reply:",
            }
        )
        relative_transcript_dir = os.path.relpath(transcript_dir, REPO_ROOT)
        self.child = pexpect.spawn(
            "uv",
            [
                "run",
                "ccg-tui",
                "--transcript-dir",
                relative_transcript_dir,
            ],
            cwd=str(REPO_ROOT),
            env=child_env,
            dimensions=(ROWS, COLS),
            encoding="utf-8",
            codec_errors="ignore",
            timeout=10,
        )

    def close(self) -> None:
        self._drain(timeout=0.05)
        if self.child.isalive():
            self.child.sendcontrol("c")
            time.sleep(0.1)
            self._drain(timeout=0.05)
        if self.child.isalive():
            self.child.terminate(force=True)

    def wait_for(self, text: str, *, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        needle = text.lower()
        while time.monotonic() < deadline:
            self._drain(timeout=0.05)
            if needle in self.screen_text().lower():
                return
            if not self.child.isalive():
                break
            time.sleep(0.05)
        raise TimeoutError(f"Timed out waiting for {text!r}. Last screen:\n{self.screen_text()}")

    def type_text(self, text: str, *, capture_every: int = 3) -> None:
        for index, char in enumerate(text):
            self.child.send(char)
            time.sleep(0.008)
            self._drain(timeout=0.01)
            if index % capture_every == 0 or index == len(text) - 1:
                self.capture(repeat=1)

    def enter(self) -> None:
        self.child.send("\r")
        time.sleep(0.08)
        self._drain(timeout=0.05)
        self.capture(repeat=2)

    def hold(self, seconds: float) -> None:
        self._drain(timeout=0.05)
        repeats = max(1, round(seconds * FPS))
        self.capture(repeat=repeats)

    def capture(self, *, repeat: int = 1) -> None:
        self._drain(timeout=0.01)
        frame = list(self.screen.display)
        for _ in range(repeat):
            self.frames.append(frame)

    def screen_text(self) -> str:
        return "\n".join(self.screen.display)

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
            self.stream.feed(chunk)
            timeout = 0


def run_demo(transcript_dir: Path) -> list[list[str]]:
    recorder = DemoRecorder(transcript_dir=transcript_dir)
    try:
        recorder.wait_for("Select Backend")
        recorder.hold(0.8)
        recorder.type_text("1")
        recorder.wait_for("Started codex session")
        recorder.hold(0.7)

        recorder.type_text("/model gpt-5.5")
        recorder.enter()
        recorder.wait_for("Model set to gpt-5.5")
        recorder.hold(0.7)

        recorder.type_text("/permissions ask")
        recorder.enter()
        recorder.wait_for("Permissions set to Ask before actions")
        recorder.hold(0.7)

        recorder.type_text("/task start README demo implementation")
        recorder.enter()
        recorder.wait_for("Started task: README demo implementation")
        recorder.hold(0.7)

        recorder.type_text("Inspect this repo and suggest the next safest implementation step", capture_every=4)
        recorder.enter()
        recorder.wait_for("demo reply:")
        recorder.hold(0.9)

        recorder.type_text("/details")
        recorder.enter()
        recorder.wait_for("Activity details expanded")
        recorder.hold(1.0)

        recorder.type_text("/summarize")
        recorder.enter()
        recorder.wait_for("Summary saved:")
        recorder.hold(1.1)

        recorder.type_text("/capabilities")
        recorder.enter()
        recorder.wait_for("Routing Capability Registry")
        recorder.hold(1.3)

        recorder.type_text("/handoff claude sonnet continue this implementation")
        recorder.enter()
        recorder.wait_for("Handoff preview:")
        recorder.hold(1.6)

        recorder.type_text("/quit")
        recorder.enter()
        recorder.hold(0.4)
        return recorder.frames
    finally:
        recorder.close()


def render_svg(lines: Sequence[str]) -> str:
    tspans = []
    for index, line in enumerate(lines):
        escaped = html.escape(line.rstrip())
        if index == 0:
            tspans.append(f'<tspan x="{TEXT_X}" y="{TEXT_Y}">{escaped}</tspan>')
        else:
            tspans.append(f'<tspan x="{TEXT_X}" dy="{LINE_HEIGHT}">{escaped}</tspan>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  <rect width="100%" height="100%" fill="{BACKGROUND}"/>
  <text font-family="{FONT_FAMILY}" font-size="{FONT_SIZE}" fill="{FOREGROUND}" xml:space="preserve">
    {''.join(tspans)}
  </text>
</svg>
"""


def render_frames(frames: Sequence[Sequence[str]], frame_dir: Path) -> None:
    for index, frame in enumerate(frames):
        svg_path = frame_dir / f"frame_{index:04d}.svg"
        png_path = frame_dir / f"frame_{index:04d}.png"
        svg_path.write_text(render_svg(frame), encoding="utf-8")
        subprocess.run(["convert", str(svg_path), str(png_path)], check=True)


def encode_gif(frame_dir: Path, *, gif_path: Path) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frame_dir / "frame_%04d.png")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-framerate",
            str(FPS),
            "-i",
            pattern,
            "-filter_complex",
            (
                f"[0:v]scale={GIF_WIDTH}:-1:flags=lanczos,split[s0][s1];"
                "[s0]palettegen=max_colors=80[p];"
                "[s1][p]paletteuse=dither=bayer:bayer_scale=5"
            ),
            str(gif_path),
        ],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record the README demo animation for CCG TUI.")
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--transcript-dir", type=Path, default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--keep-frames", type=Path, help="Optional directory to keep rendered frame PNG/SVG files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for command in ("uv", "convert", "ffmpeg"):
        if shutil.which(command) is None:
            raise RuntimeError(f"{command} is required to record the README demo")

    frames = run_demo(args.transcript_dir)
    if args.keep_frames:
        frame_dir = args.keep_frames
        frame_dir.mkdir(parents=True, exist_ok=True)
        render_frames(frames, frame_dir)
        encode_gif(frame_dir, gif_path=args.gif)
    else:
        with tempfile.TemporaryDirectory(prefix="ccg-tui-demo-") as temp_dir:
            frame_dir = Path(temp_dir)
            render_frames(frames, frame_dir)
            encode_gif(frame_dir, gif_path=args.gif)

    print(f"Wrote {args.gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
