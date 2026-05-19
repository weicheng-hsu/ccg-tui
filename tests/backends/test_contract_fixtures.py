from __future__ import annotations

from pathlib import Path

import pytest

from ccg_tui.backends.base import BackendAdapter, BackendEvent
from ccg_tui.backends.claude import ClaudeAdapter
from ccg_tui.backends.codex import CodexAdapter
from ccg_tui.backends.gemini import GeminiAdapter

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _event_summary(event: BackendEvent) -> tuple[str, str | None, str | None, str | None]:
    return (
        event.type.value,
        event.session_id,
        event.text or None,
        event.error.kind if event.error is not None else None,
    )


def _parse_fixture(adapter: BackendAdapter, fixture_name: str) -> list[BackendEvent]:
    events: list[BackendEvent] = []
    for line in (FIXTURE_DIR / fixture_name).read_text().splitlines():
        events.extend(adapter.parse_stdout_line(line))
    return events


@pytest.mark.parametrize(
    ("adapter", "fixture_name", "expected"),
    [
        (
            CodexAdapter(),
            "codex_success.jsonl",
            [
                ("session_started", "019de88f-dc59-7ff1-a88c-46ba37c49aed", None, None),
                ("output_started", None, None, None),
                ("output_delta", None, "CCG_CODEX_SMOKE", None),
                ("backend_succeeded", None, None, None),
            ],
        ),
        (
            CodexAdapter(),
            "codex_activity_and_failure.jsonl",
            [
                ("activity", None, "tool: rg --files cwd=/tmp/project", None),
                ("backend_failed", None, None, "backend_error"),
            ],
        ),
        (
            ClaudeAdapter(),
            "claude_success.jsonl",
            [
                ("session_started", "d6e32394-8029-4c1b-b12d-bd42e56f1b59", None, None),
                ("output_started", None, None, None),
                ("output_delta", None, "CCG_CLAUDE_SMOKE", None),
                ("backend_succeeded", None, None, None),
            ],
        ),
        (
            ClaudeAdapter(),
            "claude_tool_and_failure.jsonl",
            [
                ("activity", None, "tools: Read file_path=README.md", None),
                ("backend_failed", None, None, "backend_error"),
            ],
        ),
        (
            GeminiAdapter(),
            "gemini_success.jsonl",
            [
                ("session_started", "c550c3ff-ba5f-44bf-8666-aabc22976c12", None, None),
                ("output_started", None, None, None),
                ("output_delta", None, "CCG_GEMINI_SMOKE", None),
                ("backend_succeeded", None, None, None),
            ],
        ),
        (
            GeminiAdapter(),
            "gemini_tool_and_auth_failure.jsonl",
            [
                ("activity", None, "tools: read_file file_path=pyproject.toml", None),
                ("output_started", None, None, None),
                ("output_delta", None, "checking files", None),
                ("backend_failed", None, None, "auth_error"),
            ],
        ),
        (
            GeminiAdapter(),
            "gemini_direct_tool_use.jsonl",
            [
                ("session_started", "direct-tool-session", None, None),
                ("activity", None, "tools: read_file file_path=pyproject.toml", None),
                ("activity", None, "tool result: read_file", None),
                ("output_started", None, None, None),
                ("output_delta", None, "checked", None),
                ("backend_succeeded", None, None, None),
            ],
        ),
    ],
)
def test_backend_contract_fixtures_normalize_to_expected_events(
    adapter: BackendAdapter,
    fixture_name: str,
    expected: list[tuple[str, str | None, str | None, str | None]],
) -> None:
    events = _parse_fixture(adapter, fixture_name)

    assert [_event_summary(event) for event in events] == expected
