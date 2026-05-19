import json

from ccg_tui.backends.gemini import GeminiAdapter
from ccg_tui.backends.gemini import (
    _load_gemini_chat,
    _gemini_latest_turn_message,
    _gemini_project_slug,
    _gemini_tool_activity,
    _gemini_turn_has_tool_calls,
    _gemini_turn_text,
    _gemini_ui_ready,
    _gemini_user_text,
)
from ccg_tui.models import EventType


def test_gemini_adapter_builds_headless_command(tmp_path):
    adapter = GeminiAdapter()

    command = adapter.build_command("hello", tmp_path)

    assert command == ["gemini", "-p", "hello", "-o", "stream-json", "--approval-mode", "default"]


def test_gemini_adapter_forwards_model_and_approval_mode_to_headless_command(tmp_path):
    adapter = GeminiAdapter(model="gemini-3.1-pro-preview", approval_mode="yolo")

    command = adapter.build_command("hello", tmp_path)

    assert "--approval-mode" in command
    assert command[command.index("--approval-mode") + 1] == "yolo"
    assert "--model" in command
    assert command[command.index("--model") + 1] == "gemini-3.1-pro-preview"


def test_gemini_adapter_parses_real_cli_stream_format():
    adapter = GeminiAdapter()
    lines = [
        '{"type":"init","timestamp":"2026-04-21T07:24:19.759Z","session_id":"g-1","model":"auto-gemini-3"}',
        '{"type":"message","timestamp":"2026-04-21T07:24:19.767Z","role":"user","content":"hello"}',
        '{"type":"message","timestamp":"2026-04-21T07:24:24.404Z","role":"assistant","content":"world","delta":true}',
        '{"type":"result","timestamp":"2026-04-21T07:24:24.444Z","status":"success"}',
    ]

    events = [event for line in lines for event in adapter.parse_stdout_line(line)]

    assert events[0].type is EventType.SESSION_STARTED
    assert events[0].session_id == "g-1"
    assert events[1].type is EventType.OUTPUT_STARTED
    assert events[2].type is EventType.OUTPUT_DELTA
    assert events[2].text == "world"
    assert events[3].type is EventType.BACKEND_SUCCEEDED


def test_gemini_adapter_treats_plain_auth_error_as_failure():
    adapter = GeminiAdapter()

    events = adapter.parse_stdout_line("Please set an Auth method in your settings.json")

    assert events[-1].type is EventType.BACKEND_FAILED
    assert events[-1].error.kind == "auth_error"


def test_gemini_project_slug_prefers_registry_mapping(tmp_path, monkeypatch):
    registry_root = tmp_path / ".gemini"
    registry_root.mkdir()
    registry = registry_root / "projects.json"
    cwd = tmp_path / "workspace"
    cwd.mkdir()
    registry.write_text(json.dumps({"projects": {str(cwd.resolve()): "mapped-slug"}}))
    monkeypatch.setattr("ccg_tui.backends.gemini.Path.home", lambda: tmp_path)

    assert _gemini_project_slug(cwd) == "mapped-slug"


def test_load_gemini_chat_reads_jsonl_sessions(tmp_path):
    path = tmp_path / "session-2026-04-24T09-54-b95809bf.jsonl"
    path.write_text(
        '{"sessionId":"session-jsonl","lastUpdated":"start","kind":"main"}\n'
        '{"type":"user","content":[{"text":"hello"}]}\n'
        '{"type":"gemini","content":"checking","toolCalls":[{"name":"read_file"}]}\n'
        '{"$set":{"lastUpdated":"done"}}\n'
    )

    conversation = _load_gemini_chat(path)

    assert conversation is not None
    assert conversation["sessionId"] == "session-jsonl"
    assert conversation["lastUpdated"] == "done"
    assert len(conversation["messages"]) == 2
    assert conversation["messages"][1]["toolCalls"][0]["name"] == "read_file"


def test_gemini_user_text_extracts_prompt_text():
    text = _gemini_user_text(
        {
            "type": "user",
            "content": [
                {"text": "Reply with the exact text "},
                {"text": "PONG and nothing else."},
            ],
        }
    )

    assert text == "Reply with the exact text PONG and nothing else."


def test_gemini_turn_text_joins_multiple_assistant_messages_per_turn():
    messages = [
        {"type": "user", "content": [{"text": "Which objective does this project want to attend"}]},
        {
            "type": "gemini",
            "content": "I will check the `README.md` file to see if the project's objective is stated there.",
            "toolCalls": [{"name": "read_file"}],
        },
        {
            "type": "gemini",
            "content": "Based on the project's documentation, the overall goal is to build a vendor-agnostic TUI.",
        },
    ]

    text = _gemini_turn_text(messages, previous_count=1)

    assert text == (
        "I will check the `README.md` file to see if the project's objective is stated there.\n\n"
        "Based on the project's documentation, the overall goal is to build a vendor-agnostic TUI."
    )


def test_gemini_ui_ready_uses_latest_terminal_title():
    snapshot = (
        "\x1b]0;◇  Ready (ccg-integration)\x07"
        "\x1b]0;✦  Working… (ccg-integration)\x07"
        "\x1b]0;◇  Ready (ccg-integration)\x07"
    )

    assert _gemini_ui_ready(snapshot) is True


def test_gemini_ui_ready_detects_ready_title_without_prompt_text():
    snapshot = "\x1b]0;◇  Ready (ccg-tui)\x07"

    assert _gemini_ui_ready(snapshot) is True


def test_gemini_turn_has_tool_calls_detects_tool_phase():
    messages = [
        {"type": "user", "content": [{"text": "Which objective does this project want to attend"}]},
        {"type": "gemini", "content": "", "toolCalls": [{"name": "read_file"}]},
        {"type": "gemini", "content": "Final answer"},
    ]

    assert _gemini_turn_has_tool_calls(messages, previous_count=1) is True


def test_gemini_tool_activity_names_tools():
    message = {
        "type": "gemini",
        "toolCalls": [
            {"name": "read_file", "args": {"file_path": "README.md"}},
            {"name": "run_shell_command", "args": {"command": "rg --files"}},
        ],
    }

    assert _gemini_tool_activity(message) == "tools: read_file file_path=README.md; run_shell_command command=rg --files"


def test_gemini_tool_activity_prefers_backend_display_metadata():
    message = {
        "type": "gemini",
        "toolCalls": [
            {
                "name": "read_file",
                "args": {"file_path": "ignored"},
                "displayName": "ReadFile",
                "description": "pyproject.toml",
            }
        ],
    }

    assert _gemini_tool_activity(message) == "tools: ReadFile pyproject.toml"


def test_gemini_adapter_parses_activity_with_assistant_message():
    adapter = GeminiAdapter()
    line = (
        '{"type":"message","role":"assistant","content":"checking files",'
        '"toolCalls":[{"name":"read_file","args":{"file_path":"pyproject.toml"}}]}'
    )

    events = adapter.parse_stdout_line(line)

    assert events[0].type is EventType.ACTIVITY
    assert events[0].text == "tools: read_file file_path=pyproject.toml"
    assert events[0].activity["kind"] == "tool"
    assert events[0].activity["details"]["tool_calls"][0]["args"]["file_path"] == "pyproject.toml"
    assert events[1].type is EventType.OUTPUT_STARTED
    assert events[2].type is EventType.OUTPUT_DELTA


def test_gemini_latest_turn_message_returns_latest_gemini_message():
    messages = [
        {"type": "user", "content": [{"text": "hello"}]},
        {"type": "gemini", "content": "interim", "toolCalls": [{"name": "read_file"}]},
        {"type": "info", "content": "Update successful!"},
        {"type": "gemini", "content": "final"},
    ]

    latest = _gemini_latest_turn_message(messages, previous_count=1)

    assert latest is not None
    assert latest["content"] == "final"
