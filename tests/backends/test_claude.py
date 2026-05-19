from ccg_tui.backends.claude import (
    ClaudeAdapter,
    ClaudePtySession,
    _claude_exit_plan_mode_prompt_visible,
    _claude_project_slug,
    _claude_turn_complete,
    _claude_ui_ready,
    _extract_claude_activity,
    _extract_claude_text,
)
from ccg_tui.models import EventType


def test_claude_adapter_builds_stream_json_command(tmp_path):
    adapter = ClaudeAdapter()

    command = adapter.build_command("hello", tmp_path)

    assert command[:5] == ["claude", "-p", "--verbose", "--output-format", "stream-json"]
    assert command[command.index("--permission-mode") + 1] == "default"
    assert "--model" not in command
    assert command[-1] == "hello"


def test_claude_adapter_forwards_explicit_model_override(tmp_path):
    adapter = ClaudeAdapter(model="sonnet")

    command = adapter.build_command("hello", tmp_path)

    assert "--model" in command
    assert command[command.index("--model") + 1] == "sonnet"


def test_claude_adapter_parses_successful_stream():
    adapter = ClaudeAdapter()
    lines = [
        '{"type":"system","subtype":"init","session_id":"session-1"}',
        '{"type":"assistant","message":{"content":[{"type":"text","text":"hello from claude"}]}}',
        '{"type":"result","subtype":"success","result":"hello from claude"}',
    ]

    events = [event for line in lines for event in adapter.parse_stdout_line(line)]

    assert events[0].type is EventType.SESSION_STARTED
    assert events[0].session_id == "session-1"
    assert events[1].type is EventType.OUTPUT_STARTED
    assert events[2].type is EventType.OUTPUT_DELTA
    assert events[2].text == "hello from claude"
    assert events[3].type is EventType.BACKEND_SUCCEEDED


def test_claude_adapter_parses_error_result():
    adapter = ClaudeAdapter()

    events = adapter.parse_stdout_line('{"type":"result","subtype":"error","is_error":true,"result":"permission denied"}')

    assert events[-1].type is EventType.BACKEND_FAILED
    assert events[-1].error.kind == "backend_error"
    assert events[-1].error.message == "permission denied"


def test_claude_session_helpers_follow_project_slug_and_text_format(tmp_path):
    slug = _claude_project_slug(tmp_path)
    text = _extract_claude_text(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "tool_use", "name": "Read"},
                    {"type": "text", "text": " world"},
                ]
            },
        }
    )

    assert slug.startswith("-")
    assert tmp_path.name in slug
    assert text == "hello world"


def test_claude_activity_helper_extracts_tool_use_names():
    activity = _extract_claude_activity(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "README.md"}},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "rg --files"}},
                ]
            },
        }
    )

    assert activity == "tools: Read file_path=README.md; Bash command=rg --files"


def test_claude_ui_ready_requires_prompt_and_plan_mode():
    assert _claude_ui_ready("plan mode\n❯ ") is True
    assert _claude_ui_ready("\x1b]0;✳ Claude Haiku\x07") is True
    assert _claude_ui_ready("❯ ") is False


def test_claude_exit_plan_mode_prompt_detection_handles_terminal_controls():
    snapshot = (
        "\x1b]0;✳ Claude Code\x07"
        "\x1b[6A● Ready.\r"
        "\x1b[3B Exit\x1b[1Cplan\x1b[1Cmode?\r"
        "\r\n❯ 1. Yes\r\n  2. No"
    )

    assert _claude_exit_plan_mode_prompt_visible(snapshot) is True
    assert _claude_exit_plan_mode_prompt_visible("plan mode\n❯ ") is False


class FakeClaudeTransport:
    def __init__(self, snapshot: str, *, prompt: str = "", transcript_path=None, start_on_enter: bool = False) -> None:
        self._snapshot = snapshot
        self.prompt = prompt
        self.transcript_path = transcript_path
        self.start_on_enter = start_on_enter
        self.sent: list[str] = []

    def snapshot(self) -> str:
        return self._snapshot

    def send(self, text: str) -> None:
        self.sent.append(text)
        if text == self.prompt:
            self._snapshot = "plan mode\n❯ [Pasted text #1] paste again to expand"
        if text == "\r" and self.start_on_enter and self.transcript_path is not None:
            self.transcript_path.write_text('{"type":"assistant"}\n')

    def is_running(self) -> bool:
        return True


def test_claude_pty_declines_exit_plan_mode_once():
    transport = FakeClaudeTransport("Exit\x1b[1Cplan\x1b[1Cmode?\n❯ 1. Yes\n  2. No")
    session = ClaudePtySession.__new__(ClaudePtySession)
    session.transport = transport
    session._declined_exit_plan_mode = False

    assert session._decline_exit_plan_mode_if_prompted() is True
    assert session._decline_exit_plan_mode_if_prompted() is False
    assert transport.sent == ["2\r"]


def make_claude_pty_session_for_submit_test(tmp_path, *, prompt: str, start_on_enter: bool = False):
    transcript_path = tmp_path / "claude-session.jsonl"
    transport = FakeClaudeTransport(
        "plan mode\n❯ ",
        prompt=prompt,
        transcript_path=transcript_path,
        start_on_enter=start_on_enter,
    )
    session = ClaudePtySession.__new__(ClaudePtySession)
    session.transport = transport
    session.transcript_path = transcript_path
    session.prompt_reflect_timeout = 0.01
    session.submit_start_timeout = 0.01
    session._wait_until_ready = lambda timeout: None
    return session, transport


def test_claude_pty_submit_sends_recovery_enter_for_large_paste_placeholder(tmp_path):
    prompt = "Resume context\n" + ("x" * 1200)
    session, transport = make_claude_pty_session_for_submit_test(tmp_path, prompt=prompt)

    session._submit_prompt(prompt)

    assert transport.sent == [prompt, "\r", "\r"]


def test_claude_pty_submit_does_not_recover_when_first_enter_starts_turn(tmp_path):
    prompt = "Resume context\n" + ("x" * 1200)
    session, transport = make_claude_pty_session_for_submit_test(tmp_path, prompt=prompt, start_on_enter=True)

    session._submit_prompt(prompt)

    assert transport.sent == [prompt, "\r"]


def test_claude_turn_complete_waits_for_text_after_end_turn():
    assert (
        _claude_turn_complete(
            emitted_text="",
            saw_end_turn=True,
            snapshot="plan mode\n❯ ",
            transport_idle=0.4,
            transcript_idle=0.4,
        )
        is False
    )
    assert (
        _claude_turn_complete(
            emitted_text="claude full haiku smoke",
            saw_end_turn=True,
            snapshot="plan mode\n❯ ",
            transport_idle=0.4,
            transcript_idle=0.4,
        )
        is True
    )
