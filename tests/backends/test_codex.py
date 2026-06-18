from pathlib import Path

from ccg_tui.backends.codex import (
    CodexAdapter,
    CodexPtySession,
    _codex_turn_complete,
    _codex_ui_ready,
    _extract_codex_activity,
    _extract_codex_error,
    _extract_codex_message,
    _extract_codex_task_complete_text,
    _is_codex_task_complete,
    _read_codex_session_meta,
)
from ccg_tui.models import EventType


def test_codex_adapter_builds_noninteractive_json_command(tmp_path):
    adapter = CodexAdapter()

    command = adapter.build_command("hello", tmp_path)

    assert command[:4] == ["codex", "exec", "--skip-git-repo-check", "--json"]
    assert command[-1] == "hello"


def test_codex_adapter_forwards_explicit_model_override(tmp_path):
    adapter = CodexAdapter(model="codex-mini-latest")

    command = adapter.build_command("hello", tmp_path)

    assert "--model" in command
    assert command[command.index("--model") + 1] == "codex-mini-latest"


def test_codex_adapter_maps_full_access_to_noninteractive_bypass_flag(tmp_path):
    adapter = CodexAdapter(approval_policy="never", sandbox_mode="danger-full-access")

    command = adapter.build_command("hello", tmp_path)

    assert "--dangerously-bypass-approvals-and-sandbox" in command


def test_codex_adapter_parses_successful_stream():
    adapter = CodexAdapter()
    lines = [
        '{"type":"thread.started","thread_id":"thread-123"}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"hello from codex"}}',
        '{"type":"turn.completed","usage":{"output_tokens":3}}',
    ]

    events = [event for line in lines for event in adapter.parse_stdout_line(line)]

    assert events[0].type is EventType.SESSION_STARTED
    assert events[0].session_id == "thread-123"
    assert events[1].type is EventType.OUTPUT_STARTED
    assert events[2].type is EventType.OUTPUT_DELTA
    assert events[2].text == "hello from codex"
    assert events[3].type is EventType.BACKEND_SUCCEEDED


def test_codex_session_helpers_prefer_final_answer_and_read_meta(tmp_path):
    session_file = tmp_path / "rollout.jsonl"
    session_file.write_text(
        '{"type":"session_meta","payload":{"id":"session-1","cwd":"/tmp/work"}}\n'
        '{"type":"response_item","payload":{"type":"message","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"done"}]}}\n'
    )

    session_id, cwd = _read_codex_session_meta(session_file)
    text, phase = _extract_codex_message(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "done"}],
            },
        }
    )

    assert session_id == "session-1"
    assert cwd == "/tmp/work"
    assert text == "done"
    assert phase == "final_answer"


def test_codex_task_complete_helpers_extract_final_text():
    record = {
        "type": "event_msg",
        "payload": {
            "type": "task_complete",
            "last_agent_message": "finished from task complete",
        },
    }

    assert _is_codex_task_complete(record) is True
    assert _extract_codex_task_complete_text(record) == "finished from task complete"


def test_codex_activity_helper_extracts_commentary_and_tool_calls():
    commentary = {
        "type": "event_msg",
        "payload": {
            "type": "agent_message",
            "message": "I am checking the README before editing.",
            "phase": "commentary",
        },
    }
    tool_call = {
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "exec_command",
            "arguments": '{"cmd":"rg --files","workdir":"/tmp/project"}',
            "call_id": "call-1",
        },
    }

    assert _extract_codex_activity(commentary) == "I am checking the README before editing."
    assert _extract_codex_activity(tool_call) == "tool: exec_command cmd=rg --files workdir=/tmp/project"


def test_codex_ui_ready_requires_prompt_markers():
    ready_snapshot = "Use /skills to list available skills\n› "
    working_snapshot = "Working on it...\n"

    assert _codex_ui_ready(ready_snapshot) is True
    assert _codex_ui_ready(working_snapshot) is False


def test_codex_turn_complete_waits_for_task_complete_or_ready_prompt():
    assert (
        _codex_turn_complete(
            emitted_text="Inspecting files",
            saw_task_complete=False,
            snapshot="still working",
            transport_idle=1.2,
            rollout_idle=1.2,
        )
        is False
    )
    assert (
        _codex_turn_complete(
            emitted_text="Final answer",
            saw_task_complete=True,
            snapshot="still working",
            transport_idle=0.0,
            rollout_idle=0.0,
        )
        is True
    )
    assert (
        _codex_turn_complete(
            emitted_text="Final answer",
            saw_task_complete=False,
            snapshot="Use /skills to list available skills\n› ",
            transport_idle=0.4,
            rollout_idle=0.5,
        )
        is True
    )


def test_codex_adapter_reports_nonzero_exit_as_failure():
    adapter = CodexAdapter()

    events = adapter.completion_events(exit_code=2, stderr="boom", saw_output=False)

    assert events[-1].type is EventType.BACKEND_FAILED
    assert events[-1].error.kind == "process_exit"
    assert events[-1].error.message == "boom"
    assert events[-1].error.exit_code == 2


def test_codex_adapter_parses_noninteractive_tool_activity():
    adapter = CodexAdapter()
    line = '{"type":"item.started","item":{"type":"command_execution","command":"rg --files"}}'

    events = adapter.parse_stdout_line(line)

    assert events[0].type is EventType.ACTIVITY
    assert events[0].text == "tool: rg --files"
    assert events[0].activity["kind"] == "tool"
    assert events[0].activity["status"] == "started"


def test_codex_error_helper_reads_event_msg_failures():
    error = _extract_codex_error(
        {
            "type": "event_msg",
            "payload": {
                "type": "error",
                "message": "You've hit your usage limit.",
                "codex_error_info": "usage_limit_exceeded",
            },
        }
    )

    assert error is not None
    assert error.kind == "rate_limit"
    assert error.message == "You've hit your usage limit."
    assert error.details["codex_error_info"] == "usage_limit_exceeded"
    assert error.details["original_kind"] == "usage_limit_exceeded"


class FakeCodexTransport:
    def __init__(self, *, start_on_first_enter: bool) -> None:
        self.start_on_first_enter = start_on_first_enter
        self.sent: list[str] = []
        self._snapshot = "› "
        self._enter_count = 0

    def send(self, text: str) -> None:
        self.sent.append(text)
        if text == "\r":
            self._enter_count += 1
            if self.start_on_first_enter or self._enter_count >= 2:
                self._snapshot = "Working"
            return
        self._snapshot = "› [Pasted Content 1024 chars] Ready"

    def snapshot(self) -> str:
        return self._snapshot

    def is_running(self) -> bool:
        return True

    def idle_for(self) -> float:
        return 0.5


def make_codex_pty_session_for_submit_test(transport: FakeCodexTransport) -> CodexPtySession:
    session = CodexPtySession.__new__(CodexPtySession)
    session.transport = transport
    session.tail = None
    session.session_path = None
    session.prompt_reflect_timeout = 0.0
    session.submit_start_timeout = 0.0
    session._prepare_startup = lambda: None
    session._locate_session_file = lambda *, timeout=0.0: None
    return session


def test_codex_pty_submit_sends_recovery_enter_for_large_paste_placeholder():
    transport = FakeCodexTransport(start_on_first_enter=False)
    session = make_codex_pty_session_for_submit_test(transport)

    session._submit_prompt("x" * 1200)

    assert transport.sent == ["x" * 1200, "\r", "\r"]


def test_codex_pty_submit_does_not_recover_when_first_enter_starts_turn():
    transport = FakeCodexTransport(start_on_first_enter=True)
    session = make_codex_pty_session_for_submit_test(transport)

    session._submit_prompt("x" * 1200)

    assert transport.sent == ["x" * 1200, "\r"]
