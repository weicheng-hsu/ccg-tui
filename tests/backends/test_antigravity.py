from __future__ import annotations

from ccg_tui.backends import build_backend
from ccg_tui.backends.antigravity import (
    AntigravityAdapter,
    antigravity_model_options,
    current_antigravity_model,
    discover_antigravity_model_options,
    parse_antigravity_model_options,
    set_antigravity_model,
)
from ccg_tui.models import BackendName
from ccg_tui.models import EventType


def test_build_backend_wires_antigravity_adapter():
    adapter = build_backend("antigravity", model="configured", permission_mode="sandbox")

    assert adapter.name is BackendName.ANTIGRAVITY
    assert getattr(adapter, "model") == "configured"
    assert getattr(adapter, "permission_mode") == "sandbox"


def test_antigravity_adapter_builds_print_command(tmp_path):
    adapter = AntigravityAdapter()

    command = adapter.build_command("hello", tmp_path)

    assert command == ["agy", "--print", "hello", "--print-timeout", "5m0s"]


def test_antigravity_adapter_maps_sandbox_permission_to_flag(tmp_path):
    adapter = AntigravityAdapter(permission_mode="sandbox")

    command = adapter.build_command("hello", tmp_path)

    assert command[:3] == ["agy", "--sandbox", "--print"]


def test_antigravity_adapter_maps_full_access_permission_to_skip_flag(tmp_path):
    adapter = AntigravityAdapter(permission_mode="dangerously-skip-permissions")

    command = adapter.build_command("hello", tmp_path)

    assert command[:3] == ["agy", "--dangerously-skip-permissions", "--print"]


def test_antigravity_adapter_stores_model_without_unverified_model_flag(tmp_path):
    adapter = AntigravityAdapter(model="gemini-3.5-flash")

    command = adapter.build_command("hello", tmp_path)

    assert adapter.model == "gemini-3.5-flash"
    assert "--model" not in command


def test_antigravity_model_setting_is_read_and_written(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"colorScheme":"terminal","model":"Gemini 3.5 Flash (Medium)"}')

    selected = set_antigravity_model(
        "Gemini 3.1 Pro (High)",
        settings_path,
        available_models=("Gemini 3.1 Pro (High)",),
    )

    assert selected == "Gemini 3.1 Pro (High)"
    assert current_antigravity_model(settings_path) == "Gemini 3.1 Pro (High)"


def test_antigravity_model_options_use_env_override(monkeypatch):
    monkeypatch.setenv(
        "CCG_TUI_ANTIGRAVITY_MODEL_OPTIONS",
        '["Dynamic Model A", "Future Model Z", "Dynamic Model A"]',
    )

    assert antigravity_model_options() == ("Dynamic Model A", "Future Model Z")


def test_parse_antigravity_model_options_reads_native_picker_rows():
    text = """
    Switch Model
      ↓ 29 more
    > Gemini 3.5 Flash (Low)       (current)
      Future Model Z (Experimental)
    > Claude Sonnet 4.6 (Thinking)
      GPT-OSS 120B (Medium)
    Keyboard: up/down
    esc to cancel
    """

    assert parse_antigravity_model_options(text) == (
        "Gemini 3.5 Flash (Low)",
        "Future Model Z (Experimental)",
        "Claude Sonnet 4.6 (Thinking)",
        "GPT-OSS 120B (Medium)",
    )


def test_antigravity_model_setting_accepts_native_claude_model(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")

    selected = set_antigravity_model(
        "Claude Sonnet 4.6 (Thinking)",
        settings_path,
        available_models=("Claude Sonnet 4.6 (Thinking)",),
    )

    assert selected == "Claude Sonnet 4.6 (Thinking)"
    assert current_antigravity_model(settings_path) == "Claude Sonnet 4.6 (Thinking)"


def test_antigravity_model_setting_rejects_unknown_model(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{}")

    try:
        set_antigravity_model("gemini-unknown", settings_path, available_models=("Dynamic Model A",))
    except ValueError as exc:
        assert "Unsupported Antigravity model" in str(exc)
        assert "Dynamic Model A" in str(exc)
    else:
        raise AssertionError("expected unsupported Antigravity model to fail")


def test_discover_antigravity_model_options_drives_native_picker(monkeypatch, tmp_path):
    sent: list[str] = []

    class FakeTransport:
        def __init__(self, command, cwd, *, env, max_buffer_chars):
            assert command == ["agy"]
            assert cwd == tmp_path
            assert env["AGY_CLI_HIDE_ACCOUNT_INFO"] == "1"
            assert max_buffer_chars > 100_000
            self.snapshot_text = "> ? for shortcuts"
            self.closed = False

        def is_running(self):
            return True

        def snapshot(self):
            return self.snapshot_text

        def idle_for(self):
            return 1.0

        def send(self, text):
            sent.append(text)
            if text == "/model\r":
                self.snapshot_text += "\n> Dynamic Model A\n  Future Model Z"
            elif text == "\x1b[B":
                self.snapshot_text += "\n> Dynamic Model B"

        def close(self):
            self.closed = True

    monkeypatch.setattr("ccg_tui.backends.antigravity.shutil.which", lambda executable: "/usr/bin/agy")
    monkeypatch.setattr("ccg_tui.backends.antigravity.PtyProcess", FakeTransport)

    options = discover_antigravity_model_options(cwd=tmp_path, max_steps=1, step_delay=0)

    assert options == ("Dynamic Model A", "Future Model Z", "Dynamic Model B")
    assert sent[:2] == ["/model\r", "\x1b[B"]


def test_antigravity_interactive_prompt_is_sent_after_native_start():
    class FakeTransport:
        def __init__(self):
            self.sent = []
            self.quiet_waits = []

        def is_running(self):
            return True

        def snapshot(self):
            return "> ? for shortcuts Gemini"

        def idle_for(self):
            return 1.0

        def send(self, text):
            self.sent.append(text)

        def wait_for_quiet(self, *, idle_for, timeout):
            self.quiet_waits.append((idle_for, timeout))
            return True

    transport = FakeTransport()

    AntigravityAdapter()._send_interactive_prompt(transport, "/model")

    assert transport.sent == ["/model", "\r"]
    assert transport.quiet_waits == [(0.5, 5.0)]


def test_antigravity_adapter_parses_plain_print_output():
    events = AntigravityAdapter().parse_stdout_line("hello from agy")

    assert [event.type for event in events] == [EventType.OUTPUT_STARTED, EventType.OUTPUT_DELTA]
    assert events[-1].text == "hello from agy\n"


def test_antigravity_adapter_parses_json_model_transcript_line():
    events = AntigravityAdapter().parse_stdout_line(
        '{"step_index":2,"source":"MODEL","type":"PLANNER_RESPONSE","status":"DONE","content":"done"}'
    )

    assert [event.type for event in events] == [EventType.OUTPUT_STARTED, EventType.OUTPUT_DELTA]
    assert events[-1].text == "done"


def test_antigravity_adapter_parses_generic_tool_activity():
    events = AntigravityAdapter().parse_stdout_line(
        '{"source":"TOOL","type":"TOOL_CALL","status":"STARTED","tool_name":"read_file","input":{"path":"README.md"}}'
    )

    assert [event.type for event in events] == [EventType.ACTIVITY]
    assert events[0].text == "tool: read_file"
    assert events[0].activity is not None
    assert events[0].activity["details"]["name"] == "read_file"


def test_antigravity_completion_succeeds_after_output():
    events = AntigravityAdapter().completion_events(exit_code=0, stderr="", saw_output=True)

    assert [event.type for event in events] == [EventType.BACKEND_SUCCEEDED]


def test_antigravity_completion_flags_empty_success_as_clear_failure():
    events = AntigravityAdapter().completion_events(exit_code=0, stderr="", saw_output=False)

    assert events[-1].type is EventType.BACKEND_FAILED
    assert events[-1].error is not None
    assert events[-1].error.kind == "backend_error"
    assert "no stdout" in events[-1].error.message


def test_antigravity_completion_normalizes_auth_failure():
    events = AntigravityAdapter().completion_events(exit_code=1, stderr="Please sign in to continue", saw_output=False)

    assert events[-1].type is EventType.BACKEND_FAILED
    assert events[-1].error is not None
    assert events[-1].error.kind == "auth_error"


def test_antigravity_run_reports_missing_executable_as_backend_failure(tmp_path):
    events = list(AntigravityAdapter(executable="missing-agy-for-test").run("hello", tmp_path))

    assert [event.type for event in events] == [EventType.BACKEND_FAILED]
    assert events[0].error is not None
    assert events[0].error.kind == "backend_error"
    assert "executable not found" in events[0].error.message
