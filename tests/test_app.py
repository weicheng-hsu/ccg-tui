import json
from types import SimpleNamespace

import pytest

from ccg_tui.handoff import build_handoff_packet
from ccg_tui.app import (
    ANSI_RESET,
    choose_backend,
    handle_task_command,
    build_backend_picker_lines,
    default_controller_factory,
    build_header_line,
    build_handoff_preview,
    build_sidebar_text,
    build_status_text,
    build_transcript_text,
    build_summary_backend,
    colorize_backend_label,
    apply_model_selection,
    format_capability_registry,
    format_handoff_execution_confirmation,
    current_permission_label,
    current_permission_values,
    current_model_label,
    build_packet_routing_decision,
    format_session_list,
    format_model_options,
    format_permission_options,
    format_product_status,
    format_resume_context_preview,
    format_markdown_lines,
    format_turn_summary,
    handoff_status_message,
    parse_handoff_args,
    install_prompt_toolkit_shift_enter_sequences,
    is_prompt_toolkit_shift_enter_event,
    latest_activity_text,
    main,
    model_options_for_backend,
    normalize_backend_choice,
    phase_label,
    progress_message,
    render_interface_screen,
    resume_context_status_message,
    run_interface,
    SHIFT_ENTER_SEQUENCES,
    spinner_frame,
    stalled_message,
    task_status_message,
    transcript_turn_separator,
    turn_is_busy,
    turn_meta_lines,
)
from ccg_tui.models import BackendEvent, BackendName, EventType, NormalizedError, RecordedEvent, SessionRecord, SummaryRecord, TaskRecord, TurnRecord, TurnStatus
from ccg_tui.transcript import TranscriptStore


class FakeController:
    def __init__(self, backend: str):
        self.session = SimpleNamespace(
            id="session-1",
            backend=BackendName(backend),
            turns=[],
            summaries=[],
            routing_decisions=[],
            workspace_cwd="/tmp/workspace",
            tasks=[
                TaskRecord(
                    id="task-main",
                    created_at="2026-04-21T15:05:00+00:00",
                    updated_at="2026-04-21T15:05:00+00:00",
                )
            ],
        )
        self.adapter = SimpleNamespace(model=None)
        if backend == "codex":
            self.adapter.approval_policy = "on-request"
            self.adapter.sandbox_mode = "workspace-write"
        elif backend == "claude":
            self.adapter.permission_mode = "default"
        elif backend == "gemini":
            self.adapter.approval_mode = "default"
        elif backend == "antigravity":
            self.adapter.permission_mode = "default"
        self.prompts = []
        self.closed = False

    def submit_prompt(self, prompt: str):
        self.prompts.append(prompt)
        task = self.prompt_task()
        turn = TurnRecord(
            id=f"turn-{len(self.prompts)}",
            backend=self.session.backend,
            prompt=prompt,
            output=f"reply to {prompt}",
            status=TurnStatus.COMPLETED,
            started_at="2026-04-21T15:05:01+00:00",
            completed_at="2026-04-21T15:05:02+00:00",
            task_id=task.id,
        )
        self.session.turns.append(turn)
        task.turn_ids.append(turn.id)
        task.updated_at = turn.completed_at or turn.started_at
        if task.start_turn_id is None:
            task.start_turn_id = turn.id
        return turn

    def close(self):
        self.closed = True

    def attach_backend(self, adapter):
        self.adapter = adapter

    def main_task(self):
        return self.session.tasks[0]

    def active_user_task(self):
        return next(
            (task for task in reversed(self.session.tasks) if task.id != "task-main" and task.status == "active"),
            None,
        )

    def prompt_task(self):
        return self.active_user_task() or self.main_task()

    def latest_closed_task(self):
        return next((task for task in reversed(self.session.tasks) if task.status == "closed"), None)

    def start_task(self, title: str | None = None):
        if self.active_user_task() is not None:
            raise ValueError(f"Task already active: {self.active_user_task().id}")
        task = TaskRecord(
            id=f"task-{len(self.session.tasks)}",
            created_at="2026-04-21T15:05:00+00:00",
            updated_at="2026-04-21T15:05:00+00:00",
            kind="task",
            title=title or None,
        )
        self.session.tasks.append(task)
        return task

    def close_task(self, closing_note: str | None = None):
        task = self.active_user_task()
        if task is None:
            raise ValueError("No active task to close")
        task.status = "closed"
        task.closing_note = closing_note or None
        task.end_turn_id = task.turn_ids[-1] if task.turn_ids else None
        return task

    def record_routing_decision(self, **kwargs):
        from ccg_tui.models import RoutingDecisionRecord

        kwargs.setdefault("active_backend", self.session.backend)
        decision = RoutingDecisionRecord(
            id=f"routing-{len(self.session.routing_decisions) + 1}",
            recorded_at="2026-04-21T15:05:03+00:00",
            **kwargs,
        )
        self.session.routing_decisions.append(decision)
        return decision


class FakePreviewController(FakeController):
    def preview_resume_context(self):
        return SimpleNamespace(
            metadata={
                "injected_summary_id": "summary-1",
                "injected_turn_ids": ["turn-1", "turn-2"],
                "context_char_count": 123,
            },
            context_text="CCG LOCAL RESUME CONTEXT\nsummary text",
        )


class FakeCliAdapter:
    name = BackendName.CODEX

    def __init__(self):
        self.closed = False
        self.prompts = []

    def run(self, prompt, cwd):
        self.prompts.append(prompt)
        yield BackendEvent(type=EventType.OUTPUT_STARTED)
        yield BackendEvent(type=EventType.OUTPUT_DELTA, text="cli reply")
        yield BackendEvent(type=EventType.BACKEND_SUCCEEDED)

    def close(self):
        self.closed = True


class FakeModelCliAdapter(FakeCliAdapter):
    def __init__(self, name=BackendName.CODEX, model=None):
        super().__init__()
        self.name = name
        self.model = model


class MissingFileCliAdapter:
    name = BackendName.CODEX

    def run(self, prompt, cwd):
        raise FileNotFoundError("missing backend executable")
        yield

    def close(self):
        return None


def test_format_turn_summary_uses_consistent_shape():
    turn = TurnRecord(
        id="turn-1",
        backend=BackendName.CLAUDE,
        prompt="hello",
        output="world",
        status=TurnStatus.COMPLETED,
        started_at="2026-04-21T15:05:01+00:00",
        completed_at="2026-04-21T15:05:02+00:00",
    )

    text = format_turn_summary(turn)

    assert "Backend : claude" in text
    assert "Status  : completed" in text
    assert "Prompt  : hello" in text
    assert "Output  : world" in text


def test_format_turn_summary_includes_error_details_for_failed_turns():
    turn = TurnRecord(
        id="turn-2",
        backend=BackendName.GEMINI,
        prompt="hello",
        output="",
        status=TurnStatus.FAILED,
        started_at="2026-04-21T15:05:01+00:00",
        completed_at="2026-04-21T15:05:02+00:00",
        error=NormalizedError(kind="auth_error", message="missing auth"),
    )

    text = format_turn_summary(turn)

    assert "Status  : failed" in text
    assert "Recovery: failed; error=auth_error" in text
    assert "Error   : missing auth" in text


def test_format_turn_summary_surfaces_interrupted_recovery_metadata():
    turn = TurnRecord(
        id="turn-interrupted",
        backend=BackendName.CODEX,
        prompt="hello",
        output="partial output",
        status=TurnStatus.FAILED,
        started_at="2026-04-21T15:05:01+00:00",
        completed_at="2026-04-21T15:05:02+00:00",
        error=NormalizedError(kind="interrupted", message="no terminal event"),
        metadata={"recovery": {"state": "interrupted", "terminal_event_seen": False, "partial_output": True}},
    )

    text = format_turn_summary(turn)

    assert "Recovery: interrupted; no terminal event; partial output; error=interrupted" in text
    assert "Error   : no terminal event" in text


def test_choose_backend_retries_until_valid_choice():
    answers = iter(["5", "claude"])
    printed: list[str] = []

    backend = choose_backend(input_fn=lambda _: next(answers), print_fn=printed.append)

    assert backend == "claude"
    assert any("Select backend" in line for line in printed)
    assert any("Invalid selection" in line for line in printed)


def test_normalize_backend_choice_accepts_indices_and_names():
    assert normalize_backend_choice("1") == "codex"
    assert normalize_backend_choice(" 2 ") == "claude"
    assert normalize_backend_choice("gemini") == "gemini"
    assert normalize_backend_choice("4") == "antigravity"
    assert normalize_backend_choice("antigravity") == "antigravity"
    assert normalize_backend_choice("unknown") is None


def test_build_summary_backend_supports_gemini_and_antigravity():
    assert build_summary_backend("gemini").name is BackendName.GEMINI
    assert build_summary_backend("antigravity").name is BackendName.ANTIGRAVITY
    try:
        build_summary_backend("codex")
    except ValueError as exc:
        assert "gemini" in str(exc).lower()
        assert "antigravity" in str(exc).lower()
    else:
        raise AssertionError("expected unsupported summary backend to be rejected")


def test_build_backend_picker_lines_lists_choices_and_shortcuts():
    lines = build_backend_picker_lines(selected_backend="claude")
    text = "\n".join(lines)

    assert "Select a backend" in text
    assert "1. codex" in text
    assert "2. claude" in text
    assert "3. gemini" in text
    assert "4. antigravity" in text
    assert "Enter choose backend" in text
    assert "Esc quit" in text
    assert "> claude" in text


def test_build_header_line_includes_backend_session_and_turns():
    controller = FakeController("codex")
    controller.submit_prompt("hello")

    line = build_header_line(controller)

    assert "CCG TUI" in line
    assert "[codex]" in line
    assert "Session: session-1" in line
    assert "Turns: 1" in line
    assert "Workspace:" in line


def test_build_sidebar_text_includes_backend_and_commands():
    controller = FakeController("gemini")
    controller.submit_prompt("hello")
    controller.session.vendor_session_id = "vendor-7"

    text = build_sidebar_text(controller, is_busy=True, draft_text="draft line\nsecond")

    assert "Session Info" in text
    assert "Backend : gemini" in text
    assert "Vendor  : vendor-7" in text
    assert "State   : busy" in text
    assert "Draft   : 2 lines / 17 chars" in text
    assert "Model   : default" in text
    assert "Perms   : Ask before actions" in text
    assert "Task    : task-main (default)" in text
    assert "Resume  : yes" in text
    assert "Route   : advisory only; /capabilities for registry" in text
    assert "Enter   submit prompt" in text
    assert "S-Enter newline" in text
    assert "C-J     submit fallback" in text
    assert "Esc+Ret newline fallback" in text


def test_sidebar_status_and_transcript_surface_recovery_state():
    controller = FakeController("codex")
    controller.session.turns.append(
        TurnRecord(
            id="turn-interrupted",
            backend=BackendName.CODEX,
            prompt="keep going",
            output="partial",
            status=TurnStatus.FAILED,
            started_at="2026-04-21T15:05:01+00:00",
            completed_at="2026-04-21T15:05:02+00:00",
            error=NormalizedError(kind="interrupted", message="no terminal event"),
            metadata={"recovery": {"state": "interrupted", "terminal_event_seen": False, "partial_output": True}},
        )
    )

    sidebar = build_sidebar_text(controller)
    footer = build_status_text(controller)
    transcript = build_transcript_text(controller)

    assert "Last    : interrupted" in sidebar
    assert "Recovery: interrupted; no terminal event; partial output; error=interrupted" in sidebar
    assert "recovery=interrupted, no terminal event, partial output, error=interrupted" in footer
    assert "recovery> interrupted; no terminal event; partial output; error=interrupted" in transcript
    assert "error  > no terminal event" in transcript


def test_format_product_status_includes_continuation_metadata():
    controller = FakeController("claude")
    controller.session.turns.append(
        TurnRecord(
            id="turn-1",
            backend=BackendName.CLAUDE,
            prompt="keep going",
            output="partial",
            status=TurnStatus.STREAMING,
            started_at="2026-04-21T15:05:01+00:00",
            metadata={"recovery": {"state": "interrupted", "partial_output": True}},
        )
    )

    text = format_product_status(controller)

    assert "Turns   : 1" in text
    assert "Last    : interrupted" in text
    assert "Recovery: interrupted; no terminal event; partial output" in text
    assert "Resume  : yes" in text
    assert "Task    : task-main (default)" in text
    assert "Route   : advisory only; /capabilities for registry" in text


def test_format_product_status_surfaces_latest_closed_task():
    controller = FakeController("claude")
    handle_task_command(controller, "start Fix status")
    controller.submit_prompt("work")
    handle_task_command(controller, "close done")

    text = format_product_status(controller)

    assert "Task    : task-main (default); latest closed: Fix status (task-1)" in text


def test_task_status_message_reports_active_and_closed_tasks():
    controller = FakeController("codex")

    assert task_status_message(controller) == "No active task. Prompts default to task-main."

    handle_task_command(controller, "start Fix slash routing")

    assert "Active task: Fix slash routing (task-1) turns=0" == task_status_message(controller)

    controller.submit_prompt("work on it")
    handle_task_command(controller, "close done")

    assert (
        "No active task. Latest closed: Fix slash routing (task-1) turns=1. Prompts default to task-main."
        == task_status_message(controller)
    )


def test_handle_task_command_routes_locally():
    controller = FakeController("claude")

    ok, message = handle_task_command(controller, "start Investigate resume bug")
    assert ok is True
    assert "Started task: Investigate resume bug (task-1)" == message

    ok, message = handle_task_command(controller, "status")
    assert ok is True
    assert "Active task: Investigate resume bug (task-1) turns=0" == message

    ok, message = handle_task_command(controller, "close resolved")
    assert ok is True
    assert "Closed task: Investigate resume bug (task-1) with note." == message


@pytest.mark.parametrize("sequence", SHIFT_ENTER_SEQUENCES)
def test_prompt_toolkit_shift_enter_sequences_are_detectable(sequence):
    from prompt_toolkit.input.vt100_parser import Vt100Parser
    from prompt_toolkit.keys import Keys

    key_presses = []
    install_prompt_toolkit_shift_enter_sequences()
    parser = Vt100Parser(key_presses.append)
    parser.feed_and_flush(sequence)

    assert [(key_press.key, key_press.data) for key_press in key_presses] == [(Keys.ControlM, sequence)]
    event = SimpleNamespace(key_sequence=key_presses)
    assert is_prompt_toolkit_shift_enter_event(event) is True


def test_prompt_toolkit_plain_enter_is_not_shift_enter():
    from prompt_toolkit.input.vt100_parser import Vt100Parser
    from prompt_toolkit.keys import Keys

    key_presses = []
    install_prompt_toolkit_shift_enter_sequences()
    parser = Vt100Parser(key_presses.append)
    parser.feed_and_flush("\r")

    assert [(key_press.key, key_press.data) for key_press in key_presses] == [(Keys.ControlM, "\r")]
    event = SimpleNamespace(key_sequence=key_presses)
    assert is_prompt_toolkit_shift_enter_event(event) is False


def test_format_model_options_marks_current_model(monkeypatch):
    monkeypatch.setenv(
        "CCG_TUI_GEMINI_MODEL_OPTIONS",
        json.dumps([{"value": "gemini-2.5-flash", "label": "Gemini 2.5 Flash"}]),
    )
    controller = FakeController("gemini")
    controller.adapter.model = "gemini-2.5-flash"

    text = format_model_options(controller, selected_index=2)

    assert "Model Picker (gemini)" in text
    assert "Current: gemini-2.5-flash" in text
    assert "Gemini 2.5 Flash" in text
    assert "*" in text


def test_format_model_options_lists_current_gemini_cli_visible_models(monkeypatch):
    monkeypatch.setenv(
        "CCG_TUI_GEMINI_MODEL_OPTIONS",
        json.dumps(
            [
                {"value": "auto-gemini-3", "label": "Auto (Gemini 3)"},
                {"value": "gemini-3-pro-preview", "label": "Gemini 3 Pro Preview"},
                {"value": "gemini-3-flash-preview", "label": "Gemini 3 Flash Preview"},
                {"value": "gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash Lite"},
            ]
        ),
    )
    text = format_model_options(FakeController("gemini"))

    assert "Auto (Gemini 3)" in text
    assert "auto-gemini-3" in text
    assert "Gemini 3 Pro Preview" in text
    assert "gemini-3-flash-preview" in text
    assert "Gemini 2.5 Flash Lite" in text
    assert "gemini-2.0-flash" not in text


def test_model_options_for_backend_reads_codex_models_cache(tmp_path, monkeypatch):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "hidden-model",
                        "display_name": "Hidden",
                        "description": "Should not be shown.",
                        "visibility": "hide",
                        "priority": 0,
                    },
                    {
                        "slug": "second-model",
                        "display_name": "Second Model",
                        "description": "Lower priority model.",
                        "visibility": "list",
                        "priority": 20,
                    },
                    {
                        "slug": "first-model",
                        "display_name": "First Model",
                        "description": "Higher priority model.",
                        "visibility": "list",
                        "priority": 10,
                    },
                ]
            }
        )
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    options = model_options_for_backend("codex")

    assert [option.value for option in options] == [None, "first-model", "second-model"]
    assert options[1].label == "First Model"
    assert options[1].description == "Higher priority model."


def test_model_options_for_backend_refreshes_codex_debug_catalog(monkeypatch):
    payload = {
        "models": [
            {
                "slug": "codex-dynamic-model",
                "display_name": "Codex Dynamic",
                "description": "Loaded from live catalog.",
                "visibility": "list",
                "priority": 1,
            }
        ]
    }

    monkeypatch.setattr("ccg_tui.app.shutil.which", lambda executable: "/usr/bin/codex")
    monkeypatch.setattr(
        "ccg_tui.app.subprocess.run",
        lambda *_, **__: SimpleNamespace(returncode=0, stdout=json.dumps(payload)),
    )

    options = model_options_for_backend("codex", refresh=True)

    assert [option.value for option in options] == [None, "codex-dynamic-model"]
    assert options[1].label == "Codex Dynamic"


def test_model_options_for_backend_uses_claude_aliases_and_custom_env(monkeypatch):
    monkeypatch.setattr(
        "ccg_tui.app._claude_model_values_from_binary",
        lambda: ("sonnet", "opus", "haiku", "sonnet[1m]", "opus[1m]", "opusplan", "claude-sonnet-4-6"),
    )
    monkeypatch.setattr("ccg_tui.app._claude_model_values_from_config", lambda: ())
    monkeypatch.setenv("ANTHROPIC_CUSTOM_MODEL_OPTION", "custom-claude-model")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_MODEL_OPTION_NAME", "Custom Claude")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION", "Org-specific Claude deployment.")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL_NAME", "Team Sonnet")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL_DESCRIPTION", "Team Sonnet routing alias.")

    options = model_options_for_backend("claude", refresh=True)

    assert [option.value for option in options[:3]] == [None, "custom-claude-model", "sonnet"]
    assert options[1].label == "Custom Claude"
    assert options[1].description == "Org-specific Claude deployment."
    assert options[2].label == "Team Sonnet"
    assert options[2].description == "Team Sonnet routing alias."
    assert any(option.value == "opus[1m]" for option in options)
    assert all(option.value != "claude-sonnet-4-5-20250929" for option in options)


def test_model_options_for_backend_uses_claude_dynamic_env_override(monkeypatch):
    monkeypatch.setenv(
        "CCG_TUI_CLAUDE_MODEL_OPTIONS",
        json.dumps(
            [
                {
                    "value": "claude-future-model",
                    "label": "Claude Future",
                    "description": "Loaded dynamically.",
                }
            ]
        ),
    )

    options = model_options_for_backend("claude", refresh=True)

    assert [option.value for option in options] == [None, "claude-future-model"]
    assert options[1].label == "Claude Future"


def test_model_options_for_backend_reads_gemini_cli_model_definitions(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    gemini_bin = bundle_dir / "gemini.js"
    gemini_bin.write_text("")
    (bundle_dir / "chunk.js").write_text(
        """
var DEFAULT_MODEL_CONFIGS = {
  modelDefinitions: {
    "gemini-future-pro": {
      displayName: "Gemini Future Pro",
      isVisible: true,
      dialogDescription: "Loaded from installed Gemini CLI."
    },
    "gemini-hidden": {
      isVisible: false
    },
    "auto-gemini-future": {
      displayName: "Auto (Gemini Future)",
      isVisible: true,
      dialogDescription: "Dynamic auto router."
    }
  },
  modelIdResolutions: {}
}
"""
    )
    monkeypatch.setattr("ccg_tui.app.shutil.which", lambda executable: str(gemini_bin))
    monkeypatch.setattr("ccg_tui.app._gemini_model_options_from_settings", lambda: ())

    options = model_options_for_backend("gemini", refresh=True)

    assert [option.value for option in options] == [None, "auto-gemini-future", "gemini-future-pro"]
    assert options[1].description == "Dynamic auto router."
    assert all(option.value != "gemini-hidden" for option in options)


def test_model_options_for_backend_refreshes_antigravity_provider(monkeypatch):
    calls: list[bool] = []

    def fake_antigravity_model_options(*, refresh=False):
        calls.append(refresh)
        return ("Dynamic Antigravity Model",)

    monkeypatch.setattr("ccg_tui.app.antigravity_model_options", fake_antigravity_model_options)

    options = model_options_for_backend("antigravity", refresh=True)

    assert calls == [True]
    assert [option.value for option in options] == ["Dynamic Antigravity Model"]


def test_current_model_label_defaults_when_adapter_has_no_model():
    assert current_model_label(FakeController("codex")) == "default"


def test_antigravity_model_selection_stays_in_ccg_and_writes_native_setting(tmp_path, monkeypatch):
    monkeypatch.setattr("ccg_tui.backends.antigravity.Path.home", lambda: tmp_path)
    monkeypatch.setattr(
        "ccg_tui.app.antigravity_model_options",
        lambda **_: (
            "Gemini 3.1 Pro (High)",
            "Claude Sonnet 4.6 (Thinking)",
            "Claude Opus 4.6 (Thinking)",
        ),
    )
    settings_path = tmp_path / ".gemini" / "antigravity-cli" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text('{"model":"Gemini 3.5 Flash (Medium)"}')
    controller = FakeController("antigravity")

    text = format_model_options(controller)
    message = apply_model_selection(controller, "Gemini 3.1 Pro (High)")

    assert "Model Picker (antigravity)" in text
    assert "Gemini 3.1 Pro (High)" in text
    assert "Claude Sonnet 4.6 (Thinking)" in text
    assert "Claude Opus 4.6 (Thinking)" in text
    assert message == "Model set to Gemini 3.1 Pro (High) for antigravity."
    assert current_model_label(controller) == "Gemini 3.1 Pro (High)"
    assert "Gemini 3.1 Pro (High)" in settings_path.read_text()


def test_format_permission_options_marks_current_permissions():
    controller = FakeController("codex")

    text = format_permission_options(controller, selected_index=1)

    assert "Permissions Picker (codex)" in text
    assert "Current: Ask before actions" in text
    assert "approval_policy=on-request, sandbox_mode=workspace-write" in text
    assert "Ask before actions" in text
    assert "*" in text


def test_current_permission_label_matches_backend_values():
    assert current_permission_label(FakeController("codex")) == "Ask before actions"
    assert current_permission_label(FakeController("claude")) == "Ask before actions"
    assert current_permission_label(FakeController("gemini")) == "Ask before actions"
    assert current_permission_label(FakeController("antigravity")) == "Ask before actions"


@pytest.mark.parametrize(
    ("backend", "expected_values"),
    [
        ("codex", {"approval_policy": "never", "sandbox_mode": "danger-full-access"}),
        ("claude", {"permission_mode": "dangerously-skip-permissions"}),
        ("gemini", {"approval_mode": "yolo"}),
        ("antigravity", {"permission_mode": "dangerously-skip-permissions"}),
    ],
)
def test_default_controller_factory_uses_full_access_permissions(tmp_path, backend, expected_values):
    controller = default_controller_factory("transcripts", tmp_path)(backend)

    try:
        assert current_permission_label(controller) == "Full access"
        assert current_permission_values(controller) == expected_values
    finally:
        controller.close()


def test_format_capability_registry_is_advisory_and_covers_backends():
    controller = FakeController("codex")
    original_backend = controller.session.backend
    original_adapter = controller.adapter

    text = format_capability_registry(controller)

    assert "Routing Capability Registry" in text
    assert "advisory only" in text
    assert "Backend : codex" in text
    assert "Backend : claude" in text
    assert "Backend : gemini" in text
    assert "Backend : antigravity" in text
    assert "Model flag support: yes" in text
    assert "Summary suitability: no" in text
    assert "Summary suitability: yes" in text
    assert "confirmation required" in text
    assert controller.session.backend is original_backend
    assert controller.adapter is original_adapter


def test_build_packet_routing_decision_records_permission_compatibility():
    session = SessionRecord(
        id="session-route-packet",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
    )
    packet = build_handoff_packet(session, target_backend=BackendName.CLAUDE, target_model="sonnet")

    decision = build_packet_routing_decision(
        session,
        packet,
        source_permission_values={
            "approval_policy": "on-request",
            "sandbox_mode": "workspace-write",
        },
        target_permission_values={"permission_mode": "default"},
        user_decision="deferred",
        final_action="previewed",
        reason="preview",
    )

    assert decision.permission_state["backend"] == "codex"
    assert decision.permission_state["preset_key"] == "ask"
    assert decision.compatibility["target_state"]["backend"] == "claude"
    assert decision.compatibility["target_state"]["preset_key"] == "ask"
    assert decision.compatibility["widens_permissions"] is False


def test_resume_context_preview_helpers_format_pending_context():
    controller = FakePreviewController("codex")

    assert resume_context_status_message(controller) == "Resume context pending: summary=summary-1 turns=2 chars=123"

    preview = format_resume_context_preview(controller)

    assert "Resume context preview" in preview
    assert "Summary: summary-1" in preview
    assert "Turns  : turn-1, turn-2" in preview
    assert "CCG LOCAL RESUME CONTEXT" in preview


def test_handoff_preview_helpers_format_packet_and_errors():
    controller = FakeController("codex")
    controller.session.summaries = [
        SummaryRecord(
            id="summary-handoff",
            scope="session",
            created_at="2026-04-21T15:06:00+00:00",
            text="Summary for handoff.",
        )
    ]
    controller.submit_prompt("prior work")

    ok, preview = build_handoff_preview(controller.session, "claude sonnet continue now")

    assert ok is True
    assert "Handoff Packet" in preview
    assert "Target Backend : claude" in preview
    assert "Target Model   : sonnet" in preview
    assert "Summary        : summary-handoff" in preview
    assert "prior work" in preview
    status = handoff_status_message(controller.session, "claude sonnet continue now")
    assert "source=session-1" in status
    assert "target=claude" in status
    assert "model=sonnet" in status
    assert "summary=summary-handoff" in status
    assert "included=turn-1" in status
    assert "omitted=<none>" in status
    assert "truncation=" in status

    ok, preview = build_handoff_preview(controller.session, "")
    assert ok is False
    assert "Usage: /handoff" in preview

    ok, preview = build_handoff_preview(controller.session, "unknown")
    assert ok is False
    assert "Unsupported target backend" in preview


def test_handoff_preview_defaults_to_active_task_boundary():
    controller = FakeController("codex")
    controller.session.summaries = [
        SummaryRecord(
            id="summary-session",
            scope="session",
            created_at="2026-04-21T15:06:00+00:00",
            text="Session summary",
        ),
        SummaryRecord(
            id="summary-task",
            scope="task:task-1",
            created_at="2026-04-21T15:07:00+00:00",
            text="Task summary",
        ),
    ]
    controller.submit_prompt("main work")
    handle_task_command(controller, "start Focused task")
    controller.submit_prompt("task work")

    ok, preview = build_handoff_preview(controller.session, "claude sonnet continue now")

    assert ok is True
    assert "Source Scope   : task:task-1" in preview
    assert "Summary        : summary-task" in preview
    assert "task work" in preview
    assert "main work" not in preview


def test_handoff_preview_accepts_curated_selection_args():
    controller = FakeController("codex")
    controller.submit_prompt("turn one")
    controller.submit_prompt("turn two")

    ok, preview = build_handoff_preview(
        controller.session,
        "claude sonnet continue --turn-id turn-2 --status completed --recent 1",
    )

    assert ok is True
    assert "Source Turns   : turn-2" in preview
    assert "Included IDs   : summary=<none> turns=turn-2" in preview
    assert "Criteria       : task=<none> statuses=completed recent=1" in preview
    assert "turn one" not in preview
    assert "turn two" in preview


def test_parse_handoff_args_parses_goal_and_filters():
    parsed = parse_handoff_args(
        ["claude", "sonnet", "continue", "now", "--task-id", "task-1", "--turn-id", "turn-2,turn-3", "--status", "completed,failed", "--recent", "2"]
    )

    assert parsed["target_backend"] == "claude"
    assert parsed["target_model"] == "sonnet"
    assert parsed["user_goal"] == "continue now"
    assert parsed["task_id"] == "task-1"
    assert parsed["turn_ids"] == ["turn-2", "turn-3"]
    assert parsed["statuses"] == ["completed", "failed"]
    assert parsed["recent_turn_limit"] == 2


def test_build_status_text_reflects_streaming_state():
    controller = FakeController("claude")
    controller.submit_prompt("hello")
    controller.session.turns[-1].status = TurnStatus.STREAMING

    text = build_status_text(controller, composer_message="Streaming reply", is_busy=True)

    assert "Streaming reply" in text
    assert "backend=claude" in text
    assert "Enter submit" in text
    assert "Shift-Enter newline" in text
    assert "Ctrl-J submit" in text
    assert "Esc-Enter newline fallback" in text
    assert "Esc quit" in text


def test_spinner_frame_cycles_braille_frames():
    assert spinner_frame(0) == "⠋"
    assert spinner_frame(1) == "⠙"
    assert spinner_frame(10) == "⠋"


def test_phase_label_distinguishes_launching_waiting_streaming_and_finalizing():
    waiting = phase_label(TurnStatus.SUBMITTING, output="", tick=0)
    streaming_empty = phase_label(TurnStatus.STREAMING, output="", tick=1)
    streaming_partial = phase_label(TurnStatus.STREAMING, output="hello", tick=2)
    finalizing = phase_label(TurnStatus.COMPLETED, output="hello", tick=3)

    assert "launching" in waiting
    assert "waiting" in streaming_empty
    assert "streaming" in streaming_partial
    assert "finalizing" in finalizing


def test_turn_is_busy_covers_submitting_and_streaming_only():
    submitting = TurnRecord(
        id="turn-submitting",
        backend=BackendName.CLAUDE,
        prompt="hello",
        output="",
        status=TurnStatus.SUBMITTING,
        started_at="2026-04-21T15:05:01+00:00",
    )
    streaming = TurnRecord(
        id="turn-streaming",
        backend=BackendName.CLAUDE,
        prompt="hello",
        output="partial",
        status=TurnStatus.STREAMING,
        started_at="2026-04-21T15:05:01+00:00",
    )
    completed = TurnRecord(
        id="turn-completed",
        backend=BackendName.CLAUDE,
        prompt="hello",
        output="done",
        status=TurnStatus.COMPLETED,
        started_at="2026-04-21T15:05:01+00:00",
    )

    assert turn_is_busy(submitting) is True
    assert turn_is_busy(streaming) is True
    assert turn_is_busy(completed) is False


def test_progress_message_uses_phase_label_while_busy_and_last_turn_when_done():
    turn = TurnRecord(
        id="turn-progress",
        backend=BackendName.GEMINI,
        prompt="hello",
        output="",
        status=TurnStatus.SUBMITTING,
        started_at="2026-04-21T15:05:01+00:00",
    )

    assert "launching" in progress_message(turn, tick=0)

    turn.status = TurnStatus.STREAMING
    turn.output = "partial"
    assert "streaming" in progress_message(turn, tick=2)
    assert "7 chars" in progress_message(turn, tick=2)

    turn.status = TurnStatus.COMPLETED
    assert progress_message(turn, tick=3) == "Last turn: completed"


def test_progress_and_meta_lines_include_recent_backend_activity():
    turn = TurnRecord(
        id="turn-activity",
        backend=BackendName.CODEX,
        prompt="inspect",
        output="",
        status=TurnStatus.STREAMING,
        started_at="2026-04-21T15:05:01+00:00",
        events=[
            RecordedEvent(
                type="activity",
                observed_at="2026-04-21T15:05:02+00:00",
                text="tool: exec_command cmd=rg --files",
                activity={
                    "kind": "tool_started",
                    "title": "exec_command",
                    "backend_label": "tool: exec_command cmd=rg --files",
                    "status": "started",
                    "details": {"arguments": {"cmd": "rg --files"}},
                },
            ),
        ],
    )

    assert latest_activity_text(turn) == "tool: exec_command cmd=rg --files"
    assert "tool: exec_command cmd=rg --files" in progress_message(turn)
    assert "activity > tool: exec_command cmd=rg --files [started]" in turn_meta_lines(turn, turn_number=1)
    assert "           arguments: {\"cmd\": \"rg --files\"}" in turn_meta_lines(
        turn,
        turn_number=1,
        show_activity_details=True,
    )


def test_transcript_turn_separator_labels_turn_number():
    assert "turn 2" in transcript_turn_separator(2)


def test_turn_meta_lines_include_status_backend_and_errors():
    completed = TurnRecord(
        id="turn-complete",
        backend=BackendName.CLAUDE,
        prompt="hello",
        output="done",
        status=TurnStatus.COMPLETED,
        started_at="2026-04-21T15:05:01+00:00",
    )
    failed = TurnRecord(
        id="turn-failed",
        backend=BackendName.GEMINI,
        prompt="hello",
        output="",
        status=TurnStatus.FAILED,
        started_at="2026-04-21T15:05:01+00:00",
        error=NormalizedError(kind="backend_error", message="bad auth"),
    )

    completed_lines = turn_meta_lines(completed, turn_number=1)
    failed_lines = turn_meta_lines(failed, turn_number=2)

    assert completed_lines[0] == "meta   > turn 1 • claude • completed"
    assert failed_lines[0] == "meta   > turn 2 • gemini • failed"
    assert failed_lines[1] == "recovery> failed; error=backend_error"
    assert failed_lines[2] == "error  > bad auth"


def test_stalled_message_appears_after_long_wait():
    text = stalled_message(TurnStatus.SUBMITTING, elapsed_seconds=9.5)

    assert "still waiting" in text


def test_colorize_backend_label_wraps_text_in_ansi_sequences():
    colored = colorize_backend_label("gemini", "gemini")

    assert colored != "gemini"
    assert colored.endswith(ANSI_RESET)


def test_build_transcript_text_formats_chat_log():
    controller = FakeController("claude")
    controller.submit_prompt("hello")

    text = build_transcript_text(controller)

    assert "──── turn 1" in text
    assert "──── turn 1 ────────────────────\n\nYou    > hello\n\nclaude > reply to hello\n\nmeta   > turn 1 • claude • completed" in text


def test_build_transcript_text_includes_active_streaming_turn():
    controller = FakeController("codex")
    controller.active_turn = TurnRecord(
        id="turn-stream",
        backend=BackendName.CODEX,
        prompt="stream me",
        output="partial",
        status=TurnStatus.STREAMING,
        started_at="2026-04-21T15:05:01+00:00",
    )

    text = build_transcript_text(controller)

    assert "turn 1" in text
    assert "You    > stream me" in text
    assert "codex  > partial" in text
    assert "meta   > turn 1 • codex • ⠋ streaming…" in text


def test_build_transcript_text_does_not_duplicate_persisted_active_turn():
    controller = FakeController("codex")
    active_turn = TurnRecord(
        id="turn-stream",
        backend=BackendName.CODEX,
        prompt="stream me",
        output="partial",
        status=TurnStatus.STREAMING,
        started_at="2026-04-21T15:05:01+00:00",
    )
    controller.session.turns.append(active_turn)
    controller.active_turn = active_turn

    text = build_transcript_text(controller)

    assert text.count("──── turn ") == 1
    assert "turn 2" not in text


def test_build_transcript_text_shows_waiting_placeholder_before_first_output():
    controller = FakeController("claude")
    controller.active_turn = TurnRecord(
        id="turn-waiting",
        backend=BackendName.CLAUDE,
        prompt="do something",
        output="",
        status=TurnStatus.SUBMITTING,
        started_at="2026-04-21T15:05:01+00:00",
    )

    text = build_transcript_text(controller)

    assert "You    > do something" in text
    assert "claude > ⠋ launching…" in text
    assert "meta   > turn 1 • claude • ⠋ launching…" in text


def test_format_markdown_lines_renders_basic_code_blocks_and_bullets():
    lines = format_markdown_lines("""Here is code:\n```python\nprint('hi')\n```\n- item one\n## Heading\n""")

    joined = "\n".join(lines)
    assert "Here is code:" in joined
    assert "│ print('hi')" in joined
    assert "• item one" in joined
    assert "HEADING" in joined


def test_render_interface_screen_looks_like_chat_console():
    controller = FakeController("claude")
    controller.submit_prompt("hello")

    screen = render_interface_screen(controller)

    assert "CCG TUI" in screen
    assert "[claude]" in screen
    assert "Session: session-1" in screen
    assert "Conversation" in screen
    assert "──── turn 1 ────────────────────\n\nYou    > hello\n\nclaude > reply to hello\n\nmeta   > turn 1 • claude • completed" in screen
    assert "Composer" in screen
    assert "Enter submits" in screen
    assert "Shift-Enter adds a newline" in screen
    assert "Ctrl-J submits as a fallback" in screen
    assert "Esc-Enter adds a newline fallback" in screen
    assert "/quit" in screen


def test_render_interface_screen_formats_multiline_assistant_output():
    controller = FakeController("claude")
    turn = controller.submit_prompt("hello")
    turn.output = "\nfirst line\nsecond line"

    screen = render_interface_screen(controller)

    assert "claude > first line" in screen
    assert "         second line" in screen


def test_render_interface_screen_formats_markdown_code_block_output():
    controller = FakeController("claude")
    turn = controller.submit_prompt("show code")
    turn.output = "```python\nprint('hi')\n```"

    screen = render_interface_screen(controller)

    assert "claude > ┌─ code:python" in screen
    assert "         │ print('hi')" in screen
    assert "         └─\n\nmeta   > turn 1 • claude • completed" in screen


def test_render_interface_screen_shows_running_feedback_in_conversation():
    controller = FakeController("gemini")
    controller.active_turn = TurnRecord(
        id="turn-running",
        backend=BackendName.GEMINI,
        prompt="summarize this",
        output="",
        status=TurnStatus.SUBMITTING,
        started_at="2026-04-21T15:05:01+00:00",
    )

    screen = render_interface_screen(controller)

    assert "gemini > ⠋ launching…" in screen
    assert "meta   > turn 1 • gemini • ⠋ launching…" in screen


def test_render_interface_screen_shows_stalled_feedback_after_delay():
    controller = FakeController("codex")
    controller.active_turn = TurnRecord(
        id="turn-stalled",
        backend=BackendName.CODEX,
        prompt="long task",
        output="",
        status=TurnStatus.SUBMITTING,
        started_at="2026-04-21T15:05:01+00:00",
    )

    screen = render_interface_screen(controller, composer_text="", tick=3, elapsed_seconds=12.0)

    assert "meta   > still waiting for backend response" in screen


def test_run_interface_selects_backend_sends_prompt_and_shows_history():
    answers = iter(["gemini", "hello", "/history", "/quit"])
    printed: list[str] = []
    created: list[FakeController] = []

    def factory(backend: str):
        controller = FakeController(backend)
        created.append(controller)
        return controller

    code = run_interface(
        controller_factory=factory,
        input_fn=lambda _: next(answers),
        print_fn=printed.append,
    )

    assert code == 0
    assert created[0].session.backend is BackendName.GEMINI
    assert created[0].prompts == ["hello"]
    assert any("[gemini]" in line for line in printed)
    assert any("reply to hello" in line for line in printed)
    assert any("Conversation" in line for line in printed)


def test_run_interface_translates_core_backend_slash_commands():
    answers = iter(["codex", "/memory", "/compact now", "/quit"])
    created: list[FakeController] = []

    def factory(backend: str):
        controller = FakeController(backend)
        created.append(controller)
        return controller

    code = run_interface(
        controller_factory=factory,
        input_fn=lambda _: next(answers),
        print_fn=lambda _: None,
    )

    assert code == 0
    assert created[0].prompts == ["/memories", "/compact now"]


def test_run_interface_handoff_preview_does_not_submit_backend_prompt():
    answers = iter(["codex", "before", "/handoff claude sonnet continue", "/quit"])
    printed: list[str] = []
    created: list[FakeController] = []

    def factory(backend: str):
        controller = FakeController(backend)
        created.append(controller)
        return controller

    code = run_interface(
        controller_factory=factory,
        input_fn=lambda _: next(answers),
        print_fn=printed.append,
    )

    assert code == 0
    assert created[0].prompts == ["before"]
    assert any("Handoff Packet" in line for line in printed)
    assert any("Target Backend : claude" in line for line in printed)
    assert created[0].session.routing_decisions[0].suggested_backend is BackendName.CLAUDE
    assert created[0].session.routing_decisions[0].user_decision == "deferred"
    assert created[0].session.routing_decisions[0].final_action == "previewed"


def test_run_interface_capabilities_command_is_advisory_and_records_audit():
    answers = iter(["codex", "/capabilities", "/quit"])
    printed: list[str] = []
    created: list[FakeController] = []

    def factory(backend: str):
        controller = FakeController(backend)
        created.append(controller)
        return controller

    code = run_interface(
        controller_factory=factory,
        input_fn=lambda _: next(answers),
        print_fn=printed.append,
    )

    assert code == 0
    assert created[0].prompts == []
    assert any("Routing Capability Registry" in line for line in printed)
    assert created[0].session.backend is BackendName.CODEX
    assert created[0].session.routing_decisions[0].trigger == "capability_registry_inspected"
    assert created[0].session.routing_decisions[0].final_action == "capabilities_displayed"


def test_run_interface_clear_starts_fresh_session():
    answers = iter(["claude", "before", "/clear", "after", "/quit"])
    created: list[FakeController] = []

    def factory(backend: str):
        controller = FakeController(backend)
        created.append(controller)
        return controller

    code = run_interface(
        controller_factory=factory,
        input_fn=lambda _: next(answers),
        print_fn=lambda _: None,
    )

    assert code == 0
    assert len(created) == 2
    assert created[0].prompts == ["before"]
    assert created[0].closed is True
    assert created[1].prompts == ["after"]


def test_run_interface_model_command_updates_adapter_model():
    answers = iter(["gemini", "/model gemini-2.5-flash", "/quit"])
    created: list[FakeController] = []

    def factory(backend: str):
        controller = FakeController(backend)
        created.append(controller)
        return controller

    code = run_interface(
        controller_factory=factory,
        input_fn=lambda _: next(answers),
        print_fn=lambda _: None,
    )

    assert code == 0
    assert created[0].adapter.model == "gemini-2.5-flash"


def test_run_interface_model_command_lists_options_when_no_argument(tmp_path, monkeypatch):
    answers = iter(["codex", "/model", "/quit"])
    printed: list[str] = []
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "gpt-test-codex",
                        "display_name": "GPT Test Codex",
                        "description": "Model from Codex cache.",
                        "visibility": "list",
                    }
                ]
            }
        )
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr("ccg_tui.app._codex_model_options_from_debug", lambda: ())

    code = run_interface(
        controller_factory=lambda backend: FakeController(backend),
        input_fn=lambda _: next(answers),
        print_fn=printed.append,
    )

    assert code == 0
    assert any("Model Picker (codex)" in line for line in printed)
    assert any("gpt-test-codex" in line for line in printed)


@pytest.mark.parametrize(
    ("backend", "preset", "expected_attrs"),
    [
        ("codex", "full-access", {"approval_policy": "never", "sandbox_mode": "danger-full-access"}),
        ("claude", "ask", {"permission_mode": "default"}),
        ("gemini", "auto-edit", {"approval_mode": "auto_edit"}),
    ],
)
def test_run_interface_permissions_command_updates_backend_specific_adapter_permissions(
    backend,
    preset,
    expected_attrs,
):
    answers = iter([backend, f"/permissions {preset}", "/quit"])
    created: list[FakeController] = []

    def factory(selected_backend: str):
        controller = FakeController(selected_backend)
        created.append(controller)
        return controller

    code = run_interface(
        controller_factory=factory,
        input_fn=lambda _: next(answers),
        print_fn=lambda _: None,
    )

    assert code == 0
    for attr, value in expected_attrs.items():
        assert getattr(created[0].adapter, attr) == value


def test_run_interface_permissions_command_preserves_current_model():
    answers = iter(["claude", "/model claude-sonnet-4-5-20250929", "/permissions ask", "/quit"])
    created: list[FakeController] = []

    def factory(backend: str):
        controller = FakeController(backend)
        created.append(controller)
        return controller

    code = run_interface(
        controller_factory=factory,
        input_fn=lambda _: next(answers),
        print_fn=lambda _: None,
    )

    assert code == 0
    assert created[0].adapter.model == "claude-sonnet-4-5-20250929"
    assert created[0].adapter.permission_mode == "default"


def test_run_interface_permissions_command_lists_options_when_no_argument():
    answers = iter(["gemini", "/permissions", "/quit"])
    printed: list[str] = []

    code = run_interface(
        controller_factory=lambda backend: FakeController(backend),
        input_fn=lambda _: next(answers),
        print_fn=printed.append,
    )

    assert code == 0
    assert any("Permissions Picker (gemini)" in line for line in printed)
    assert any("approval_mode=plan" in line for line in printed)
    assert any("auto_edit" in line for line in printed)


def test_run_interface_permissions_command_reports_unknown_preset():
    answers = iter(["codex", "/permissions unknown", "/quit"])
    printed: list[str] = []

    code = run_interface(
        controller_factory=lambda backend: FakeController(backend),
        input_fn=lambda _: next(answers),
        print_fn=printed.append,
    )

    assert code == 0
    assert any("Unknown permission preset: unknown" in line for line in printed)


def test_format_session_list_renders_core_metadata(tmp_path):
    store = TranscriptStore(tmp_path)
    store.save_session(
        SessionRecord(
            id="session-list",
            backend=BackendName.CLAUDE,
            created_at="2026-04-21T15:05:00+00:00",
            updated_at="2026-04-21T15:06:00+00:00",
            workspace_cwd="/tmp/workspace-list",
            turns=[
                TurnRecord(
                    id="turn-list",
                    backend=BackendName.CLAUDE,
                    prompt="hello",
                    output="world",
                    status=TurnStatus.COMPLETED,
                    started_at="2026-04-21T15:05:01+00:00",
                    completed_at="2026-04-21T15:05:02+00:00",
                )
            ],
            summaries=[
                SummaryRecord(
                    id="summary-list",
                    scope="task:task-main",
                    created_at="2026-04-21T15:06:00+00:00",
                    text="summary",
                )
            ],
        )
    )

    text = format_session_list(store.list_sessions())

    assert "session_id" in text
    assert "updated_at" in text
    assert "created_at" in text
    assert "resumable" in text
    assert "session-list" in text
    assert "claude" in text
    assert "completed" in text
    assert "yes" in text
    assert "workspace-list" in text


def test_cli_list_sessions_prints_table(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    store = TranscriptStore(tmp_path / "runtime" / "transcripts")
    store.save_session(
        SessionRecord(
            id="session-cli-list",
            backend=BackendName.GEMINI,
            created_at="2026-04-21T15:05:00+00:00",
            workspace_cwd=str(tmp_path),
        )
    )

    code = main(["--list-sessions"])
    output = capsys.readouterr().out

    assert code == 0
    assert "session_id" in output
    assert "session-cli-list" in output
    assert "gemini" in output
    assert "idle" in output
    assert "resumable" in output


def test_cli_handoff_session_prints_preview_without_backend_call(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    store = TranscriptStore(tmp_path / "runtime" / "transcripts")
    store.save_session(
        SessionRecord(
            id="session-cli-handoff",
            backend=BackendName.CODEX,
            created_at="2026-04-21T15:05:00+00:00",
            workspace_cwd=str(tmp_path),
                summaries=[
                    SummaryRecord(
                        id="summary-cli-handoff",
                        scope="session",
                        created_at="2026-04-21T15:06:00+00:00",
                        text="Keep the packet deterministic.",
                )
            ],
            turns=[
                TurnRecord(
                    id="turn-cli-handoff",
                    backend=BackendName.CODEX,
                    prompt="build a packet",
                    output="packet built",
                    status=TurnStatus.COMPLETED,
                    started_at="2026-04-21T15:05:01+00:00",
                    completed_at="2026-04-21T15:05:02+00:00",
                )
            ],
        )
    )
    monkeypatch.setattr(
        "ccg_tui.app.build_backend",
        lambda name: (_ for _ in ()).throw(AssertionError("handoff preview should not build a backend")),
    )

    code = main(
        [
            "--handoff-session",
            "session-cli-handoff",
            "--target-backend",
            "claude",
            "--target-model",
            "sonnet",
            "--handoff-goal",
            "continue now",
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "Handoff Packet" in output
    assert "Source Session : session-cli-handoff" in output
    assert "Source Backend : codex" in output
    assert "Target Backend : claude" in output
    assert "Target Model   : sonnet" in output
    assert "Summary        : summary-cli-handoff" in output
    assert "Source Turns   : turn-cli-handoff" in output
    assert "CCG MANUAL HANDOFF PACKET" in output
    assert "Keep the packet deterministic." in output
    assert "build a packet" in output
    assert 'Current user goal JSON: "continue now"' in output
    decision = store.load_session("session-cli-handoff").routing_decisions[0]
    assert decision.active_backend is BackendName.CODEX
    assert decision.suggested_backend is BackendName.CLAUDE
    assert decision.suggested_model == "sonnet"
    assert decision.user_decision == "deferred"
    assert decision.final_action == "previewed"


def test_cli_handoff_session_writes_export_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    store = TranscriptStore(tmp_path / "runtime" / "transcripts")
    store.save_session(
        SessionRecord(
            id="session-cli-handoff-export",
            backend=BackendName.GEMINI,
            created_at="2026-04-21T15:05:00+00:00",
            turns=[
                TurnRecord(
                    id="turn-export",
                    backend=BackendName.GEMINI,
                    prompt="export me",
                    output="exported",
                    status=TurnStatus.COMPLETED,
                    started_at="2026-04-21T15:05:01+00:00",
                    completed_at="2026-04-21T15:05:02+00:00",
                )
            ],
        )
    )
    output_path = tmp_path / "handoffs" / "packet.txt"

    code = main(
        [
            "--handoff-session",
            "session-cli-handoff-export",
            "--target-backend",
            "codex",
            "--handoff-output",
            str(output_path),
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert f"Handoff packet written: {output_path}" in output
    exported = output_path.read_text()
    assert "Handoff Packet" in exported
    assert "Target Backend : codex" in exported
    assert "export me" in exported
    decision = store.load_session("session-cli-handoff-export").routing_decisions[0]
    assert decision.suggested_backend is BackendName.CODEX
    assert decision.final_action == "preview_exported"


def test_cli_handoff_session_writes_curated_selection_metadata(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    store = TranscriptStore(tmp_path / "runtime" / "transcripts")
    store.save_session(
        SessionRecord(
            id="session-cli-handoff-curated",
            backend=BackendName.GEMINI,
            created_at="2026-04-21T15:05:00+00:00",
            turns=[
                TurnRecord(
                    id="turn-1",
                    backend=BackendName.GEMINI,
                    prompt="first",
                    output="first output",
                    status=TurnStatus.COMPLETED,
                    started_at="2026-04-21T15:05:01+00:00",
                    completed_at="2026-04-21T15:05:02+00:00",
                    task_id="task-main",
                ),
                TurnRecord(
                    id="turn-2",
                    backend=BackendName.GEMINI,
                    prompt="second",
                    output="second output",
                    status=TurnStatus.COMPLETED,
                    started_at="2026-04-21T15:05:03+00:00",
                    completed_at="2026-04-21T15:05:04+00:00",
                    task_id="task-main",
                ),
            ],
        )
    )
    output_path = tmp_path / "handoffs" / "curated.txt"

    code = main(
        [
            "--handoff-session",
            "session-cli-handoff-curated",
            "--target-backend",
            "codex",
            "--handoff-turn-id",
            "turn-2",
            "--handoff-status",
            "completed",
            "--handoff-recent",
            "1",
            "--handoff-output",
            str(output_path),
        ]
    )

    assert code == 0
    exported = output_path.read_text()
    assert "Source Turns   : turn-2" in exported
    assert "Included IDs   : summary=<none> turns=turn-2" in exported
    assert "Criteria       : task=<none> statuses=completed recent=1" in exported
    assert "first" not in exported
    assert "second" in exported


def test_cli_handoff_preview_does_not_create_target_session_or_turn(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    store = TranscriptStore(tmp_path / "runtime" / "transcripts")
    store.save_session(
        SessionRecord(
            id="session-cli-handoff-preview-only",
            backend=BackendName.CODEX,
            created_at="2026-04-21T15:05:00+00:00",
            workspace_cwd=str(tmp_path),
            turns=[
                TurnRecord(
                    id="turn-source",
                    backend=BackendName.CODEX,
                    prompt="before",
                    output="old",
                    status=TurnStatus.COMPLETED,
                    started_at="2026-04-21T15:05:01+00:00",
                    completed_at="2026-04-21T15:05:02+00:00",
                )
            ],
        )
    )

    code = main(
        [
            "--handoff-session",
            "session-cli-handoff-preview-only",
            "--target-backend",
            "claude",
            "--handoff-goal",
            "continue in claude",
        ]
    )
    output = capsys.readouterr().out
    sessions = store.list_sessions()
    source = store.load_session("session-cli-handoff-preview-only")

    assert code == 0
    assert "Handoff Packet" in output
    assert [session.id for session in sessions] == ["session-cli-handoff-preview-only"]
    assert len(source.turns) == 1
    assert source.routing_decisions[0].final_action == "previewed"


def test_cli_handoff_execute_creates_new_target_session_with_lineage(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    adapter = FakeModelCliAdapter(name=BackendName.CODEX)

    def fake_build_backend(name, **kwargs):
        adapter.name = BackendName(name)
        adapter.model = kwargs.get("model")
        return adapter

    monkeypatch.setattr("ccg_tui.app.build_backend", fake_build_backend)
    store = TranscriptStore(tmp_path / "runtime" / "transcripts")
    store.save_session(
        SessionRecord(
            id="session-cli-handoff-source",
            backend=BackendName.CLAUDE,
            created_at="2026-04-21T15:05:00+00:00",
            workspace_cwd=str(tmp_path),
            turns=[
                TurnRecord(
                    id="turn-source",
                    backend=BackendName.CLAUDE,
                    prompt="before",
                    output="old",
                    status=TurnStatus.COMPLETED,
                    started_at="2026-04-21T15:05:01+00:00",
                    completed_at="2026-04-21T15:05:02+00:00",
                )
            ],
        )
    )
    source_before = store.load_session("session-cli-handoff-source").to_dict()

    code = main(
        [
            "--handoff-session",
            "session-cli-handoff-source",
            "--target-backend",
            "codex",
            "--target-model",
            "gpt-test",
            "--handoff-goal",
            "continue in codex",
            "--handoff-execute",
        ]
    )
    output = capsys.readouterr().out
    sessions = store.list_sessions()
    target_session_id = next(session.id for session in sessions if session.id != "session-cli-handoff-source")
    target = store.load_session(target_session_id)
    source = store.load_session("session-cli-handoff-source")

    assert code == 0
    assert "Handoff execution confirmed by --handoff-execute" in output
    assert "Source Scope   : session" in output
    assert "Exclusions     : <none>" in output
    assert "Session :" in output
    assert "cli reply" in output
    assert adapter.model == "gpt-test"
    assert len(adapter.prompts) == 1
    assert "CCG MANUAL HANDOFF PACKET" in adapter.prompts[0]
    assert 'Current user goal JSON: "continue in codex"' in adapter.prompts[0]
    assert target.backend is BackendName.CODEX
    assert target.lineage.kind == "handoff"
    assert target.lineage.parent_session_id == "session-cli-handoff-source"
    assert target.lineage.forked_from_turn_id == "turn-source"
    handoff_relationship = next(
        relationship for relationship in target.lineage.relationships if relationship.kind == "handoff"
    )
    assert handoff_relationship.metadata["source_turn_ids"] == ["turn-source"]
    assert handoff_relationship.metadata["target_model"] == "gpt-test"
    assert handoff_relationship.metadata["audit"]["turns"]["included_source_ids"] == ["turn-source"]
    assert target.turns[0].prompt == "continue in codex"
    assert target.turns[0].metadata["handoff"]["injected"] is True
    assert target.turns[0].metadata["handoff"]["execution_confirmed"] is True
    assert target.turns[0].metadata["handoff"]["confirmation_method"] == "--handoff-execute"
    assert target.turns[0].metadata["handoff"]["visible_prompt"] == "continue in codex"
    assert target.routing_decisions[0].active_backend is BackendName.CLAUDE
    assert target.routing_decisions[0].suggested_backend is BackendName.CODEX
    assert target.routing_decisions[0].user_decision == "confirmed"
    assert target.routing_decisions[0].final_action == "handoff_session_started"
    assert "resume_context" not in target.turns[0].metadata
    assert source.to_dict() == source_before


def test_cli_handoff_execute_anchors_lineage_to_curated_packet_turn(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    adapter = FakeModelCliAdapter(name=BackendName.CODEX)

    def fake_build_backend(name, **kwargs):
        adapter.name = BackendName(name)
        adapter.model = kwargs.get("model")
        return adapter

    monkeypatch.setattr("ccg_tui.app.build_backend", fake_build_backend)
    store = TranscriptStore(tmp_path / "runtime" / "transcripts")
    store.save_session(
        SessionRecord(
            id="session-cli-handoff-curated-source",
            backend=BackendName.CLAUDE,
            created_at="2026-04-21T15:05:00+00:00",
            workspace_cwd=str(tmp_path),
            summaries=[
                SummaryRecord(
                    id="summary-alpha",
                    scope="task:task-alpha",
                    created_at="2026-04-21T15:05:03+00:00",
                    text="Alpha checkpoint",
                )
            ],
            turns=[
                TurnRecord(
                    id="turn-alpha",
                    backend=BackendName.CLAUDE,
                    prompt="alpha work",
                    output="alpha done",
                    status=TurnStatus.COMPLETED,
                    started_at="2026-04-21T15:05:01+00:00",
                    completed_at="2026-04-21T15:05:02+00:00",
                    task_id="task-alpha",
                ),
                TurnRecord(
                    id="turn-latest",
                    backend=BackendName.CLAUDE,
                    prompt="unrelated latest",
                    output="ignore me",
                    status=TurnStatus.COMPLETED,
                    started_at="2026-04-21T15:06:01+00:00",
                    completed_at="2026-04-21T15:06:02+00:00",
                    task_id="task-main",
                ),
            ],
        )
    )
    source_before = store.load_session("session-cli-handoff-curated-source").to_dict()

    code = main(
        [
            "--handoff-session",
            "session-cli-handoff-curated-source",
            "--target-backend",
            "codex",
            "--target-model",
            "gpt-test",
            "--handoff-goal",
            "continue alpha",
            "--handoff-task-id",
            "task-alpha",
            "--handoff-turn-id",
            "turn-alpha",
            "--handoff-status",
            "completed",
            "--handoff-recent",
            "1",
            "--handoff-execute",
        ]
    )
    capsys.readouterr()
    target_session_id = next(
        session.id for session in store.list_sessions() if session.id != "session-cli-handoff-curated-source"
    )
    target = store.load_session(target_session_id)

    assert code == 0
    assert target.lineage.forked_from_turn_id == "turn-alpha"
    handoff_relationship = next(
        relationship for relationship in target.lineage.relationships if relationship.kind == "handoff"
    )
    assert handoff_relationship.source_turn_id == "turn-alpha"
    assert handoff_relationship.metadata["source_scope"] == "task:task-alpha"
    assert handoff_relationship.metadata["source_summary_id"] == "summary-alpha"
    assert handoff_relationship.metadata["source_turn_ids"] == ["turn-alpha"]
    assert handoff_relationship.metadata["selection_criteria"] == {
        "scope": "task:task-alpha",
        "task_id": "task-alpha",
        "turn_ids": ["turn-alpha"],
        "statuses": ["completed"],
        "recent_turn_limit": 1,
    }
    assert target.turns[0].metadata["handoff"]["source_turn_ids"] == ["turn-alpha"]
    assert target.turns[0].metadata["handoff"]["source_scope"] == "task:task-alpha"
    assert store.load_session("session-cli-handoff-curated-source").to_dict() == source_before


def test_cli_handoff_execute_requires_goal(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    store = TranscriptStore(tmp_path / "runtime" / "transcripts")
    store.save_session(
        SessionRecord(
            id="session-cli-handoff-no-goal",
            backend=BackendName.CODEX,
            created_at="2026-04-21T15:05:00+00:00",
        )
    )

    code = main(
        [
            "--handoff-session",
            "session-cli-handoff-no-goal",
            "--target-backend",
            "claude",
            "--handoff-execute",
        ]
    )
    captured = capsys.readouterr()

    assert code == 2
    assert "--handoff-goal is required" in captured.err


def test_handoff_execution_confirmation_includes_exclusions():
    session = SessionRecord(
        id="session-confirm",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            TurnRecord(
                id="turn-1",
                backend=BackendName.CODEX,
                prompt="keep",
                output="ok",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:01+00:00",
            ),
            TurnRecord(
                id="turn-2",
                backend=BackendName.CODEX,
                prompt="exclude",
                output="failed",
                status=TurnStatus.FAILED,
                started_at="2026-04-21T15:05:02+00:00",
            ),
        ],
    )
    packet = build_handoff_packet(
        session,
        target_backend=BackendName.CLAUDE,
        statuses=["completed"],
    )

    text = format_handoff_execution_confirmation(packet, confirmation_method="test")

    assert "Source Scope   : session" in text
    assert "Included Turns : turn-1" in text
    assert "Exclusions     : turn-2=filtered_by_criteria" in text


def test_cli_handoff_session_requires_target_backend(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    code = main(["--handoff-session", "missing-target"])
    captured = capsys.readouterr()

    assert code == 2
    assert "--target-backend is required" in captured.err


def test_cli_handoff_session_rejects_invalid_target_backend(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    code = main(["--handoff-session", "session-any", "--target-backend", "unknown"])
    captured = capsys.readouterr()

    assert code == 2
    assert "Unsupported target backend: unknown" in captured.err


def test_cli_handoff_session_reports_missing_session(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    code = main(["--handoff-session", "session-missing", "--target-backend", "gemini"])
    captured = capsys.readouterr()

    assert code == 2
    assert "Session not found: session-missing" in captured.err


def test_cli_resume_session_prompt_uses_existing_session_id_and_appends_turn(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    adapter = FakeCliAdapter()
    monkeypatch.setattr("ccg_tui.app.build_backend", lambda name, **kwargs: adapter)
    store = TranscriptStore(tmp_path / "runtime" / "transcripts")
    store.save_session(
        SessionRecord(
            id="session-cli-resume",
            backend=BackendName.CODEX,
            created_at="2026-04-21T15:05:00+00:00",
            workspace_cwd=str(tmp_path),
            turns=[
                TurnRecord(
                    id="turn-existing",
                    backend=BackendName.CODEX,
                    prompt="before",
                    output="old",
                    status=TurnStatus.COMPLETED,
                    started_at="2026-04-21T15:05:01+00:00",
                    completed_at="2026-04-21T15:05:02+00:00",
                )
            ],
        )
    )

    code = main(["--resume-session", "session-cli-resume", "--prompt", "after"])
    output = capsys.readouterr().out
    loaded = store.load_session("session-cli-resume")

    assert code == 0
    assert "cli reply" in output
    assert "Context : injected" in output
    assert "\n" not in adapter.prompts[0]
    assert "CCG LOCAL RESUME CONTEXT" in adapter.prompts[0]
    assert "before" in adapter.prompts[0]
    assert 'Current user prompt JSON: "after"' in adapter.prompts[0]
    assert loaded.id == "session-cli-resume"
    assert [turn.prompt for turn in loaded.turns] == ["before", "after"]
    assert loaded.turns[-1].metadata["resume_context"]["injected_turn_ids"] == ["turn-existing"]


def test_cli_resume_context_can_be_disabled(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    adapter = FakeCliAdapter()
    monkeypatch.setattr("ccg_tui.app.build_backend", lambda name, **kwargs: adapter)
    store = TranscriptStore(tmp_path / "runtime" / "transcripts")
    store.save_session(
        SessionRecord(
            id="session-cli-context-off",
            backend=BackendName.CODEX,
            created_at="2026-04-21T15:05:00+00:00",
            workspace_cwd=str(tmp_path),
            turns=[
                TurnRecord(
                    id="turn-existing",
                    backend=BackendName.CODEX,
                    prompt="before",
                    output="old",
                    status=TurnStatus.COMPLETED,
                    started_at="2026-04-21T15:05:01+00:00",
                    completed_at="2026-04-21T15:05:02+00:00",
                )
            ],
        )
    )

    code = main(["--resume-session", "session-cli-context-off", "--resume-context", "off", "--prompt", "after"])
    loaded = store.load_session("session-cli-context-off")

    assert code == 0
    assert adapter.prompts == ["after"]
    assert loaded.turns[-1].metadata["recovery"]["state"] == "completed"


def test_cli_resume_session_rejects_cross_backend_existing_turns(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    store = TranscriptStore(tmp_path / "runtime" / "transcripts")
    store.save_session(
        SessionRecord(
            id="session-cli-reject",
            backend=BackendName.CODEX,
            created_at="2026-04-21T15:05:00+00:00",
            turns=[
                TurnRecord(
                    id="turn-existing",
                    backend=BackendName.CODEX,
                    prompt="before",
                    output="old",
                    status=TurnStatus.COMPLETED,
                    started_at="2026-04-21T15:05:01+00:00",
                    completed_at="2026-04-21T15:05:02+00:00",
                )
            ],
        )
    )

    code = main(["--resume-session", "session-cli-reject", "--backend", "claude", "--prompt", "after"])
    captured = capsys.readouterr()

    assert code == 2
    assert "one backend per session" in captured.err.lower()


def test_cli_resume_session_does_not_translate_backend_file_errors_to_missing_session(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ccg_tui.app.build_backend", lambda name, **kwargs: MissingFileCliAdapter())
    store = TranscriptStore(tmp_path / "runtime" / "transcripts")
    store.save_session(
        SessionRecord(
            id="session-cli-file-error",
            backend=BackendName.CODEX,
            created_at="2026-04-21T15:05:00+00:00",
            workspace_cwd=str(tmp_path),
            turns=[
                TurnRecord(
                    id="turn-existing",
                    backend=BackendName.CODEX,
                    prompt="before",
                    output="old",
                    status=TurnStatus.COMPLETED,
                    started_at="2026-04-21T15:05:01+00:00",
                    completed_at="2026-04-21T15:05:02+00:00",
                )
            ],
        )
    )

    try:
        main(["--resume-session", "session-cli-file-error", "--prompt", "after"])
    except FileNotFoundError as exc:
        assert "missing backend executable" in str(exc)
    else:
        raise AssertionError("expected backend FileNotFoundError to propagate")

    captured = capsys.readouterr()
    assert "Session not found" not in captured.err
