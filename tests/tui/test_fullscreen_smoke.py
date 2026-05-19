from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.tui


def load_tui_sessions(app) -> list[dict]:
    transcript_dir = app.transcript_dir
    assert transcript_dir is not None
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(transcript_dir).glob("*.json"))
    ]


def seed_interrupted_session(tui_app, *, prompt: str, cols: int = 180) -> tuple[str, str]:
    app = tui_app("--backend", "codex", cols=cols)

    app.expect_text("Ready")
    app.type(prompt)
    app.enter()
    app.expect_text(f"fake reply to {prompt}")
    app.type("/quit")
    app.enter()
    app.expect_exit(0)

    sessions = load_tui_sessions(app)
    assert len(sessions) == 1
    session = sessions[0]

    transcript_dir = app.transcript_dir
    assert transcript_dir is not None
    session_path = next(Path(transcript_dir).glob("*.json"))
    data = json.loads(session_path.read_text(encoding="utf-8"))
    data["turns"][0]["status"] = "failed"
    data["turns"][0]["output"] = f"partial reply to {prompt}"
    data["turns"][0]["error"] = {
        "kind": "interrupted",
        "message": "no terminal event",
        "exit_code": None,
        "details": {},
    }
    data["turns"][0]["metadata"] = {
        **data["turns"][0].get("metadata", {}),
        "recovery": {
            "state": "interrupted",
            "terminal_event_seen": False,
            "partial_output": True,
        },
    }
    if data.get("backend_sessions"):
        data["backend_sessions"][0]["status"] = "interrupted"
    session_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return session["id"], prompt


def test_fullscreen_local_commands_are_operable_through_pty(tui_app) -> None:
    app = tui_app("--backend", "codex")

    app.expect_text("CCG TUI")
    app.expect_text("Composer")
    app.expect_text("Sidebar")
    app.expect_text("Ready")
    app.expect_text("enter to send")

    app.type("/details")
    app.enter()
    app.expect_text("Activity details expanded")

    app.type("/details")
    app.enter()
    app.expect_text("Activity details collapsed")

    app.type("/history")
    app.enter()
    app.expect_text("Conversation refreshed")

    app.type("/quit")
    app.enter()
    app.expect_exit(0)


def test_fullscreen_keeps_terminal_mouse_reporting_disabled_for_text_selection(tui_app) -> None:
    app = tui_app("--backend", "codex")

    app.expect_text("Ready")
    mouse_reporting_enable_sequences = (
        "\x1b[?1000h",
        "\x1b[?1003h",
        "\x1b[?1006h",
        "\x1b[?1015h",
    )
    assert not any(sequence in app.raw_output for sequence in mouse_reporting_enable_sequences)

    app.type("/quit")
    app.enter()
    app.expect_exit(0)


def test_fullscreen_slash_palette_opens_in_handoff_order_and_esc_closes(tui_app) -> None:
    app = tui_app("--backend", "codex", cols=140, rows=40)

    app.expect_text("Ready")
    app.type("/")
    app.expect_text("Slash Commands")
    app.expect_text("/handoff")
    app.expect_text("/capabilities")

    app.press("escape")
    app.expect_text("Slash palette closed")
    app.expect_no_text("/capabilities")

    app.press("ctrl-c")
    app.expect_exit(0)


def test_fullscreen_slash_palette_scrolls_with_down_selection(tui_app) -> None:
    app = tui_app("--backend", "codex", cols=140, rows=34)

    app.expect_text("Ready")
    app.type("/")
    app.expect_text("/handoff")

    for _ in range(13):
        app.press("down")

    app.expect_text("/clear")
    app.expect_no_text("/handoff")

    app.press("escape")
    app.expect_text("Slash palette closed")
    app.press("ctrl-c")
    app.expect_exit(0)


def test_fullscreen_backend_picker_accepts_number_selection(tui_app) -> None:
    app = tui_app()

    app.expect_text("Select Backend")
    app.expect_text("1. codex")

    app.type("1")
    app.expect_text("Started codex session")
    app.expect_text("[codex]")

    app.type("/quit")
    app.enter()
    app.expect_exit(0)


def test_fullscreen_backend_picker_ignores_invalid_number_before_selection(tui_app) -> None:
    app = tui_app()

    app.expect_text("Select Backend")
    app.type("4")
    app.expect_text("Select Backend")

    app.type("2")
    app.expect_text("Started claude session")
    app.expect_text("[claude]")

    app.type("/quit")
    app.enter()
    app.expect_exit(0)


def test_fullscreen_prompt_submission_history_and_fake_backend_output(tui_app) -> None:
    app = tui_app(cols=120)

    app.expect_text("Select Backend")
    app.type("3")
    app.expect_text("Started gemini session")

    app.type("hello")
    app.enter()
    app.expect_text("fake reply to hello")

    app.type("/history")
    app.enter()
    app.expect_text("Conversation refreshed")

    app.type("/quit")
    app.enter()
    app.expect_exit(0)

    sessions = load_tui_sessions(app)
    turns = [turn for session in sessions for turn in session["turns"]]
    assert [turn["prompt"] for turn in turns] == ["hello"]
    assert turns[0]["status"] == "completed"


def test_fullscreen_backend_slash_commands_are_translated_before_submission(tui_app) -> None:
    app = tui_app("--backend", "codex", cols=120)

    app.expect_text("Ready")
    app.type("/memory")
    app.enter()
    app.expect_text("fake reply to /memories")

    app.type("/compact now")
    app.enter()
    app.expect_text("fake reply to /compact now")

    app.type("/quit")
    app.enter()
    app.expect_exit(0)

    turns = [turn for session in load_tui_sessions(app) for turn in session["turns"]]
    assert [turn["prompt"] for turn in turns] == ["/memories", "/compact now"]


def test_fullscreen_handoff_preview_is_local_and_does_not_submit_prompt(tui_app) -> None:
    app = tui_app("--backend", "codex", cols=180)

    app.expect_text("Ready")
    app.type("before")
    app.enter()
    app.expect_text("fake reply to before")

    app.type("/handoff claude sonnet continue")
    app.enter()
    app.expect_text("Handoff preview:")
    app.expect_text("source")
    app.expect_text("target")
    app.expect_text("target=claude")

    app.type("/quit")
    app.enter()
    app.expect_exit(0)

    session = load_tui_sessions(app)[0]
    assert [turn["prompt"] for turn in session["turns"]] == ["before"]
    assert session["routing_decisions"][0]["suggested_backend"] == "claude"
    assert session["routing_decisions"][0]["final_action"] == "previewed"


def test_fullscreen_interrupted_recovery_shortcuts_show_local_partial_summary_and_handoff(tui_app) -> None:
    session_id, prompt = seed_interrupted_session(tui_app, prompt="inspect this turn")

    app = tui_app("--resume-session", session_id, "--backend", "codex", "--resume-context", "off", cols=180)

    app.expect_text("TURN INTERRUPTED")

    app.type("c")
    app.expect_text("Partial output preview")
    app.expect_text("non-authoritative")

    app.type("s")
    app.expect_text("Summary checkpoint")
    app.expect_text("Resume context preview")

    app.type("h")
    app.expect_text("Handoff preview prefilled")
    app.expect_text("/handoff")

    app.press("ctrl-c")
    app.expect_exit(0)


def test_fullscreen_interrupted_retry_resubmits_original_prompt_only_when_idle(tui_app) -> None:
    session_id, prompt = seed_interrupted_session(tui_app, prompt="retry this prompt")

    app = tui_app("--resume-session", session_id, "--backend", "codex", "--resume-context", "off", cols=180)

    app.expect_text("TURN INTERRUPTED")

    app.type("r")
    app.expect_text(f"fake reply to {prompt}")

    app.type("/quit")
    app.enter()
    app.expect_exit(0)

    session = load_tui_sessions(app)[0]
    assert [turn["prompt"] for turn in session["turns"]] == [prompt, prompt]


def test_fullscreen_capabilities_command_is_local_and_records_audit(tui_app) -> None:
    app = tui_app("--backend", "codex", cols=220)

    app.expect_text("Ready")
    app.type("/capabilities")
    app.enter()
    app.expect_text("Routing Capability Registry")
    app.expect_text("capability matrix")
    app.expect_text("permission compatibility")
    app.expect_text("fit / partial / no")

    app.type("/quit")
    app.enter()
    app.expect_exit(0)

    session = load_tui_sessions(app)[0]
    assert session["turns"] == []
    assert session["routing_decisions"][0]["trigger"] == "capability_registry_inspected"
    assert session["routing_decisions"][0]["final_action"] == "capabilities_displayed"


def test_fullscreen_clear_starts_fresh_session(tui_app) -> None:
    app = tui_app("--backend", "claude", cols=120)

    app.expect_text("Ready")
    app.type("before")
    app.enter()
    app.expect_text("fake reply to before")

    app.type("/clear")
    app.enter()
    app.expect_text("Started fresh claude session")

    app.type("after")
    app.enter()
    app.expect_text("fake reply to after")

    app.type("/quit")
    app.enter()
    app.expect_exit(0)

    sessions = load_tui_sessions(app)
    prompts = sorted(turn["prompt"] for session in sessions for turn in session["turns"])
    assert prompts == ["after", "before"]
    assert len(sessions) == 2


def test_fullscreen_model_command_updates_adapter_model(tui_app) -> None:
    app = tui_app("--backend", "gemini", cols=140)

    app.expect_text("Ready")
    app.type("/model gemini-2.5-flash")
    app.enter()
    app.expect_text("Model set to gemini-2.5-flash for gemini.")
    app.expect_text("Model   : gemini-2.5-flash")

    app.type("/quit")
    app.enter()
    app.expect_exit(0)


def test_fullscreen_model_command_without_args_opens_picker(tui_app) -> None:
    app = tui_app("--backend", "codex", cols=140)

    app.expect_text("Ready")
    app.type("/model")
    app.enter()
    app.expect_text("Model Picker")
    app.expect_text("GPT-5.5")

    app.press("escape")
    app.expect_text("Model selection cancelled.")
    app.type("/quit")
    app.enter()
    app.expect_exit(0)


@pytest.mark.parametrize(
    ("backend", "preset", "expected_text"),
    [
        ("codex", "full-access", "sandbox_mode=danger-full-access"),
        ("claude", "ask", "permission_mode=default"),
        ("gemini", "auto-edit", "approval_mode=auto_edit"),
    ],
)
def test_fullscreen_permissions_command_updates_backend_specific_adapter_permissions(
    tui_app,
    backend,
    preset,
    expected_text,
) -> None:
    app = tui_app("--backend", backend, cols=160)

    app.expect_text("Ready")
    app.type(f"/permissions {preset}")
    app.enter()
    app.expect_text(expected_text)

    app.type("/quit")
    app.enter()
    app.expect_exit(0)


def test_fullscreen_permissions_command_preserves_current_model(tui_app) -> None:
    app = tui_app("--backend", "claude", cols=160)

    app.expect_text("Ready")
    app.type("/model claude-sonnet-4-5-20250929")
    app.enter()
    app.expect_text("Model set to claude-sonnet-4-5-20250929 for claude.")

    app.type("/permissions ask")
    app.enter()
    app.expect_text("permission_mode=default")
    app.expect_text("Model   : claude-sonnet-4-5-2025")
    app.expect_text("0929")

    app.type("/quit")
    app.enter()
    app.expect_exit(0)


def test_fullscreen_permissions_command_without_args_opens_picker(tui_app) -> None:
    app = tui_app("--backend", "gemini", cols=160)

    app.expect_text("Ready")
    app.type("/permissions")
    app.enter()
    app.expect_text("Permissions Picker")
    app.expect_text("approval_mode=plan")
    app.expect_text("auto_edit")

    app.press("escape")
    app.expect_text("Permissions selection cancelled.")
    app.type("/quit")
    app.enter()
    app.expect_exit(0)


def test_fullscreen_permissions_command_reports_unknown_preset(tui_app) -> None:
    app = tui_app("--backend", "codex", cols=140)

    app.expect_text("Ready")
    app.type("/permissions unknown")
    app.enter()
    app.expect_text("Unknown permission preset: unknown")

    app.type("/quit")
    app.enter()
    app.expect_exit(0)


def test_fullscreen_task_commands_are_local(tui_app) -> None:
    app = tui_app("--backend", "claude", cols=160)

    app.expect_text("Ready")
    app.type("/task start Investigate resume bug")
    app.enter()
    app.expect_text("Started task: Investigate resume bug")

    app.type("/task status")
    app.enter()
    app.expect_text("Active task: Investigate resume bug")

    app.type("/task close resolved")
    app.enter()
    app.expect_text("Closed task: Investigate resume bug")

    app.type("/quit")
    app.enter()
    app.expect_exit(0)

    session = load_tui_sessions(app)[0]
    assert session["turns"] == []
    assert session["tasks"][1]["title"] == "Investigate resume bug"
    assert session["tasks"][1]["status"] == "closed"
