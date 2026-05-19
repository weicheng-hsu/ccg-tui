from ccg_tui.models import BackendName, NormalizedError, SessionRecord, SummaryRecord, TurnRecord, TurnStatus
from ccg_tui.resume_context import ResumeContextConfig, build_resume_context_payload


def _turn(turn_id, prompt, output):
    return TurnRecord(
        id=turn_id,
        backend=BackendName.CODEX,
        prompt=prompt,
        output=output,
        status=TurnStatus.COMPLETED,
        started_at="2026-04-21T15:05:01+00:00",
        completed_at="2026-04-21T15:05:02+00:00",
    )


def test_build_resume_context_payload_uses_latest_summary_and_recent_turns():
    session = SessionRecord(
        id="session-resume",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd="/tmp/workspace",
        summaries=[
            SummaryRecord(
                id="summary-old",
                scope="task:task-main",
                created_at="2026-04-21T15:05:03+00:00",
                text="Old summary",
            ),
            SummaryRecord(
                id="summary-new",
                scope="task:task-main",
                created_at="2026-04-21T15:06:03+00:00",
                text="Latest summary",
            ),
        ],
        turns=[
            _turn("turn-1", "old prompt", "old output"),
            _turn("turn-2", "new prompt", "new output"),
        ],
    )

    payload = build_resume_context_payload(
        session,
        user_prompt="continue now",
        config=ResumeContextConfig(recent_turn_limit=1),
    )

    assert payload is not None
    assert "\n" not in payload.backend_prompt
    assert "CCG LOCAL RESUME CONTEXT" in payload.context_text
    assert "Latest summary" in payload.backend_prompt
    assert "new prompt" in payload.backend_prompt
    assert "old prompt" not in payload.backend_prompt
    assert 'Current user prompt JSON: "continue now"' in payload.backend_prompt
    assert payload.metadata["injected_summary_id"] == "summary-new"
    assert payload.metadata["injected_turn_ids"] == ["turn-2"]
    assert payload.metadata["serialized_format"] == "single_line_json"
    assert payload.metadata["context_char_count"] == len(payload.context_text)


def test_build_resume_context_payload_can_be_disabled():
    session = SessionRecord(
        id="session-resume",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[_turn("turn-1", "prompt", "output")],
    )

    payload = build_resume_context_payload(
        session,
        user_prompt="continue now",
        config=ResumeContextConfig(enabled=False),
    )

    assert payload is None


def test_build_resume_context_payload_escapes_multiline_user_prompt_for_pty_safety():
    session = SessionRecord(
        id="session-resume",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[_turn("turn-1", "prior\nprompt", "prior\noutput")],
    )

    payload = build_resume_context_payload(
        session,
        user_prompt="line one\nline two",
        config=ResumeContextConfig(recent_turn_limit=1),
    )

    assert payload is not None
    assert "\n" not in payload.backend_prompt
    assert "line one\\nline two" in payload.backend_prompt
    assert "prior\\nprompt" in payload.backend_prompt


def test_build_resume_context_payload_treats_completed_turn_as_authoritative():
    session = SessionRecord(
        id="session-resume",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            TurnRecord(
                id="turn-1",
                backend=BackendName.CODEX,
                prompt="finished prompt",
                output="finished output",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:01+00:00",
                completed_at="2026-04-21T15:05:02+00:00",
                metadata={
                    "recovery": {
                        "state": "completed",
                        "terminal_event_seen": True,
                        "partial_output": False,
                    }
                },
            )
        ],
    )

    payload = build_resume_context_payload(
        session,
        user_prompt="continue now",
        config=ResumeContextConfig(recent_turn_limit=1),
    )

    assert payload is not None
    assert "Latest turn recovery: none" in payload.context_text
    assert "Status: completed" in payload.context_text
    assert "Assistant output:\nfinished output" in payload.context_text
    assert "partial, not authoritative" not in payload.context_text
    assert payload.metadata["unreliable_turn_ids"] == []
    assert payload.metadata["partial_output_turn_ids"] == []


def test_build_resume_context_payload_marks_latest_incomplete_turn_as_unfinished():
    session = SessionRecord(
        id="session-resume",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            _turn("turn-1", "done prompt", "done output"),
            TurnRecord(
                id="turn-2",
                backend=BackendName.CODEX,
                prompt="latest prompt",
                output="partial output",
                status=TurnStatus.STREAMING,
                started_at="2026-04-21T15:06:01+00:00",
                metadata={"recovery": {"state": "interrupted", "partial_output": True}},
            ),
        ],
    )

    payload = build_resume_context_payload(
        session,
        user_prompt="continue now",
        config=ResumeContextConfig(recent_turn_limit=2),
    )

    assert payload is not None
    assert "Latest turn recovery: turn-2 ended interrupted." in payload.context_text
    assert "Resume should treat that turn as unfinished" in payload.context_text
    assert "Status: interrupted" in payload.context_text
    assert "The recorded assistant output may be partial." in payload.context_text
    assert payload.metadata["latest_recovery_turn_id"] == "turn-2"
    assert payload.metadata["latest_recovery_state"] == "interrupted"
    assert payload.metadata["latest_recovery_partial_output"] is True
    assert payload.metadata["unreliable_turn_ids"] == ["turn-2"]
    assert payload.metadata["partial_output_turn_ids"] == ["turn-2"]


def test_build_resume_context_payload_marks_failed_partial_output_as_unreliable():
    session = SessionRecord(
        id="session-resume",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            _turn("turn-1", "done prompt", "done output"),
            TurnRecord(
                id="turn-2",
                backend=BackendName.CODEX,
                prompt="latest prompt",
                output="partial failure output",
                status=TurnStatus.FAILED,
                started_at="2026-04-21T15:06:01+00:00",
                completed_at="2026-04-21T15:06:02+00:00",
                error=NormalizedError(kind="backend_error", message="backend failed"),
                metadata={
                    "recovery": {
                        "state": "failed",
                        "terminal_event_seen": True,
                        "partial_output": True,
                    }
                },
            ),
        ],
    )

    payload = build_resume_context_payload(
        session,
        user_prompt="continue now",
        config=ResumeContextConfig(recent_turn_limit=2),
    )

    assert payload is not None
    assert "Latest turn recovery: turn-2 ended failed." in payload.context_text
    assert "Assistant output (partial, not authoritative):" in payload.context_text
    assert "Do not treat the assistant output as an authoritative completion." in payload.context_text
    assert "Status: completed\nTask: task-main\nUser prompt:\ndone prompt" in payload.context_text
    assert payload.metadata["latest_recovery_turn_id"] == "turn-2"
    assert payload.metadata["latest_recovery_state"] == "failed"
    assert payload.metadata["latest_recovery_partial_output"] is True
    assert payload.metadata["unreliable_turn_ids"] == ["turn-2"]
    assert payload.metadata["partial_output_turn_ids"] == ["turn-2"]
