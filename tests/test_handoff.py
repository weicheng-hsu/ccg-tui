from ccg_tui.app import format_handoff_packet
from ccg_tui.handoff import (
    DelegatedContextPacket,
    HandoffConfig,
    build_delegated_context_packet,
    build_delegated_result_payload,
    build_handoff_packet,
    build_handoff_selected_context,
)
from ccg_tui.models import (
    ArtifactRecord,
    BackendName,
    EventType,
    NormalizedError,
    RecordedEvent,
    SessionLineageRecord,
    SessionRecord,
    SummaryRecord,
    TurnRecord,
    TurnStatus,
)


def _turn(
    turn_id,
    prompt,
    output,
    *,
    status=TurnStatus.COMPLETED,
    error=None,
    events=None,
    task_id="task-main",
):
    return TurnRecord(
        id=turn_id,
        backend=BackendName.CODEX,
        prompt=prompt,
        output=output,
        status=status,
        started_at="2026-04-21T15:05:01+00:00",
        completed_at="2026-04-21T15:05:02+00:00",
        task_id=task_id,
        error=error,
        events=events or [],
    )


def test_build_handoff_packet_uses_latest_summary_recent_turns_and_target_metadata():
    session = SessionRecord(
        id="session-handoff",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd="/tmp/workspace",
        summaries=[
            SummaryRecord(
                id="summary-old",
                scope="session",
                created_at="2026-04-21T15:05:03+00:00",
                text="Old summary",
            ),
            SummaryRecord(
                id="summary-new",
                scope="session",
                created_at="2026-04-21T15:06:03+00:00",
                text="Latest summary",
                kind="session_checkpoint",
            ),
        ],
        turns=[
            _turn("turn-1", "old prompt", "old output"),
            _turn("turn-2", "new prompt", "new output"),
        ],
    )

    packet = build_handoff_packet(
        session,
        target_backend=BackendName.CLAUDE,
        target_model="sonnet",
        user_goal="continue the implementation",
        config=HandoffConfig(recent_turn_limit=1),
    )

    assert "CCG MANUAL HANDOFF PACKET" in packet.context_text
    assert "not a vendor-native resume" in packet.context_text
    assert "\n" not in packet.backend_prompt
    assert "Latest summary" in packet.backend_prompt
    assert "new prompt" in packet.backend_prompt
    assert "old prompt" not in packet.backend_prompt
    assert 'Current user goal JSON: "continue the implementation"' in packet.backend_prompt
    assert packet.metadata["source_session_id"] == "session-handoff"
    assert packet.metadata["source_backend"] == "codex"
    assert packet.metadata["target_backend"] == "claude"
    assert packet.metadata["target_model"] == "sonnet"
    assert packet.metadata["source_scope"] == "session"
    assert packet.metadata["source_task_id"] is None
    assert packet.metadata["source_summary_id"] == "summary-new"
    assert packet.metadata["source_turn_ids"] == ["turn-2"]
    assert packet.metadata["selected_context"]["source_turn_ids"] == ["turn-2"]
    assert packet.metadata["selected_context"]["turn_count"] == 1
    assert packet.metadata["context_char_count"] == len(packet.context_text)


def test_build_handoff_packet_session_scope_ignores_task_scoped_summary():
    session = SessionRecord(
        id="session-summary-scope",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        summaries=[
            SummaryRecord(
                id="summary-session",
                scope="session",
                created_at="2026-04-21T15:05:02+00:00",
                text="Session summary",
            ),
            SummaryRecord(
                id="summary-task",
                scope="task:task-alpha",
                created_at="2026-04-21T15:05:03+00:00",
                text="Task summary should not leak",
            ),
        ],
        turns=[
            _turn("turn-main", "main prompt", "main output"),
            _turn("turn-alpha", "alpha prompt", "alpha output", task_id="task-alpha"),
        ],
    )

    packet = build_handoff_packet(session, target_backend=BackendName.CLAUDE)

    assert "Session summary" in packet.context_text
    assert "Task summary should not leak" not in packet.context_text
    assert packet.metadata["source_summary_id"] == "summary-session"
    assert packet.metadata["source_summary_scope"] == "session"


def test_build_handoff_packet_can_anchor_to_selected_task_boundary():
    session = SessionRecord(
        id="session-task-handoff",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd="/tmp/workspace",
        summaries=[
            SummaryRecord(
                id="summary-session",
                scope="session",
                created_at="2026-04-21T15:05:02+00:00",
                text="Session summary",
            ),
            SummaryRecord(
                id="summary-task-alpha",
                scope="task:task-alpha",
                created_at="2026-04-21T15:05:03+00:00",
                text="Alpha summary",
                kind="task_checkpoint",
            ),
        ],
        turns=[
            _turn("turn-main", "main prompt", "main output"),
            _turn("turn-alpha-1", "alpha prompt 1", "alpha output 1", task_id="task-alpha"),
            _turn("turn-beta", "beta prompt", "beta output", task_id="task-beta"),
            _turn("turn-alpha-2", "alpha prompt 2", "alpha output 2", task_id="task-alpha"),
        ],
    )

    packet = build_handoff_packet(
        session,
        target_backend=BackendName.CLAUDE,
        target_model="sonnet",
        user_goal="continue alpha",
        scope="task",
        task_id="task-alpha",
        config=HandoffConfig(recent_turn_limit=5),
    )

    assert "source_scope: task:task-alpha" in packet.context_text
    assert "source_task_id: task-alpha" in packet.context_text
    assert "Alpha summary" in packet.context_text
    assert "alpha prompt 1" in packet.context_text
    assert "alpha prompt 2" in packet.context_text
    assert "main prompt" not in packet.context_text
    assert "beta prompt" not in packet.context_text
    assert packet.metadata["source_scope"] == "task:task-alpha"
    assert packet.metadata["source_task_id"] == "task-alpha"
    assert packet.metadata["source_summary_id"] == "summary-task-alpha"
    assert packet.metadata["source_summary_scope"] == "task:task-alpha"
    assert packet.metadata["source_turn_ids"] == ["turn-alpha-1", "turn-alpha-2"]


def test_build_handoff_packet_without_summary_records_recent_turns():
    session = SessionRecord(
        id="session-no-summary",
        backend=BackendName.GEMINI,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[_turn("turn-1", "prompt", "output")],
    )

    packet = build_handoff_packet(
        session,
        target_backend=BackendName.CODEX,
        user_goal="pick up the work",
    )

    assert "Latest summary checkpoint: <none>" in packet.context_text
    assert "prompt" in packet.context_text
    assert packet.metadata["source_summary_id"] is None
    assert packet.metadata["source_turn_ids"] == ["turn-1"]


def test_build_delegated_context_packet_uses_curated_parent_context_and_explicit_child_settings():
    session = SessionRecord(
        id="session-parent",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd="/tmp/workspace",
        summaries=[
            SummaryRecord(
                id="summary-parent",
                scope="task:task-alpha",
                created_at="2026-04-21T15:05:03+00:00",
                text="Alpha checkpoint",
                kind="task_checkpoint",
            )
        ],
        turns=[
            _turn("turn-main", "main prompt", "main output"),
            _turn("turn-alpha-1", "alpha prompt 1", "alpha output 1", task_id="task-alpha"),
            _turn("turn-alpha-2", "alpha prompt 2", "alpha output 2", task_id="task-alpha"),
        ],
    )

    packet = build_delegated_context_packet(
        session,
        target_backend=BackendName.CLAUDE,
        target_model="sonnet",
        permission_mode="workspace-write",
        delegate_goal="verify the alpha parser",
        scope="task",
        task_id="task-alpha",
        turn_ids=["turn-alpha-2"],
    )

    assert isinstance(packet, DelegatedContextPacket)
    assert "CCG DELEGATED CONTEXT PACKET" in packet.context_text
    assert "delegated parent context" in packet.context_text
    assert "not a vendor-native continuation" in packet.context_text
    assert "\n" not in packet.backend_prompt
    assert 'Permission mode JSON: "workspace-write"' in packet.backend_prompt
    assert packet.metadata["parent_session_id"] == "session-parent"
    assert packet.metadata["parent_backend"] == "codex"
    assert packet.metadata["child_backend"] == "claude"
    assert packet.metadata["child_model"] == "sonnet"
    assert packet.metadata["permission_mode"] == "workspace-write"
    assert packet.metadata["source_scope"] == "task:task-alpha"
    assert packet.metadata["source_summary_id"] == "summary-parent"
    assert packet.metadata["source_turn_ids"] == ["turn-alpha-2"]
    assert packet.metadata["selection_criteria"]["turn_ids"] == ["turn-alpha-2"]
    assert packet.metadata["forked_from_turn_id"] == "turn-alpha-2"


def test_build_delegated_result_payload_records_child_references():
    session = SessionRecord(
        id="session-child",
        backend=BackendName.CLAUDE,
        created_at="2026-04-21T15:05:00+00:00",
        summaries=[
            SummaryRecord(
                id="summary-child",
                scope="task:task-main",
                created_at="2026-04-21T15:05:03+00:00",
                text="Child checkpoint",
                kind="task_checkpoint",
            )
        ],
        artifacts=[
            ArtifactRecord(
                id="artifact-log",
                kind="file",
                created_at="2026-04-21T15:05:04+00:00",
                path="logs/pytest.txt",
            )
        ],
        turns=[
            TurnRecord(
                id="turn-child",
                backend=BackendName.CLAUDE,
                prompt="verify parser",
                output="Parser verified",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:01+00:00",
                completed_at="2026-04-21T15:05:02+00:00",
                metadata={
                    "delegated_context": {
                        "injected": True,
                        "parent_session_id": "session-parent",
                        "child_model": "sonnet",
                        "permission_mode": "read-only",
                        "source_summary_id": "summary-parent",
                        "source_turn_ids": ["turn-parent"],
                        "selection_criteria": {"scope": "task:task-main", "recent_turn_limit": 1},
                    }
                },
            )
        ],
        lineage=SessionLineageRecord.for_delegated("session-parent", forked_from_turn_id="turn-parent"),
    )

    payload = build_delegated_result_payload(
        session,
        result_text="  Parser constraints verified; logs attached.  ",
        summary_id="summary-child",
        turn_id="turn-child",
        artifact_ids=["artifact-log"],
    )

    assert payload.result_text == "Parser constraints verified; logs attached."
    assert payload.metadata["mode"] == "delegated_result"
    assert payload.metadata["delegated_session_id"] == "session-child"
    assert payload.metadata["parent_session_id"] == "session-parent"
    assert payload.metadata["delegated_backend"] == "claude"
    assert payload.metadata["delegated_model"] == "sonnet"
    assert payload.metadata["permission_mode"] == "read-only"
    assert payload.metadata["source_summary_id"] == "summary-parent"
    assert payload.metadata["source_turn_ids"] == ["turn-parent"]
    assert payload.metadata["delegated_summary_id"] == "summary-child"
    assert payload.metadata["delegated_turn_id"] == "turn-child"
    assert payload.metadata["delegated_artifact_ids"] == ["artifact-log"]


def test_build_handoff_packet_applies_turn_summary_and_total_limits():
    session = SessionRecord(
        id="session-limits",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        summaries=[
            SummaryRecord(
                id="summary-long",
                scope="session",
                created_at="2026-04-21T15:05:03+00:00",
                text="summary " * 100,
            )
        ],
        turns=[
            _turn("turn-old", "old prompt", "old output " * 120),
            _turn("turn-mid", "mid prompt", "mid output " * 120),
            _turn("turn-new", "new prompt", "new output " * 120),
        ],
    )

    packet = build_handoff_packet(
        session,
        target_backend=BackendName.CLAUDE,
        user_goal="continue",
        config=HandoffConfig(
            recent_turn_limit=3,
            max_context_chars=1_400,
            max_turn_chars=280,
            max_summary_chars=120,
        ),
    )

    assert len(packet.context_text) <= 1_400
    assert "turn-new" in packet.metadata["source_turn_ids"]
    assert "turn-old" not in packet.metadata["source_turn_ids"]
    assert "new prompt" in packet.context_text
    assert "[truncated" in packet.context_text
    assert packet.metadata["source_summary_id"] == "summary-long"


def test_build_handoff_packet_includes_failed_turn_error_and_activity():
    session = SessionRecord(
        id="session-failed",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            _turn(
                "turn-failed",
                "run tests",
                "",
                status=TurnStatus.FAILED,
                error=NormalizedError(kind="backend_failed", message="pytest failed", exit_code=1),
                events=[
                    RecordedEvent(
                        type=EventType.ACTIVITY.value,
                        observed_at="2026-04-21T15:05:02+00:00",
                        text="tool: uv run pytest",
                    )
                ],
            )
        ],
    )

    packet = build_handoff_packet(
        session,
        target_backend=BackendName.GEMINI,
        user_goal="diagnose failure",
    )

    assert "Status: failed" in packet.context_text
    assert "tool: uv run pytest" in packet.context_text
    assert "pytest failed" in packet.context_text
    assert packet.metadata["source_turn_ids"] == ["turn-failed"]


def test_build_handoff_packet_renders_interrupted_turn_as_unfinished():
    interrupted = TurnRecord(
        id="turn-interrupted",
        backend=BackendName.CODEX,
        prompt="continue",
        output="partial answer",
        status=TurnStatus.FAILED,
        started_at="2026-04-21T15:05:01+00:00",
        completed_at="2026-04-21T15:05:02+00:00",
        error=NormalizedError(kind="interrupted", message="no terminal event"),
        metadata={"recovery": {"state": "interrupted", "partial_output": True}},
    )
    session = SessionRecord(
        id="session-interrupted",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[interrupted],
    )

    packet = build_handoff_packet(
        session,
        target_backend=BackendName.GEMINI,
        user_goal="continue safely",
        statuses=["interrupted"],
    )

    assert "Status: interrupted" in packet.context_text
    assert "Assistant output (partial, not authoritative):" in packet.context_text
    assert "Treat this source turn as unfinished." in packet.context_text
    assert packet.metadata["source_turn_ids"] == ["turn-interrupted"]


def test_build_handoff_packet_does_not_mutate_source_session():
    session = SessionRecord(
        id="session-immutable",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd="/tmp/workspace",
        summaries=[
            SummaryRecord(
                id="summary-immutable",
                scope="session",
                created_at="2026-04-21T15:05:03+00:00",
                text="Stable summary",
            )
        ],
        turns=[
            _turn("turn-1", "first prompt", "first output"),
            _turn("turn-2", "second prompt", "second output"),
        ],
    )
    before = session.to_dict()

    packet = build_handoff_packet(
        session,
        target_backend=BackendName.CLAUDE,
        target_model="sonnet",
        user_goal="continue without mutation",
        config=HandoffConfig(recent_turn_limit=1),
    )

    assert packet.metadata["source_turn_ids"] == ["turn-2"]
    assert packet.metadata["source_summary_id"] == "summary-immutable"
    assert session.to_dict() == before


def test_build_handoff_packet_handles_empty_session():
    session = SessionRecord(
        id="session-empty",
        backend=BackendName.CLAUDE,
        created_at="2026-04-21T15:05:00+00:00",
    )

    packet = build_handoff_packet(
        session,
        target_backend=BackendName.CODEX,
        user_goal="start from scratch with lineage",
    )

    assert "Recent source turns:\n<none>" in packet.context_text
    assert packet.metadata["source_turn_ids"] == []


def test_build_handoff_selected_context_supports_explicit_turn_ids_and_status():
    session = SessionRecord(
        id="session-selected-context",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            _turn("turn-1", "main prompt", "main output", status=TurnStatus.COMPLETED),
            _turn("turn-2", "failed prompt", "failed output", status=TurnStatus.FAILED),
            _turn("turn-3", "main prompt 2", "main output 2", status=TurnStatus.COMPLETED),
        ],
    )

    selected = build_handoff_selected_context(
        session,
        turn_ids=["turn-3", "turn-1"],
        statuses=["completed"],
        recent_turn_limit=5,
    )

    assert [turn.id for turn in selected.turns] == ["turn-3", "turn-1"]
    assert selected.selected_context.source_turn_ids == ("turn-3", "turn-1")
    assert selected.selected_context.turn_count == 2


def test_build_handoff_selected_context_rejects_unknown_turn_ids():
    session = SessionRecord(
        id="session-selected-context-errors",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[_turn("turn-1", "prompt", "output")],
    )

    try:
        build_handoff_selected_context(session, turn_ids=["turn-missing"])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unknown turn id(s): turn-missing" in str(exc)


def test_build_handoff_packet_records_selection_criteria_metadata():
    session = SessionRecord(
        id="session-selection-criteria",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            _turn("turn-1", "prompt 1", "output 1", status=TurnStatus.COMPLETED),
            _turn("turn-2", "prompt 2", "output 2", status=TurnStatus.FAILED, task_id="task-a"),
            _turn("turn-3", "prompt 3", "output 3", status=TurnStatus.COMPLETED, task_id="task-a"),
        ],
    )

    packet = build_handoff_packet(
        session,
        target_backend=BackendName.CLAUDE,
        scope="task",
        task_id="task-a",
        turn_ids=["turn-3"],
        statuses=["completed"],
        recent_turn_limit=1,
    )

    criteria = packet.metadata["selection_criteria"]
    assert criteria["scope"] == "task:task-a"
    assert criteria["task_id"] == "task-a"
    assert criteria["turn_ids"] == ["turn-3"]
    assert criteria["statuses"] == ["completed"]
    assert criteria["recent_turn_limit"] == 1
    assert packet.metadata["recent_turn_limit"] == 1


def test_build_handoff_packet_records_audit_metadata_for_selected_context():
    session = SessionRecord(
        id="session-audit",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            _turn("turn-1", "keep one", "A" * 900),
            _turn("turn-2", "drop by status", "B" * 900, status=TurnStatus.FAILED),
            _turn("turn-3", "keep three", "C" * 900),
            _turn("turn-4", "keep four", "D" * 900),
        ],
    )

    packet = build_handoff_packet(
        session,
        target_backend=BackendName.CLAUDE,
        statuses=["completed"],
        recent_turn_limit=3,
        config=HandoffConfig(
            recent_turn_limit=3,
            max_context_chars=700,
            max_turn_chars=180,
            max_summary_chars=80,
        ),
    )

    audit = packet.metadata["audit"]

    assert audit["turns"]["included_source_ids"] == packet.metadata["source_turn_ids"]
    assert audit["turns"]["selected_before_recent_limit_source_ids"] == ["turn-1", "turn-3", "turn-4"]
    assert audit["turns"]["selected_before_context_limit_source_ids"]
    assert any(
        item["source_id"] == "turn-2" and item["reason"] == "filtered_by_criteria"
        for item in audit["turns"]["excluded_source_ids"]
    )
    assert audit["summary"]["source_id"] is None
    assert audit["summary"]["exclusion_reason"] == "summary_absent"
    assert audit["truncation"]["dropped_for_recent_limit_count"] == 0
    assert audit["truncation"]["dropped_for_context_limit_count"] > 0
    assert audit["truncation"]["context_text_clipped"] is True

    rendered = format_handoff_packet(packet)
    assert "Audit" in rendered
    assert "Turns included :" in rendered
    assert "Turn exclusions :" in rendered
    assert "filtered_by_criteria" in rendered
    assert "summary_absent" in rendered
    assert "context_drop=" in rendered
