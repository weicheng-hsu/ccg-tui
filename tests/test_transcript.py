import json

from ccg_tui.models import (
    BackendName,
    NormalizedError,
    RecordedEvent,
    RoutingDecision,
    RoutingDecisionRecord,
    SessionLineageRecord,
    SessionRecord,
    SummaryRecord,
    TurnRecord,
    TurnStatus,
    normalize_routing_decision,
)
from ccg_tui.transcript import (
    TranscriptStore,
    build_selected_turn_context,
    filter_transcript_turns,
    turn_has_partial_output,
)


def test_transcript_store_round_trips_session(tmp_path):
    store = TranscriptStore(tmp_path)
    session = SessionRecord(
        id="session-1",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            TurnRecord(
                id="turn-1",
                backend=BackendName.CODEX,
                prompt="hello",
                output="world",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:01+00:00",
                completed_at="2026-04-21T15:05:02+00:00",
                metadata={"resume_context": {"injected": True}},
            )
        ],
    )

    path = store.save_session(session)

    saved = json.loads(path.read_text())
    assert saved["backend"] == "codex"
    assert saved["turns"][0]["output"] == "world"

    loaded = store.load_session("session-1")
    assert loaded.id == session.id
    assert loaded.turns[0].status is TurnStatus.COMPLETED
    assert loaded.turns[0].metadata["resume_context"]["injected"] is True


def test_transcript_store_round_trips_recorded_events(tmp_path):
    store = TranscriptStore(tmp_path)
    session = SessionRecord(
        id="session-events",
        backend=BackendName.GEMINI,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            TurnRecord(
                id="turn-events",
                backend=BackendName.GEMINI,
                prompt="inspect",
                output="partial",
                status=TurnStatus.FAILED,
                started_at="2026-04-21T15:05:01+00:00",
                events=[
                    RecordedEvent(
                        type="activity",
                        observed_at="2026-04-21T15:05:02+00:00",
                        text="tools: read_file pyproject.toml",
                        session_id="vendor-1",
                        activity={
                            "kind": "tool",
                            "title": "Gemini tool call",
                            "status": "finished",
                            "details": {"path": "pyproject.toml"},
                        },
                        raw={"type": "message", "role": "assistant"},
                    ),
                    RecordedEvent(
                        type="backend_failed",
                        observed_at="2026-04-21T15:05:03+00:00",
                        error=NormalizedError(
                            kind="timeout",
                            message="timed out",
                            details={"source": "test"},
                        ),
                    ),
                ],
            )
        ],
    )

    store.save_session(session)
    loaded = store.load_session("session-events")
    events = loaded.turns[0].events

    assert events[0].activity["details"]["path"] == "pyproject.toml"
    assert events[0].raw == {"type": "message", "role": "assistant"}
    assert events[0].session_id == "vendor-1"
    assert events[1].error is not None
    assert events[1].error.kind == "timeout_error"
    assert events[1].error.details["original_kind"] == "timeout"
    assert events[1].error.details["source"] == "test"


def test_transcript_store_round_trips_routing_decision_audit(tmp_path):
    store = TranscriptStore(tmp_path)
    session = SessionRecord(
        id="session-routing",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        routing_decisions=[
            RoutingDecisionRecord(
                id="routing-1",
                recorded_at="2026-04-21T15:05:04+00:00",
                active_backend=BackendName.CODEX,
                suggested_backend=BackendName.CLAUDE,
                suggested_model="sonnet",
                trigger="manual_handoff",
                permission_state={
                    "backend": "codex",
                    "values": {
                        "approval_policy": "on-request",
                        "sandbox_mode": "workspace-write",
                    },
                    "preset_key": "ask",
                },
                user_decision="deferred",
                final_action="previewed",
                reason="manual handoff route inspected",
                compatibility={
                    "widens_permissions": False,
                    "target_state": {"values": {"permission_mode": "default"}},
                },
                metadata={"source_turn_ids": ["turn-1"]},
            )
        ],
    )

    path = store.save_session(session)
    saved = json.loads(path.read_text())

    assert saved["schema_version"] >= 4
    assert saved["routing_decisions"][0]["active_backend"] == "codex"
    assert saved["routing_decisions"][0]["suggested_backend"] == "claude"
    assert saved["routing_decisions"][0]["decision"] == "deferred"
    assert saved["routing_decisions"][0]["permission_state"]["preset_key"] == "ask"

    loaded = store.load_session("session-routing")
    decision = loaded.routing_decisions[0]
    assert decision.active_backend is BackendName.CODEX
    assert decision.suggested_backend is BackendName.CLAUDE
    assert decision.suggested_model == "sonnet"
    assert decision.trigger == "manual_handoff"
    assert decision.user_decision == "deferred"
    assert decision.decision == "deferred"
    assert decision.final_action == "previewed"
    assert decision.compatibility["widens_permissions"] is False
    assert decision.metadata["source_turn_ids"] == ["turn-1"]


def test_routing_decision_accepts_only_supported_decisions():
    assert normalize_routing_decision(RoutingDecision.CONFIRMED) == "confirmed"
    assert normalize_routing_decision("rejected") == "rejected"
    assert normalize_routing_decision("deferred") == "deferred"
    assert normalize_routing_decision("not_applicable") == "not_applicable"
    assert normalize_routing_decision("unknown") == "not_applicable"

    try:
        normalize_routing_decision("maybe")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unknown routing decision" in str(exc)


def test_transcript_store_preserves_multiple_routing_decision_order(tmp_path):
    store = TranscriptStore(tmp_path)
    session = SessionRecord(
        id="session-routing-order",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        routing_decisions=[
            RoutingDecisionRecord(
                id="routing-1",
                recorded_at="2026-04-21T15:05:04+00:00",
                active_backend=BackendName.CODEX,
                suggested_backend=BackendName.CLAUDE,
                trigger="manual_handoff",
                user_decision="deferred",
                final_action="previewed",
            ),
            RoutingDecisionRecord(
                id="routing-2",
                recorded_at="2026-04-21T15:05:05+00:00",
                active_backend=BackendName.CODEX,
                suggested_backend=BackendName.GEMINI,
                trigger="manual_handoff",
                user_decision="confirmed",
                final_action="handoff_session_started",
            ),
        ],
    )

    store.save_session(session)
    loaded = store.load_session("session-routing-order")

    assert [decision.id for decision in loaded.routing_decisions] == ["routing-1", "routing-2"]
    assert [decision.user_decision for decision in loaded.routing_decisions] == ["deferred", "confirmed"]


def test_transcript_store_round_trips_extended_context_collections(tmp_path):
    store = TranscriptStore(tmp_path)
    session = SessionRecord(
        id="session-ctx",
        backend=BackendName.CLAUDE,
        created_at="2026-04-21T15:05:00+00:00",
        workspace_cwd="/tmp/workspace",
    )

    path = store.save_session(session)
    saved = json.loads(path.read_text())

    assert saved["schema_version"] >= 2
    assert saved["workspace_cwd"] == "/tmp/workspace"
    assert saved["backend_sessions"][0]["backend"] == "claude"
    assert saved["tasks"][0]["id"] == "task-main"
    assert saved["summaries"] == []
    assert saved["artifacts"] == []
    assert saved["subagent_runs"] == []

    loaded = store.load_session("session-ctx")
    assert loaded.workspace_cwd == "/tmp/workspace"
    assert loaded.backend_sessions[0].backend is BackendName.CLAUDE
    assert loaded.tasks[0].id == "task-main"


def test_transcript_store_loads_legacy_session_without_routing_decisions(tmp_path):
    store = TranscriptStore(tmp_path)
    store.session_path("legacy-routing").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "id": "legacy-routing",
                "backend": "codex",
                "created_at": "2026-04-21T15:05:00+00:00",
                "updated_at": "2026-04-21T15:05:00+00:00",
                "workspace_cwd": "/tmp/workspace",
                "vendor_session_id": None,
                "backend_sessions": [],
                "tasks": [],
                "summaries": [],
                "artifacts": [],
                "subagent_runs": [],
                "lineage": {"kind": "root"},
                "turns": [],
            },
            indent=2,
        )
        + "\n"
    )

    loaded = store.load_session("legacy-routing")

    assert loaded.routing_decisions == []
    assert loaded.schema_version == 3


def test_transcript_store_round_trips_session_relationship_graph(tmp_path):
    store = TranscriptStore(tmp_path)
    session = SessionRecord(
        id="session-fork",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        lineage=SessionLineageRecord.for_fork("session-parent", forked_from_turn_id="turn-42"),
    )

    path = store.save_session(session)
    saved = json.loads(path.read_text())

    assert saved["schema_version"] >= 3
    assert saved["lineage"]["kind"] == "fork"
    assert saved["lineage"]["relationships"] == [
        {"kind": "parent", "session_id": "session-parent", "source_turn_id": "turn-42", "metadata": {}},
        {"kind": "fork", "session_id": "session-parent", "source_turn_id": "turn-42", "metadata": {}},
    ]

    loaded = store.load_session("session-fork")
    assert loaded.lineage.kind == "fork"
    assert [(relationship.kind, relationship.session_id) for relationship in loaded.lineage.relationships] == [
        ("parent", "session-parent"),
        ("fork", "session-parent"),
    ]
    assert loaded.lineage.relationships[0].source_turn_id == "turn-42"


def test_transcript_store_loads_legacy_handoff_lineage_without_relationships(tmp_path):
    store = TranscriptStore(tmp_path)
    path = store.session_path("legacy-handoff")
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "id": "legacy-handoff",
                "backend": "claude",
                "created_at": "2026-04-21T15:05:00+00:00",
                "updated_at": "2026-04-21T15:06:00+00:00",
                "workspace_cwd": "/tmp/workspace",
                "vendor_session_id": None,
                "backend_sessions": [],
                "tasks": [],
                "summaries": [],
                "artifacts": [],
                "subagent_runs": [],
                "lineage": {
                    "kind": "handoff",
                    "parent_session_id": "source-session",
                    "resumed_from_session_id": "source-session",
                    "forked_from_turn_id": "turn-source",
                },
                "turns": [],
            },
            indent=2,
        )
        + "\n"
    )

    loaded = store.load_session("legacy-handoff")

    assert loaded.lineage.kind == "handoff"
    assert loaded.lineage.parent_session_id == "source-session"
    assert loaded.lineage.resumed_from_session_id == "source-session"
    assert [(relationship.kind, relationship.session_id) for relationship in loaded.lineage.relationships] == [
        ("parent", "source-session"),
        ("handoff", "source-session"),
    ]
    assert [relationship.source_turn_id for relationship in loaded.lineage.relationships] == [
        "turn-source",
        "turn-source",
    ]


def test_transcript_store_lists_session_metadata_sorted_by_update_time(tmp_path):
    store = TranscriptStore(tmp_path)
    older = SessionRecord(
        id="session-old",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        updated_at="2026-04-21T15:06:00+00:00",
        workspace_cwd="/tmp/older-workspace",
        turns=[
            TurnRecord(
                id="turn-old",
                backend=BackendName.CODEX,
                prompt="hello",
                output="world",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:01+00:00",
                completed_at="2026-04-21T15:05:02+00:00",
            )
        ],
    )
    newer = SessionRecord(
        id="session-new",
        backend=BackendName.GEMINI,
        created_at="2026-04-21T15:07:00+00:00",
        updated_at="2026-04-21T15:08:00+00:00",
        workspace_cwd="/tmp/new-workspace",
        summaries=[
            SummaryRecord(
                id="summary-1",
                scope="task:task-main",
                created_at="2026-04-21T15:08:00+00:00",
                text="summary",
            )
        ],
    )
    store.save_session(older)
    store.save_session(newer)

    sessions = store.list_sessions()

    assert [session.id for session in sessions] == ["session-new", "session-old"]
    assert sessions[0].backend == "gemini"
    assert sessions[0].turn_count == 0
    assert sessions[0].summary_count == 1
    assert sessions[0].latest_status == "idle"
    assert sessions[0].resumable is False
    assert sessions[0].workspace_basename == "new-workspace"
    assert sessions[1].turn_count == 1
    assert sessions[1].latest_status == "completed"
    assert sessions[1].resumable is True


def test_transcript_store_loads_incomplete_latest_turn_without_corruption(tmp_path):
    store = TranscriptStore(tmp_path)
    session = SessionRecord(
        id="session-recovery",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            TurnRecord(
                id="turn-incomplete",
                backend=BackendName.CODEX,
                prompt="resume me",
                output="partial output",
                status=TurnStatus.STREAMING,
                started_at="2026-04-21T15:05:01+00:00",
                metadata={"recovery": {"state": "interrupted", "partial_output": True}},
            )
        ],
    )
    store.save_session(session)

    loaded = store.load_session("session-recovery")
    sessions = store.list_sessions()

    assert loaded.turns[0].status is TurnStatus.STREAMING
    assert loaded.turns[0].metadata["recovery"]["state"] == "interrupted"
    assert loaded.turns[0].output == "partial output"
    assert sessions[0].latest_status == "interrupted"
    assert sessions[0].resumable is True
    assert turn_has_partial_output(loaded.turns[0]) is True


def test_turn_has_partial_output_falls_back_to_recorded_output_when_metadata_is_stale():
    turn = TurnRecord(
        id="turn-stale",
        backend=BackendName.CODEX,
        prompt="resume me",
        output="partial output",
        status=TurnStatus.FAILED,
        started_at="2026-04-21T15:05:01+00:00",
        metadata={"recovery": {"state": "interrupted", "partial_output": False}},
    )

    assert turn_has_partial_output(turn) is True


def test_transcript_store_related_sessions_returns_parent_child_and_specialized_kinds(tmp_path):
    store = TranscriptStore(tmp_path)
    parent = SessionRecord(
        id="session-parent",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
    )
    child = SessionRecord(
        id="session-child",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:01+00:00",
        lineage=SessionLineageRecord.for_child("session-parent", forked_from_turn_id="turn-root"),
    )
    fork = SessionRecord(
        id="session-fork",
        backend=BackendName.CLAUDE,
        created_at="2026-04-21T15:05:02+00:00",
        lineage=SessionLineageRecord.for_fork("session-parent", forked_from_turn_id="turn-root"),
    )
    handoff = SessionRecord(
        id="session-handoff",
        backend=BackendName.GEMINI,
        created_at="2026-04-21T15:05:03+00:00",
        lineage=SessionLineageRecord.for_handoff("session-parent", forked_from_turn_id="turn-handoff"),
    )
    delegated = SessionRecord(
        id="session-delegated",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:04+00:00",
        lineage=SessionLineageRecord.for_delegated("session-parent", forked_from_turn_id="turn-delegated"),
    )

    for session in [parent, child, fork, handoff, delegated]:
        store.save_session(session)

    parent_related = store.related_sessions("session-parent")
    assert [(item.session_id, item.relationship_kinds) for item in parent_related] == [
        ("session-child", ("child",)),
        ("session-delegated", ("child", "delegated")),
        ("session-fork", ("child", "fork")),
        ("session-handoff", ("child", "handoff")),
    ]
    assert [item.source_turn_ids for item in parent_related] == [
        ("turn-root",),
        ("turn-delegated",),
        ("turn-root",),
        ("turn-handoff",),
    ]

    handoff_related = store.related_sessions("session-handoff")
    assert [(item.session_id, item.relationship_kinds) for item in handoff_related] == [
        ("session-parent", ("parent", "handoff")),
    ]
    assert handoff_related[0].lineage_kind == "root"
    assert handoff_related[0].source_turn_ids == ("turn-handoff",)

    assert [item.session_id for item in store.related_sessions("session-parent", relationship_kinds=["handoff"])] == [
        "session-handoff"
    ]


def test_filter_transcript_turns_supports_task_status_turn_ids_and_recent_count():
    session = SessionRecord(
        id="session-filter",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            TurnRecord(
                id="turn-1",
                backend=BackendName.CODEX,
                prompt="p1",
                output="o1",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:01+00:00",
                task_id="task-alpha",
            ),
            TurnRecord(
                id="turn-2",
                backend=BackendName.CODEX,
                prompt="p2",
                output="o2",
                status=TurnStatus.FAILED,
                started_at="2026-04-21T15:05:02+00:00",
                task_id="task-beta",
                metadata={"recovery": {"state": "interrupted", "partial_output": True}},
            ),
            TurnRecord(
                id="turn-3",
                backend=BackendName.CODEX,
                prompt="p3",
                output="o3",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:03+00:00",
                task_id="task-alpha",
            ),
        ],
    )

    assert [turn.id for turn in filter_transcript_turns(session, task_id="task-alpha")] == [
        "turn-1",
        "turn-3",
    ]
    assert [turn.id for turn in filter_transcript_turns(session, statuses=["failed"])] == ["turn-2"]
    assert [turn.id for turn in filter_transcript_turns(session, statuses=["interrupted"])] == ["turn-2"]
    assert [turn.id for turn in filter_transcript_turns(session, turn_ids=["turn-3", "turn-1"])] == [
        "turn-3",
        "turn-1",
    ]
    assert [turn.id for turn in filter_transcript_turns(session, recent_count=2)] == ["turn-2", "turn-3"]


def test_filter_transcript_turns_reports_missing_or_invalid_constraints():
    session = SessionRecord(
        id="session-filter-errors",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            TurnRecord(
                id="turn-1",
                backend=BackendName.CODEX,
                prompt="p1",
                output="o1",
                status=TurnStatus.COMPLETED,
                started_at="2026-04-21T15:05:01+00:00",
                task_id="task-main",
            )
        ],
    )

    try:
        filter_transcript_turns(session, turn_ids=["turn-missing"])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unknown turn id(s): turn-missing" in str(exc)

    try:
        filter_transcript_turns(session, statuses=["not-a-status"])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unknown turn status: 'not-a-status'" in str(exc)

    try:
        filter_transcript_turns(session, task_id="task-other", turn_ids=["turn-1"])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "excluded by active filters: turn-1" in str(exc)


def test_build_selected_turn_context_returns_source_ids_and_char_counts():
    turns = [
        TurnRecord(
            id="turn-1",
            backend=BackendName.CODEX,
            prompt="hello",
            output="world",
            status=TurnStatus.COMPLETED,
            started_at="2026-04-21T15:05:01+00:00",
        ),
        TurnRecord(
            id="turn-2",
            backend=BackendName.CODEX,
            prompt="abc",
            output="12345",
            status=TurnStatus.COMPLETED,
            started_at="2026-04-21T15:05:02+00:00",
        ),
    ]

    context = build_selected_turn_context(turns, rendered_turns=["x" * 10, "y" * 7])

    assert context.source_turn_ids == ("turn-1", "turn-2")
    assert context.turn_count == 2
    assert context.prompt_char_count == 8
    assert context.output_char_count == 10
    assert context.rendered_char_count == 17
