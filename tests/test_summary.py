from ccg_tui.models import BackendEvent, BackendName, EventType, SessionRecord, SummaryRecord, TurnRecord, TurnStatus
from ccg_tui.session import SessionController
from ccg_tui.summary import (
    SummaryGenerationError,
    build_summary_prompt,
    collect_summary_text,
    generate_and_persist_summary,
    generate_summary_record,
)
from ccg_tui.transcript import TranscriptStore


class FakeSummaryAdapter:
    name = BackendName.GEMINI

    def __init__(self, text="## Goal\nShip summary support.", events=None):
        self.text = text
        self.events = events
        self.prompts = []
        self.closed = False

    def run(self, prompt, cwd):
        self.prompts.append(prompt)
        if self.events is not None:
            yield from self.events
            return
        yield BackendEvent(type=EventType.OUTPUT_STARTED)
        yield BackendEvent(type=EventType.OUTPUT_DELTA, text=self.text)
        yield BackendEvent(type=EventType.BACKEND_SUCCEEDED)

    def close(self):
        self.closed = True


class FakeWorkAdapter:
    name = BackendName.CODEX

    def run(self, prompt, cwd):
        yield BackendEvent(type=EventType.OUTPUT_STARTED)
        yield BackendEvent(type=EventType.OUTPUT_DELTA, text="Implemented phase 1.")
        yield BackendEvent(type=EventType.BACKEND_SUCCEEDED)

    def close(self):
        return None


def _turn(turn_id, prompt, output, task_id="task-main"):
    return TurnRecord(
        id=turn_id,
        backend=BackendName.CODEX,
        prompt=prompt,
        output=output,
        status=TurnStatus.COMPLETED,
        started_at="2026-04-21T15:05:01+00:00",
        completed_at="2026-04-21T15:05:02+00:00",
        task_id=task_id,
    )


def test_build_summary_prompt_uses_task_scope_recent_turns_and_prior_summary():
    session = SessionRecord(
        id="session-1",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        summaries=[
            SummaryRecord(
                id="summary-old",
                scope="task:task-main",
                created_at="2026-04-21T15:05:03+00:00",
                text="Previous decision: do not summarize every turn.",
            )
        ],
        turns=[
            _turn("turn-1", "old main", "old output"),
            _turn("turn-2", "other task", "ignore me", task_id="task-other"),
            _turn("turn-3", "new main", "new output"),
        ],
    )

    prompt, source_turn_ids = build_summary_prompt(session, scope="task", recent_turn_limit=1)

    assert source_turn_ids == ["turn-3"]
    assert "Previous decision: do not summarize every turn." in prompt
    assert "new main" in prompt
    assert "old main" not in prompt
    assert "other task" not in prompt
    assert "## Next Steps" in prompt


def test_build_summary_prompt_preserves_newest_turns_when_clipped():
    session = SessionRecord(
        id="session-clip",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[
            _turn("turn-old", "old prompt", "old output " * 80),
            _turn("turn-mid", "mid prompt", "mid output " * 80),
            _turn("turn-new", "new prompt", "new output " * 80),
        ],
    )

    prompt, source_turn_ids = build_summary_prompt(
        session,
        scope="task",
        recent_turn_limit=3,
        max_prompt_chars=1_300,
        max_turn_chars=600,
    )

    assert "turn-new" in source_turn_ids
    assert "turn-old" not in source_turn_ids
    assert "new prompt" in prompt


def test_build_summary_prompt_clips_prior_summary_before_recent_turns():
    session = SessionRecord(
        id="session-prior-clip",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        summaries=[
            SummaryRecord(
                id="summary-long",
                scope="task:task-main",
                created_at="2026-04-21T15:05:03+00:00",
                text="old summary " * 300,
            )
        ],
        turns=[_turn("turn-new", "latest important prompt", "latest important output")],
    )

    prompt, source_turn_ids = build_summary_prompt(
        session,
        scope="task",
        max_prompt_chars=1_500,
        max_turn_chars=400,
    )

    assert source_turn_ids == ["turn-new"]
    assert "latest important prompt" in prompt
    assert "latest important output" in prompt


def test_collect_summary_text_raises_on_backend_failure():
    events = [
        BackendEvent(type=EventType.BACKEND_FAILED, error=None),
    ]

    try:
        collect_summary_text(events)
    except SummaryGenerationError as exc:
        assert "failed" in str(exc).lower()
    else:
        raise AssertionError("expected summary failure")


def test_generate_summary_record_returns_gemini_metadata(tmp_path):
    session = SessionRecord(
        id="session-2",
        backend=BackendName.CLAUDE,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[_turn("turn-1", "summarize this", "summary-worthy output")],
    )
    adapter = FakeSummaryAdapter(text="## Goal\nKeep context portable.")

    summary = generate_summary_record(session, adapter=adapter, cwd=tmp_path)

    assert summary.scope == "task:task-main"
    assert summary.kind == "task_checkpoint"
    assert summary.text == "## Goal\nKeep context portable."
    assert summary.source_turn_ids == ["turn-1"]
    assert summary.metadata["backend"] == "gemini"
    assert summary.metadata["source_turn_count"] == 1
    assert summary.metadata["summary_scope"] == "task:task-main"
    assert summary.metadata["task_id"] == "task-main"
    assert summary.metadata["source_summary_id"] is None
    assert "summary-worthy output" in adapter.prompts[0]


def test_generate_summary_record_supports_dynamic_task_id_and_prior_summary_metadata(tmp_path):
    session = SessionRecord(
        id="session-task-dynamic",
        backend=BackendName.CLAUDE,
        created_at="2026-04-21T15:05:00+00:00",
        summaries=[
            SummaryRecord(
                id="summary-task-alpha-old",
                scope="task:task-alpha",
                created_at="2026-04-21T15:05:03+00:00",
                text="Previous alpha summary",
            ),
            SummaryRecord(
                id="summary-session",
                scope="session",
                created_at="2026-04-21T15:05:04+00:00",
                text="Session summary should not be reused for task scope",
            ),
        ],
        turns=[
            _turn("turn-main", "main prompt", "main output"),
            _turn("turn-alpha-1", "alpha prompt 1", "alpha output 1", task_id="task-alpha"),
            _turn("turn-alpha-2", "alpha prompt 2", "alpha output 2", task_id="task-alpha"),
        ],
    )

    summary = generate_summary_record(
        session,
        adapter=FakeSummaryAdapter(text="## Goal\nTrack alpha."),
        cwd=tmp_path,
        scope="task",
        task_id="task-alpha",
        recent_turn_limit=5,
    )

    assert summary.scope == "task:task-alpha"
    assert summary.source_turn_ids == ["turn-alpha-1", "turn-alpha-2"]
    assert summary.metadata["summary_scope"] == "task:task-alpha"
    assert summary.metadata["task_id"] == "task-alpha"
    assert summary.metadata["source_summary_id"] == "summary-task-alpha-old"


def test_session_controller_generates_and_persists_summary(tmp_path):
    controller = SessionController(adapter=FakeWorkAdapter(), store=TranscriptStore(tmp_path), cwd=tmp_path)
    controller.submit_prompt("ship phase 1")
    summary_adapter = FakeSummaryAdapter(text="## Goal\nResume safely.")

    summary = controller.generate_summary(summary_adapter)
    loaded = controller.store.load_session(controller.session.id)

    assert summary.id.startswith("summary-")
    assert loaded.summaries[0].id == summary.id
    assert loaded.summaries[0].text == "## Goal\nResume safely."
    assert loaded.summaries[0].metadata["backend"] == "gemini"


def test_generate_and_persist_summary_centralizes_save_semantics(tmp_path):
    session = SessionRecord(
        id="session-persist",
        backend=BackendName.CODEX,
        created_at="2026-04-21T15:05:00+00:00",
        turns=[_turn("turn-1", "persist this", "persisted output")],
    )
    store = TranscriptStore(tmp_path)

    summary = generate_and_persist_summary(
        session,
        adapter=FakeSummaryAdapter(text="## Goal\nPersist once."),
        cwd=tmp_path,
        save_session=store.save_session,
    )
    loaded = store.load_session(session.id)

    assert loaded.summaries[0].id == summary.id
    assert loaded.updated_at == summary.created_at
