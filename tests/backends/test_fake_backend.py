from __future__ import annotations

from ccg_tui.backends import FakeBackendAdapter
from ccg_tui.models import EventType


def test_fake_backend_uses_default_test_labels(tmp_path) -> None:
    events = list(FakeBackendAdapter("codex").run("hello", tmp_path))

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.ACTIVITY,
        EventType.OUTPUT_STARTED,
        EventType.OUTPUT_DELTA,
        EventType.BACKEND_SUCCEEDED,
    ]
    assert events[1].text == "fake activity: hello"
    assert events[1].activity is not None
    assert events[1].activity["title"] == "Fake backend activity"
    assert events[1].activity["backend_label"] == "fake activity: hello"
    assert events[3].text == "fake reply to hello"


def test_fake_backend_allows_demo_labels_from_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CCG_TUI_FAKE_ACTIVITY_PREFIX", "backend activity:")
    monkeypatch.setenv("CCG_TUI_FAKE_ACTIVITY_TITLE", "Backend activity")
    monkeypatch.setenv("CCG_TUI_FAKE_REPLY_PREFIX", "demo reply:")

    events = list(FakeBackendAdapter("codex").run("hello", tmp_path))

    assert events[1].text == "backend activity: hello"
    assert events[1].activity is not None
    assert events[1].activity["title"] == "Backend activity"
    assert events[1].activity["backend_label"] == "backend activity: hello"
    assert events[3].text == "demo reply: hello"
