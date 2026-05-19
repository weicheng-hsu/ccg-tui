from ccg_tui.capabilities import (
    backend_capability_fact,
    backend_capability_facts,
    compare_permission_compatibility,
)
from ccg_tui.models import BackendName


REQUIRED_FACTS = {
    "model_flag_support",
    "persistent_session_support",
    "permission_concepts",
    "native_slash_passthrough",
    "summary_suitability",
    "handoff_suitability",
}


def test_structured_capability_facts_cover_required_dimensions():
    for backend in BackendName:
        facts = backend_capability_facts(backend)
        by_key = {fact.key: fact for fact in facts}

        assert set(by_key) == REQUIRED_FACTS
        assert all(fact.explanation for fact in facts)
        assert by_key["model_flag_support"].supported is True
        assert by_key["persistent_session_support"].supported is True
        assert by_key["native_slash_passthrough"].supported is True
        assert by_key["handoff_suitability"].supported is True


def test_unsupported_capability_returns_explanation():
    codex_summary = backend_capability_fact("codex", "summary_suitability")
    claude_summary = backend_capability_fact("claude", "summary_suitability")
    gemini_summary = backend_capability_fact("gemini", "summary_suitability")

    assert codex_summary.supported is False
    assert "Gemini-backed" in codex_summary.explanation
    assert claude_summary.supported is False
    assert "Gemini-backed" in claude_summary.explanation
    assert gemini_summary.supported is True


def test_unknown_capability_reports_valid_keys():
    try:
        backend_capability_fact("codex", "not-a-capability")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "unknown backend capability" in str(exc)
        assert "model_flag_support" in str(exc)


def test_permission_compatibility_explains_missing_sandbox_dimension():
    compatibility = compare_permission_compatibility(
        "codex",
        {"approval_policy": "on-request", "sandbox_mode": "workspace-write"},
        "claude",
    )

    assert compatibility.compatible is True
    assert compatibility.widens_permissions is False
    assert any("filesystem sandbox" in warning for warning in compatibility.warnings)


def test_permission_compatibility_covers_custom_unknown_target():
    compatibility = compare_permission_compatibility(
        "claude",
        {"permission_mode": "default"},
        "gemini",
        target_values={"approval_mode": "team-custom"},
    )

    assert compatibility.target_state.known is False
    assert any("target permission state is custom or unknown" == warning for warning in compatibility.warnings)
