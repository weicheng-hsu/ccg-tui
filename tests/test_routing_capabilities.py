from ccg_tui.models import BackendName
from ccg_tui.routing_capabilities import (
    all_backend_capabilities,
    compare_permission_compatibility,
    permission_state_for_backend,
    permission_values_for_backend,
)


def test_capability_registry_covers_all_supported_backends():
    profiles = all_backend_capabilities()

    assert {profile.backend for profile in profiles} == {
        BackendName.CODEX,
        BackendName.CLAUDE,
        BackendName.GEMINI,
        BackendName.ANTIGRAVITY,
    }
    assert all(profile.routing_triggers for profile in profiles)
    assert all(profile.permission_dimensions for profile in profiles)


def test_permission_state_infers_known_cross_backend_presets():
    codex_state = permission_state_for_backend(
        "codex",
        {"approval_policy": "on-request", "sandbox_mode": "workspace-write"},
    )
    claude_state = permission_state_for_backend("claude", {"permission_mode": "default"})
    gemini_state = permission_state_for_backend("gemini", {"approval_mode": "default"})
    antigravity_state = permission_state_for_backend("antigravity", {"permission_mode": "default"})

    assert codex_state.preset_key == "ask"
    assert claude_state.preset_key == "ask"
    assert gemini_state.preset_key == "ask"
    assert antigravity_state.preset_key == "ask"
    assert codex_state.level == claude_state.level == gemini_state.level == antigravity_state.level


def test_permission_compatibility_uses_equivalent_preset_without_widening():
    compatibility = compare_permission_compatibility(
        "codex",
        {"approval_policy": "on-request", "sandbox_mode": "read-only"},
        "claude",
    )

    assert compatibility.compatible is True
    assert compatibility.widens_permissions is False
    assert compatibility.target_state.values == {"permission_mode": "plan"}
    assert "does not widen" in compatibility.reason


def test_permission_compatibility_flags_widening_target_state():
    compatibility = compare_permission_compatibility(
        "codex",
        {"approval_policy": "on-request", "sandbox_mode": "read-only"},
        "gemini",
        target_values={"approval_mode": "default"},
    )

    assert compatibility.compatible is False
    assert compatibility.widens_permissions is True
    assert compatibility.target_state.values == {"approval_mode": "default"}
    assert any("broader" in warning for warning in compatibility.warnings)


def test_permission_values_for_backend_are_backend_specific():
    assert permission_values_for_backend("full-access", "codex") == {
        "approval_policy": "never",
        "sandbox_mode": "danger-full-access",
    }
    assert permission_values_for_backend("full-access", "claude") == {
        "permission_mode": "bypassPermissions",
    }
    assert permission_values_for_backend("full-access", "gemini") == {
        "approval_mode": "yolo",
    }
    assert permission_values_for_backend("full-access", "antigravity") == {
        "permission_mode": "dangerously-skip-permissions",
    }
