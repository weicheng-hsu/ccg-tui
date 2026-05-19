from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ccg_tui.models import BackendName


ROUTING_POLICY_REFERENCE = "README.md"
DEFAULT_PERMISSION_PRESET_KEY = "ask"


@dataclass(frozen=True, slots=True)
class PermissionPresetSpec:
    key: str
    label: str
    description: str
    codex_approval_policy: str
    codex_sandbox_mode: str
    claude_permission_mode: str
    gemini_approval_mode: str

    def values_for_backend(self, backend: BackendName | str) -> dict[str, str]:
        normalized = normalize_backend_name(backend)
        if normalized is BackendName.CODEX:
            return {
                "approval_policy": self.codex_approval_policy,
                "sandbox_mode": self.codex_sandbox_mode,
            }
        if normalized is BackendName.CLAUDE:
            return {"permission_mode": self.claude_permission_mode}
        if normalized is BackendName.GEMINI:
            return {"approval_mode": self.gemini_approval_mode}
        return {}


PERMISSION_PRESET_SPECS: tuple[PermissionPresetSpec, ...] = (
    PermissionPresetSpec(
        key="plan",
        label="Plan / read-only",
        description="Read-only planning mode; safest for investigation before edits.",
        codex_approval_policy="on-request",
        codex_sandbox_mode="read-only",
        claude_permission_mode="plan",
        gemini_approval_mode="plan",
    ),
    PermissionPresetSpec(
        key="ask",
        label="Ask before actions",
        description="Prompt before risky tool use while allowing workspace context.",
        codex_approval_policy="on-request",
        codex_sandbox_mode="workspace-write",
        claude_permission_mode="default",
        gemini_approval_mode="default",
    ),
    PermissionPresetSpec(
        key="auto-edit",
        label="Auto-edit workspace",
        description="Allow normal workspace edits with fewer prompts.",
        codex_approval_policy="never",
        codex_sandbox_mode="workspace-write",
        claude_permission_mode="acceptEdits",
        gemini_approval_mode="auto_edit",
    ),
    PermissionPresetSpec(
        key="full-access",
        label="Full access",
        description="Bypass most prompts and sandboxing; use only in an external sandbox.",
        codex_approval_policy="never",
        codex_sandbox_mode="danger-full-access",
        claude_permission_mode="bypassPermissions",
        gemini_approval_mode="yolo",
    ),
)

_PRESET_BY_KEY = {spec.key: spec for spec in PERMISSION_PRESET_SPECS}
_PERMISSION_LEVELS = {
    "plan": 0,
    "ask": 1,
    "auto-edit": 2,
    "full-access": 3,
}


@dataclass(frozen=True, slots=True)
class BackendCapabilityProfile:
    backend: BackendName
    display_name: str
    summary: str
    strengths: tuple[str, ...]
    limitations: tuple[str, ...]
    routing_triggers: tuple[str, ...]
    permission_dimensions: tuple[str, ...]
    supports_manual_handoff: bool = True
    supports_local_resume: bool = True
    supports_interactive_commands: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "display_name": self.display_name,
            "summary": self.summary,
            "strengths": list(self.strengths),
            "limitations": list(self.limitations),
            "routing_triggers": list(self.routing_triggers),
            "permission_dimensions": list(self.permission_dimensions),
            "supports_manual_handoff": self.supports_manual_handoff,
            "supports_local_resume": self.supports_local_resume,
            "supports_interactive_commands": self.supports_interactive_commands,
        }


@dataclass(frozen=True, slots=True)
class CapabilityFact:
    key: str
    label: str
    supported: bool
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "supported": self.supported,
            "explanation": self.explanation,
        }


BACKEND_CAPABILITY_REGISTRY: dict[BackendName, BackendCapabilityProfile] = {
    BackendName.CODEX: BackendCapabilityProfile(
        backend=BackendName.CODEX,
        display_name="Codex",
        summary="OpenAI-backed coding CLI with explicit approval and sandbox flags.",
        strengths=(
            "repository editing",
            "code review",
            "sandboxed tool use",
            "Codex skills and plugins",
        ),
        limitations=(
            "vendor-native auth required",
            "approval and sandbox state are Codex-specific",
        ),
        routing_triggers=(
            "need sandboxed repository edits",
            "need Codex-native skills or review commands",
            "manual handoff target is Codex",
        ),
        permission_dimensions=("approval_policy", "sandbox_mode"),
    ),
    BackendName.CLAUDE: BackendCapabilityProfile(
        backend=BackendName.CLAUDE,
        display_name="Claude",
        summary="Anthropic Claude Code CLI with a single backend-owned permission mode.",
        strengths=(
            "large-context code analysis",
            "plan-mode workflows",
            "Claude-native agents and slash commands",
        ),
        limitations=(
            "vendor-native auth required",
            "permission modes do not expose a separate filesystem sandbox flag",
        ),
        routing_triggers=(
            "need Claude-native planning or agent commands",
            "manual handoff target is Claude",
            "current task benefits from long-context inspection",
        ),
        permission_dimensions=("permission_mode",),
    ),
    BackendName.GEMINI: BackendCapabilityProfile(
        backend=BackendName.GEMINI,
        display_name="Gemini",
        summary="Google Gemini CLI with approval-mode controls and summary checkpoint support.",
        strengths=(
            "Gemini-backed summary checkpoints",
            "large-context inspection",
            "Gemini-native tool and policy commands",
        ),
        limitations=(
            "vendor-native auth required",
            "approval modes do not expose a separate filesystem sandbox flag",
        ),
        routing_triggers=(
            "need Gemini summary checkpoint generation",
            "manual handoff target is Gemini",
            "need Gemini-native policy or tool inspection",
        ),
        permission_dimensions=("approval_mode",),
    ),
}


_CAPABILITY_FACT_KEYS = (
    "model_flag_support",
    "persistent_session_support",
    "permission_concepts",
    "native_slash_passthrough",
    "summary_suitability",
    "handoff_suitability",
)


BACKEND_CAPABILITY_FACTS: dict[BackendName, dict[str, CapabilityFact]] = {
    BackendName.CODEX: {
        "model_flag_support": CapabilityFact(
            "model_flag_support",
            "Model flag support",
            True,
            "Codex one-shot and PTY launch paths accept an explicit model flag.",
        ),
        "persistent_session_support": CapabilityFact(
            "persistent_session_support",
            "Persistent session support",
            True,
            "Codex PTY sessions are reused by the session controller for fullscreen turns.",
        ),
        "permission_concepts": CapabilityFact(
            "permission_concepts",
            "Permission concepts",
            True,
            "Codex exposes both approval policy and filesystem sandbox mode.",
        ),
        "native_slash_passthrough": CapabilityFact(
            "native_slash_passthrough",
            "Native slash passthrough",
            True,
            "Backend-native Codex commands can be routed to Codex after CCG-local commands take precedence.",
        ),
        "summary_suitability": CapabilityFact(
            "summary_suitability",
            "Summary suitability",
            False,
            "CCG summary checkpoint generation is currently Gemini-backed; Codex can receive handoff context but is not the summary backend.",
        ),
        "handoff_suitability": CapabilityFact(
            "handoff_suitability",
            "Handoff suitability",
            True,
            "Codex can be a manual handoff source or target through CCG packet injection.",
        ),
    },
    BackendName.CLAUDE: {
        "model_flag_support": CapabilityFact(
            "model_flag_support",
            "Model flag support",
            True,
            "Claude one-shot and PTY launch paths accept an explicit model flag.",
        ),
        "persistent_session_support": CapabilityFact(
            "persistent_session_support",
            "Persistent session support",
            True,
            "Claude PTY sessions are reused through a vendor session id and transcript tail.",
        ),
        "permission_concepts": CapabilityFact(
            "permission_concepts",
            "Permission concepts",
            True,
            "Claude exposes permission modes, but not a CCG-equivalent filesystem sandbox flag.",
        ),
        "native_slash_passthrough": CapabilityFact(
            "native_slash_passthrough",
            "Native slash passthrough",
            True,
            "Backend-native Claude commands can be routed to Claude after CCG-local commands take precedence.",
        ),
        "summary_suitability": CapabilityFact(
            "summary_suitability",
            "Summary suitability",
            False,
            "CCG summary checkpoint generation is currently Gemini-backed; Claude can receive handoff context but is not the summary backend.",
        ),
        "handoff_suitability": CapabilityFact(
            "handoff_suitability",
            "Handoff suitability",
            True,
            "Claude can be a manual handoff source or target through CCG packet injection.",
        ),
    },
    BackendName.GEMINI: {
        "model_flag_support": CapabilityFact(
            "model_flag_support",
            "Model flag support",
            True,
            "Gemini one-shot and PTY launch paths accept an explicit model flag.",
        ),
        "persistent_session_support": CapabilityFact(
            "persistent_session_support",
            "Persistent session support",
            True,
            "Gemini PTY sessions are reused by watching the active chat transcript.",
        ),
        "permission_concepts": CapabilityFact(
            "permission_concepts",
            "Permission concepts",
            True,
            "Gemini exposes approval modes, but not a CCG-equivalent filesystem sandbox flag.",
        ),
        "native_slash_passthrough": CapabilityFact(
            "native_slash_passthrough",
            "Native slash passthrough",
            True,
            "Backend-native Gemini commands can be routed to Gemini after CCG-local commands take precedence.",
        ),
        "summary_suitability": CapabilityFact(
            "summary_suitability",
            "Summary suitability",
            True,
            "Gemini is the implemented CCG summary checkpoint backend.",
        ),
        "handoff_suitability": CapabilityFact(
            "handoff_suitability",
            "Handoff suitability",
            True,
            "Gemini can be a manual handoff source or target through CCG packet injection.",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class PermissionState:
    backend: BackendName
    values: dict[str, str]
    preset_key: str | None
    label: str
    level: int
    known: bool
    can_edit_workspace: bool
    full_access: bool
    requires_confirmation: bool
    sandboxed: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend.value,
            "values": dict(self.values),
            "preset_key": self.preset_key,
            "label": self.label,
            "level": self.level,
            "known": self.known,
            "can_edit_workspace": self.can_edit_workspace,
            "full_access": self.full_access,
            "requires_confirmation": self.requires_confirmation,
            "sandboxed": self.sandboxed,
        }


@dataclass(frozen=True, slots=True)
class PermissionCompatibility:
    source_backend: BackendName
    target_backend: BackendName
    source_state: PermissionState
    target_state: PermissionState
    widens_permissions: bool
    compatible: bool
    requires_confirmation: bool
    reason: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_backend": self.source_backend.value,
            "target_backend": self.target_backend.value,
            "source_state": self.source_state.to_dict(),
            "target_state": self.target_state.to_dict(),
            "widens_permissions": self.widens_permissions,
            "compatible": self.compatible,
            "requires_confirmation": self.requires_confirmation,
            "reason": self.reason,
            "warnings": list(self.warnings),
        }


def normalize_backend_name(backend: BackendName | str) -> BackendName:
    if isinstance(backend, BackendName):
        return backend
    return BackendName(str(backend).strip().lower())


def all_backend_capabilities() -> tuple[BackendCapabilityProfile, ...]:
    return tuple(BACKEND_CAPABILITY_REGISTRY[backend] for backend in BackendName)


def backend_capabilities(backend: BackendName | str) -> BackendCapabilityProfile:
    return BACKEND_CAPABILITY_REGISTRY[normalize_backend_name(backend)]


def backend_capability_facts(backend: BackendName | str) -> tuple[CapabilityFact, ...]:
    facts = BACKEND_CAPABILITY_FACTS[normalize_backend_name(backend)]
    return tuple(facts[key] for key in _CAPABILITY_FACT_KEYS)


def backend_capability_fact(backend: BackendName | str, key: str) -> CapabilityFact:
    normalized = normalize_backend_name(backend)
    try:
        return BACKEND_CAPABILITY_FACTS[normalized][key]
    except KeyError as exc:
        valid = ", ".join(_CAPABILITY_FACT_KEYS)
        raise ValueError(f"unknown backend capability: {key!r}; expected one of {valid}") from exc


def permission_preset_spec(key: str) -> PermissionPresetSpec:
    try:
        return _PRESET_BY_KEY[key]
    except KeyError as exc:
        raise ValueError(f"unknown permission preset: {key!r}") from exc


def permission_values_for_backend(preset_key: str, backend: BackendName | str) -> dict[str, str]:
    return permission_preset_spec(preset_key).values_for_backend(backend)


def permission_state_for_backend(
    backend: BackendName | str,
    values: dict[str, str] | None = None,
) -> PermissionState:
    normalized = normalize_backend_name(backend)
    resolved_values = dict(values or permission_values_for_backend(DEFAULT_PERMISSION_PRESET_KEY, normalized))
    for spec in PERMISSION_PRESET_SPECS:
        if spec.values_for_backend(normalized) == resolved_values:
            return _permission_state_from_preset(normalized, resolved_values, spec.key, spec.label, known=True)
    return _custom_permission_state(normalized, resolved_values)


def compare_permission_compatibility(
    source_backend: BackendName | str,
    source_values: dict[str, str] | None,
    target_backend: BackendName | str,
    *,
    target_values: dict[str, str] | None = None,
    target_preset_key: str | None = None,
) -> PermissionCompatibility:
    normalized_source = normalize_backend_name(source_backend)
    normalized_target = normalize_backend_name(target_backend)
    source_state = permission_state_for_backend(normalized_source, source_values)
    if target_values is None:
        if target_preset_key is not None:
            target_values = permission_values_for_backend(target_preset_key, normalized_target)
        elif source_state.preset_key is not None:
            target_values = permission_values_for_backend(source_state.preset_key, normalized_target)
        else:
            target_values = permission_values_for_backend(DEFAULT_PERMISSION_PRESET_KEY, normalized_target)
    target_state = permission_state_for_backend(normalized_target, target_values)
    widens = target_state.level > source_state.level
    warnings: list[str] = []
    if not source_state.known:
        warnings.append("source permission state is custom or unknown")
    if not target_state.known:
        warnings.append("target permission state is custom or unknown")
    if source_state.sandboxed is not None and target_state.sandboxed is None:
        warnings.append("target backend has no equivalent filesystem sandbox flag")
    if source_state.sandboxed is None and target_state.sandboxed is not None:
        warnings.append("source backend has no equivalent filesystem sandbox flag")
    if widens:
        warnings.append("target permission state is broader than the active state")
    compatible = not widens
    reason = (
        "target permission state would widen permissions"
        if widens
        else "target permission state does not widen the active permissions"
    )
    return PermissionCompatibility(
        source_backend=normalized_source,
        target_backend=normalized_target,
        source_state=source_state,
        target_state=target_state,
        widens_permissions=widens,
        compatible=compatible,
        requires_confirmation=True,
        reason=reason,
        warnings=tuple(warnings),
    )


def _permission_state_from_preset(
    backend: BackendName,
    values: dict[str, str],
    preset_key: str,
    label: str,
    *,
    known: bool,
) -> PermissionState:
    level = _PERMISSION_LEVELS[preset_key]
    return PermissionState(
        backend=backend,
        values=dict(values),
        preset_key=preset_key,
        label=label,
        level=level,
        known=known,
        can_edit_workspace=level >= _PERMISSION_LEVELS["ask"],
        full_access=level >= _PERMISSION_LEVELS["full-access"],
        requires_confirmation=level <= _PERMISSION_LEVELS["ask"],
        sandboxed=_sandboxed_for_backend(backend, values, level),
    )


def _custom_permission_state(backend: BackendName, values: dict[str, str]) -> PermissionState:
    level = _infer_custom_level(backend, values)
    return PermissionState(
        backend=backend,
        values=dict(values),
        preset_key=None,
        label="Custom permissions",
        level=level,
        known=False,
        can_edit_workspace=level >= _PERMISSION_LEVELS["ask"],
        full_access=level >= _PERMISSION_LEVELS["full-access"],
        requires_confirmation=level <= _PERMISSION_LEVELS["ask"],
        sandboxed=_sandboxed_for_backend(backend, values, level),
    )


def _infer_custom_level(backend: BackendName, values: dict[str, str]) -> int:
    if backend is BackendName.CODEX:
        sandbox_mode = values.get("sandbox_mode", "")
        approval_policy = values.get("approval_policy", "")
        if sandbox_mode == "read-only":
            return _PERMISSION_LEVELS["plan"]
        if sandbox_mode == "danger-full-access":
            return _PERMISSION_LEVELS["full-access"]
        if approval_policy == "never":
            return _PERMISSION_LEVELS["auto-edit"]
        return _PERMISSION_LEVELS["ask"]
    if backend is BackendName.CLAUDE:
        mode = values.get("permission_mode", "")
        return {
            "plan": _PERMISSION_LEVELS["plan"],
            "default": _PERMISSION_LEVELS["ask"],
            "acceptEdits": _PERMISSION_LEVELS["auto-edit"],
            "bypassPermissions": _PERMISSION_LEVELS["full-access"],
        }.get(mode, _PERMISSION_LEVELS["ask"])
    if backend is BackendName.GEMINI:
        mode = values.get("approval_mode", "")
        return {
            "plan": _PERMISSION_LEVELS["plan"],
            "default": _PERMISSION_LEVELS["ask"],
            "auto_edit": _PERMISSION_LEVELS["auto-edit"],
            "yolo": _PERMISSION_LEVELS["full-access"],
        }.get(mode, _PERMISSION_LEVELS["ask"])
    return _PERMISSION_LEVELS["ask"]


def _sandboxed_for_backend(backend: BackendName, values: dict[str, str], level: int) -> bool | None:
    if backend is BackendName.CODEX:
        return values.get("sandbox_mode") != "danger-full-access"
    if level >= _PERMISSION_LEVELS["full-access"]:
        return False
    return None
