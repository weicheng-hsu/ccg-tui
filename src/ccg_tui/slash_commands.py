from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from prompt_toolkit.completion import Completer, Completion


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str


@dataclass(frozen=True)
class SlashCommandSpec:
    canonical: str
    description: str
    aliases: tuple[str, ...] = ()

    def commands(self) -> tuple[SlashCommand, ...]:
        return (SlashCommand(self.canonical, self.description),) + tuple(
            SlashCommand(alias, f"Alias for {self.canonical}: {self.description}") for alias in self.aliases
        )


class SlashCommandAction(str, Enum):
    PRODUCT = "product"
    LOCAL = "local"
    BACKEND = "backend"
    INTERACTIVE_BACKEND = "interactive_backend"
    PASSTHROUGH = "passthrough"


@dataclass(frozen=True)
class ParsedSlashCommand:
    original: str
    command: str
    canonical: str
    args: str
    action: SlashCommandAction
    backend_prompt: str | None = None


SLASH_PALETTE_LOCAL_COMMANDS: tuple[str, ...] = (
    "/handoff",
    "/capabilities",
    "/summarize",
    "/context",
    "/history",
    "/details",
    "/task",
)

SLASH_PALETTE_PRODUCT_COMMANDS: tuple[str, ...] = (
    "/model",
    "/permissions",
    "/status",
    "/help",
    "/quit",
    "/clear",
    "/copy",
    "/resume",
)

SLASH_PALETTE_BACKEND_COMMANDS: tuple[str, ...] = (
    "/compact",
    "/plan",
    "/init",
    "/memory",
    "/mcp",
)

PRODUCT_SLASH_COMMAND_SPECS: tuple[SlashCommandSpec, ...] = (
    SlashCommandSpec("/help", "Show available CCG slash commands.", aliases=("/commands",)),
    SlashCommandSpec("/quit", "Exit the current CCG session.", aliases=("/exit",)),
    SlashCommandSpec("/clear", "Start a fresh conversation context.", aliases=("/new", "/reset")),
    SlashCommandSpec("/compact", "Compact or summarize the active backend context.", aliases=("/compress",)),
    SlashCommandSpec("/model", "Choose or inspect the active backend model."),
    SlashCommandSpec("/plan", "Switch the active backend into planning mode."),
    SlashCommandSpec("/permissions", "Review or change backend tool permissions.", aliases=("/allowed-tools", "/policies")),
    SlashCommandSpec("/status", "Show CCG session and backend status.", aliases=("/about", "/usage", "/stats", "/cost")),
    SlashCommandSpec("/capabilities", "Show advisory backend capability and permission compatibility information."),
    SlashCommandSpec("/copy", "Copy the latest assistant output when supported."),
    SlashCommandSpec("/resume", "Resume a saved conversation session.", aliases=("/continue",)),
    SlashCommandSpec("/init", "Create or update the backend project instruction file."),
    SlashCommandSpec("/memory", "Manage project memory for the active backend.", aliases=("/memories",)),
    SlashCommandSpec("/mcp", "Inspect or manage MCP server connections."),
    SlashCommandSpec("/history", "Refresh the local CCG conversation view."),
    SlashCommandSpec("/details", "Toggle backend activity details in the transcript."),
    SlashCommandSpec("/context", "Preview local resume context for this session."),
    SlashCommandSpec("/summarize", "Create a Gemini-backed CCG summary checkpoint."),
    SlashCommandSpec("/handoff", "Preview a manual cross-backend handoff packet."),
    SlashCommandSpec("/task", "Manage local task boundaries: start, status, and close."),
)


PRODUCT_COMMAND_ALIASES: dict[str, str] = {
    "/commands": "/help",
    "/exit": "/quit",
    "/new": "/clear",
    "/reset": "/clear",
    "/compress": "/compact",
    "/allowed-tools": "/permissions",
    "/policies": "/permissions",
    "/about": "/status",
    "/usage": "/status",
    "/stats": "/status",
    "/cost": "/status",
    "/continue": "/resume",
    "/memories": "/memory",
}

PRODUCT_HANDLED_COMMANDS = {
    "/help",
    "/quit",
    "/clear",
    "/model",
    "/permissions",
    "/status",
    "/capabilities",
    "/copy",
    "/resume",
}
LOCAL_UTILITY_COMMANDS = {"/history", "/details", "/context", "/summarize", "/handoff", "/task"}

TASK_SUBCOMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/task start", "Start a new local task boundary. Add an optional title after the command."),
    SlashCommand("/task status", "Show the active local task or the most recent closed task."),
    SlashCommand("/task close", "Close the active local task. Add an optional short closing note."),
)

BACKEND_CORE_COMMANDS = {"/compact", "/model", "/plan", "/permissions", "/init", "/memory", "/mcp"}

INTERACTIVE_BACKEND_COMMANDS = {"/mcp"}

BACKEND_COMMAND_TRANSLATIONS: dict[str, dict[str, str]] = {
    "codex": {
        "/memory": "/memories",
    },
    "claude": {},
    "gemini": {
        "/compact": "/compress",
    },
    "antigravity": {
        "/compact": "/compress",
    },
}


BACKEND_SLASH_COMMAND_SPECS: dict[str, tuple[SlashCommandSpec, ...]] = {
    "codex": (
        SlashCommandSpec("/model", "Choose or inspect the Codex model."),
        SlashCommandSpec("/fast", "Toggle faster Codex reasoning."),
        SlashCommandSpec("/permissions", "Change Codex approval and sandbox settings."),
        SlashCommandSpec("/keymap", "Remap Codex TUI shortcuts."),
        SlashCommandSpec("/experimental", "Open Codex experimental settings."),
        SlashCommandSpec("/autoreview", "Configure Codex automatic review behavior."),
        SlashCommandSpec("/memories", "Manage Codex memory."),
        SlashCommandSpec("/skills", "List or manage Codex skills."),
        SlashCommandSpec("/review", "Ask Codex to review code changes."),
        SlashCommandSpec("/rename", "Rename the current Codex session."),
        SlashCommandSpec("/new", "Start a new Codex conversation."),
        SlashCommandSpec("/resume", "Resume a Codex session."),
        SlashCommandSpec("/fork", "Fork the current Codex conversation."),
        SlashCommandSpec("/init", "Create or update Codex project instructions."),
        SlashCommandSpec("/compact", "Compact the Codex conversation context."),
        SlashCommandSpec("/plan", "Use Codex planning mode."),
        SlashCommandSpec("/collab", "Open Codex collaboration controls."),
        SlashCommandSpec("/agent", "Manage Codex agent work."),
        SlashCommandSpec("/side", "Start a Codex side conversation."),
        SlashCommandSpec("/copy", "Copy Codex output."),
        SlashCommandSpec("/diff", "Show Codex-visible git diff."),
        SlashCommandSpec("/mention", "Insert a file or symbol mention."),
        SlashCommandSpec("/status", "Show Codex status."),
        SlashCommandSpec("/title", "Set or inspect the session title."),
        SlashCommandSpec("/statusline", "Configure Codex statusline."),
        SlashCommandSpec("/theme", "Change Codex theme."),
        SlashCommandSpec("/mcp", "Manage Codex MCP servers."),
        SlashCommandSpec("/plugins", "Manage Codex plugins."),
        SlashCommandSpec("/logout", "Sign out of Codex."),
        SlashCommandSpec("/exit", "Exit Codex.", aliases=("/quit",)),
        SlashCommandSpec("/feedback", "Send Codex feedback."),
        SlashCommandSpec("/ps", "List Codex background tasks."),
        SlashCommandSpec("/stop", "Stop Codex background work."),
        SlashCommandSpec("/clear", "Clear Codex conversation state."),
        SlashCommandSpec("/personality", "Change Codex response personality."),
        SlashCommandSpec("/realtime", "Configure Codex realtime mode."),
        SlashCommandSpec("/settings", "Open Codex settings."),
        SlashCommandSpec("/subagents", "Manage Codex subagents."),
        SlashCommandSpec("/goal", "Set or view the long-running task goal."),
    ),
    "claude": (
        SlashCommandSpec("/add-dir", "Add another working directory."),
        SlashCommandSpec("/agents", "Manage Claude agents."),
        SlashCommandSpec("/autofix-pr", "Automatically fix PR feedback."),
        SlashCommandSpec("/batch", "Run batch Claude work."),
        SlashCommandSpec("/branch", "Branch the conversation.", aliases=("/fork",)),
        SlashCommandSpec("/btw", "Ask a side question."),
        SlashCommandSpec("/chrome", "Connect Claude to Chrome."),
        SlashCommandSpec("/claude-api", "Configure Claude API usage."),
        SlashCommandSpec("/clear", "Clear Claude context.", aliases=("/reset",)),
        SlashCommandSpec("/new", "Start a new Claude conversation."),
        SlashCommandSpec("/color", "Change Claude color settings."),
        SlashCommandSpec("/compact", "Compact Claude context."),
        SlashCommandSpec("/config", "Open Claude configuration.", aliases=("/settings",)),
        SlashCommandSpec("/context", "Show Claude context information."),
        SlashCommandSpec("/copy", "Copy Claude output."),
        SlashCommandSpec("/cost", "Show Claude usage cost.", aliases=("/usage",)),
        SlashCommandSpec("/debug", "Show Claude debug information."),
        SlashCommandSpec("/desktop", "Open Claude desktop integration.", aliases=("/app",)),
        SlashCommandSpec("/diff", "Show git diff."),
        SlashCommandSpec("/doctor", "Diagnose Claude Code setup."),
        SlashCommandSpec("/effort", "Change reasoning effort."),
        SlashCommandSpec("/exit", "Exit Claude.", aliases=("/quit",)),
        SlashCommandSpec("/export", "Export the conversation."),
        SlashCommandSpec("/extra-usage", "Show extra usage details."),
        SlashCommandSpec("/fast", "Toggle Claude fast mode."),
        SlashCommandSpec("/feedback", "Send Claude feedback.", aliases=("/bug",)),
        SlashCommandSpec("/fewer-permission-prompts", "Reduce permission prompts."),
        SlashCommandSpec("/focus", "Focus Claude on current work."),
        SlashCommandSpec("/heapdump", "Capture a heap dump."),
        SlashCommandSpec("/help", "Show Claude help."),
        SlashCommandSpec("/hooks", "Manage Claude hooks."),
        SlashCommandSpec("/ide", "Connect IDE integration."),
        SlashCommandSpec("/init", "Create or update CLAUDE.md."),
        SlashCommandSpec("/insights", "Show Claude insights."),
        SlashCommandSpec("/install-github-app", "Install the GitHub app."),
        SlashCommandSpec("/install-slack-app", "Install the Slack app."),
        SlashCommandSpec("/keybindings", "Configure keybindings."),
        SlashCommandSpec("/login", "Sign in to Claude."),
        SlashCommandSpec("/logout", "Sign out of Claude."),
        SlashCommandSpec("/loop", "Configure proactive loop mode.", aliases=("/proactive",)),
        SlashCommandSpec("/mcp", "Manage Claude MCP servers."),
        SlashCommandSpec("/memory", "Manage Claude memory."),
        SlashCommandSpec("/mobile", "Set up mobile app links.", aliases=("/ios", "/android")),
        SlashCommandSpec("/model", "Choose or inspect Claude model."),
        SlashCommandSpec("/passes", "Configure passes."),
        SlashCommandSpec("/permissions", "Manage allowed tools.", aliases=("/allowed-tools",)),
        SlashCommandSpec("/plan", "Use Claude planning mode."),
        SlashCommandSpec("/plugin", "Manage Claude plugins."),
        SlashCommandSpec("/powerup", "Open Claude power-up flow."),
        SlashCommandSpec("/privacy-settings", "Open privacy settings."),
        SlashCommandSpec("/recap", "Recap the current conversation."),
        SlashCommandSpec("/release-notes", "Show release notes."),
        SlashCommandSpec("/reload-plugins", "Reload plugins."),
        SlashCommandSpec("/remote-control", "Configure remote control.", aliases=("/rc",)),
        SlashCommandSpec("/remote-env", "Inspect remote environment."),
        SlashCommandSpec("/rename", "Rename the current session."),
        SlashCommandSpec("/resume", "Resume a Claude session.", aliases=("/continue",)),
        SlashCommandSpec("/review", "Review code changes."),
        SlashCommandSpec("/rewind", "Rewind to a checkpoint.", aliases=("/checkpoint", "/undo")),
        SlashCommandSpec("/sandbox", "Configure Claude sandboxing."),
        SlashCommandSpec("/schedule", "Manage scheduled routines.", aliases=("/routines",)),
        SlashCommandSpec("/security-review", "Run a security review."),
        SlashCommandSpec("/setup-bedrock", "Configure Amazon Bedrock."),
        SlashCommandSpec("/setup-vertex", "Configure Vertex AI."),
        SlashCommandSpec("/simplify", "Ask Claude to simplify code."),
        SlashCommandSpec("/skills", "Manage Claude skills."),
        SlashCommandSpec("/stats", "Show Claude usage stats."),
        SlashCommandSpec("/status", "Show Claude status."),
        SlashCommandSpec("/statusline", "Configure statusline."),
        SlashCommandSpec("/stickers", "Open stickers."),
        SlashCommandSpec("/tasks", "Manage background tasks.", aliases=("/bashes",)),
        SlashCommandSpec("/team-onboarding", "Open team onboarding."),
        SlashCommandSpec("/teleport", "Teleport context.", aliases=("/tp",)),
        SlashCommandSpec("/terminal-setup", "Configure terminal integration."),
        SlashCommandSpec("/theme", "Change Claude theme."),
        SlashCommandSpec("/tui", "Configure TUI behavior."),
        SlashCommandSpec("/ultraplan", "Run deeper planning."),
        SlashCommandSpec("/ultrareview", "Run deeper review."),
        SlashCommandSpec("/upgrade", "Upgrade Claude plan or install."),
        SlashCommandSpec("/voice", "Configure voice input."),
        SlashCommandSpec("/web-setup", "Set up web integration."),
    ),
    "gemini": (
        SlashCommandSpec("/about", "Show Gemini CLI information."),
        SlashCommandSpec("/agents", "Manage Gemini agents."),
        SlashCommandSpec("/auth", "Manage Gemini authentication."),
        SlashCommandSpec("/bug", "File Gemini CLI feedback."),
        SlashCommandSpec("/chat", "Manage saved Gemini chats."),
        SlashCommandSpec("/clear", "Clear Gemini conversation."),
        SlashCommandSpec("/commands", "Manage custom Gemini commands."),
        SlashCommandSpec("/compress", "Compress Gemini context.", aliases=("/summarize", "/compact")),
        SlashCommandSpec("/copy", "Copy Gemini output."),
        SlashCommandSpec("/corgi", "Toggle Gemini corgi mode."),
        SlashCommandSpec("/docs", "Open Gemini documentation."),
        SlashCommandSpec("/directory", "Manage workspace directories.", aliases=("/dir",)),
        SlashCommandSpec("/editor", "Configure editor integration."),
        SlashCommandSpec("/extensions", "Manage Gemini extensions."),
        SlashCommandSpec("/help", "Show Gemini help."),
        SlashCommandSpec("/footer", "Configure footer/statusline.", aliases=("/statusline",)),
        SlashCommandSpec("/shortcuts", "Show keyboard shortcuts."),
        SlashCommandSpec("/hooks", "Manage Gemini hooks."),
        SlashCommandSpec("/rewind", "Rewind Gemini conversation."),
        SlashCommandSpec("/ide", "Connect IDE integration."),
        SlashCommandSpec("/init", "Create or update GEMINI.md."),
        SlashCommandSpec("/oncall", "Open on-call workflows."),
        SlashCommandSpec("/mcp", "Manage Gemini MCP servers."),
        SlashCommandSpec("/memory", "Manage Gemini memory."),
        SlashCommandSpec("/model", "Choose or inspect Gemini model."),
        SlashCommandSpec("/permissions", "Manage Gemini permissions."),
        SlashCommandSpec("/plan", "Use Gemini planning mode."),
        SlashCommandSpec("/policies", "Inspect Gemini policies."),
        SlashCommandSpec("/privacy", "Open privacy information."),
        SlashCommandSpec("/profile", "Profile Gemini CLI performance."),
        SlashCommandSpec("/quit", "Exit Gemini.", aliases=("/exit",)),
        SlashCommandSpec("/restore", "Restore a prior Gemini checkpoint."),
        SlashCommandSpec("/resume", "Resume a Gemini session."),
        SlashCommandSpec("/stats", "Show Gemini usage stats.", aliases=("/usage",)),
        SlashCommandSpec("/theme", "Change Gemini theme."),
        SlashCommandSpec("/tools", "Inspect Gemini tools."),
        SlashCommandSpec("/skills", "Manage Gemini skills."),
        SlashCommandSpec("/settings", "Open Gemini settings."),
        SlashCommandSpec("/gemma", "Check local Gemma model routing status."),
        SlashCommandSpec("/tasks", "Manage background tasks.", aliases=("/bg", "/background")),
        SlashCommandSpec("/vim", "Toggle vim mode."),
        SlashCommandSpec("/setup-github", "Set up GitHub integration."),
        SlashCommandSpec("/terminal-setup", "Configure terminal integration."),
        SlashCommandSpec("/upgrade", "Upgrade Gemini CLI."),
    ),
    "antigravity": (
        SlashCommandSpec("/about", "Show Antigravity CLI information."),
        SlashCommandSpec("/auth", "Manage Antigravity authentication."),
        SlashCommandSpec("/changelog", "Show Antigravity CLI changelog."),
        SlashCommandSpec("/clear", "Clear Antigravity conversation state."),
        SlashCommandSpec("/commands", "Manage custom Antigravity commands."),
        SlashCommandSpec("/compress", "Compress Antigravity context.", aliases=("/compact",)),
        SlashCommandSpec("/copy", "Copy Antigravity output."),
        SlashCommandSpec("/diff", "Show Antigravity-visible git diff."),
        SlashCommandSpec("/help", "Show Antigravity help."),
        SlashCommandSpec("/hooks", "Manage Antigravity hooks."),
        SlashCommandSpec("/init", "Create or update Antigravity project instructions."),
        SlashCommandSpec("/login", "Sign in to Antigravity."),
        SlashCommandSpec("/logout", "Sign out of Antigravity."),
        SlashCommandSpec("/mcp", "Manage Antigravity MCP servers."),
        SlashCommandSpec("/memory", "Manage Antigravity memory."),
        SlashCommandSpec("/model", "Choose or inspect the Antigravity model."),
        SlashCommandSpec("/permissions", "Manage Antigravity permissions."),
        SlashCommandSpec("/plan", "Use Antigravity planning mode."),
        SlashCommandSpec("/plugin", "Manage Antigravity plugins.", aliases=("/plugins",)),
        SlashCommandSpec("/quota", "Show Antigravity quota information."),
        SlashCommandSpec("/quit", "Exit Antigravity.", aliases=("/exit",)),
        SlashCommandSpec("/resume", "Resume an Antigravity conversation."),
        SlashCommandSpec("/review", "Ask Antigravity to review code changes."),
        SlashCommandSpec("/settings", "Open Antigravity settings."),
        SlashCommandSpec("/skills", "Manage Antigravity skills."),
        SlashCommandSpec("/status", "Show Antigravity status."),
        SlashCommandSpec("/statusline", "Configure Antigravity statusline."),
        SlashCommandSpec("/usage", "Show Antigravity usage information."),
    ),
}


def split_slash_command(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    command, separator, args = stripped.partition(" ")
    return command.lower(), args.strip() if separator else ""


def canonical_slash_command(command: str) -> str:
    return PRODUCT_COMMAND_ALIASES.get(command.lower(), command.lower())


def backend_slash_command(canonical: str, backend: str) -> str:
    return BACKEND_COMMAND_TRANSLATIONS.get(backend.strip().lower(), {}).get(canonical, canonical)


def parse_slash_command(text: str, backend: str) -> ParsedSlashCommand | None:
    split = split_slash_command(text)
    if split is None:
        return None
    command, args = split
    canonical = canonical_slash_command(command)
    if canonical in PRODUCT_HANDLED_COMMANDS:
        return ParsedSlashCommand(
            original=text,
            command=command,
            canonical=canonical,
            args=args,
            action=SlashCommandAction.PRODUCT,
        )
    if canonical in LOCAL_UTILITY_COMMANDS:
        return ParsedSlashCommand(
            original=text,
            command=command,
            canonical=canonical,
            args=args,
            action=SlashCommandAction.LOCAL,
        )
    if canonical in BACKEND_CORE_COMMANDS:
        backend_command = backend_slash_command(canonical, backend)
        backend_prompt = backend_command if not args else f"{backend_command} {args}"
        action = SlashCommandAction.INTERACTIVE_BACKEND if canonical in INTERACTIVE_BACKEND_COMMANDS else SlashCommandAction.BACKEND
        return ParsedSlashCommand(
            original=text,
            command=command,
            canonical=canonical,
            args=args,
            action=action,
            backend_prompt=backend_prompt,
        )
    return ParsedSlashCommand(
        original=text,
        command=command,
        canonical=canonical,
        args=args,
        action=SlashCommandAction.PASSTHROUGH,
        backend_prompt=text.strip(),
    )


def slash_commands(backend: str | None = None) -> tuple[SlashCommand, ...]:
    commands: list[SlashCommand] = []
    seen: set[str] = set()
    specs = list(PRODUCT_SLASH_COMMAND_SPECS)
    if backend is not None:
        specs.extend(BACKEND_SLASH_COMMAND_SPECS.get(backend.strip().lower(), ()))
    for spec in specs:
        for command in spec.commands():
            if command.name in seen:
                continue
            commands.append(command)
            seen.add(command.name)
    return tuple(commands)


def slash_command_palette_group(command: str, *, backend: str | None = None) -> str:
    canonical = canonical_slash_command(command)
    if canonical in SLASH_PALETTE_LOCAL_COMMANDS:
        return "local"
    if canonical in SLASH_PALETTE_PRODUCT_COMMANDS:
        return "product"
    if canonical in SLASH_PALETTE_BACKEND_COMMANDS:
        return "backend"
    return "passthrough"


def _palette_command_name(command) -> str:
    name = getattr(command, "name", None)
    if name is not None:
        return name
    text = getattr(command, "text", None)
    if text is not None:
        return text
    raise AttributeError("command has neither name nor text")


def _palette_command_rank(command) -> int:
    canonical = canonical_slash_command(_palette_command_name(command))
    palette_order = (
        *SLASH_PALETTE_LOCAL_COMMANDS,
        *SLASH_PALETTE_PRODUCT_COMMANDS,
        *SLASH_PALETTE_BACKEND_COMMANDS,
    )
    try:
        return palette_order.index(canonical)
    except ValueError:
        return len(palette_order)


def ordered_slash_commands_for_palette(
    commands: Iterable[object],
    *,
    backend: str | None = None,
) -> tuple[object, ...]:
    grouped = tuple(enumerate(commands))
    group_order = {"local": 0, "product": 1, "backend": 2, "passthrough": 3}

    def sort_key(item: tuple[int, object]) -> tuple[int, int, int, str]:
        index, command = item
        name = _palette_command_name(command)
        canonical = canonical_slash_command(name)
        group = slash_command_palette_group(name, backend=backend)
        alias_rank = 1 if canonical != name.lower() else 0
        return (group_order[group], alias_rank, _palette_command_rank(command), index, name)

    return tuple(command for _, command in sorted(grouped, key=sort_key))


def filter_slash_commands(
    prefix: str,
    commands: Iterable[SlashCommand] | None = None,
    *,
    backend: str | None = None,
) -> tuple[SlashCommand, ...]:
    normalized = prefix.strip().lower()
    if not normalized.startswith("/"):
        return ()
    if normalized.startswith("/task"):
        if normalized == "/task":
            return (SlashCommand("/task", "Manage local task boundaries: start, status, and close."),)
        if normalized.startswith("/task "):
            return tuple(command for command in TASK_SUBCOMMANDS if command.name.startswith(normalized))
    available = commands if commands is not None else slash_commands(backend)
    return tuple(command for command in available if command.name.lower().startswith(normalized))


def format_slash_command_help(commands: Iterable[SlashCommand] | None = None, *, backend: str | None = None) -> str:
    available = tuple(commands if commands is not None else slash_commands(backend))
    width = max((len(command.name) for command in available), default=0)
    lines = ["Slash Commands"]
    lines.extend(f"{command.name.ljust(width)}  {command.description}" for command in available)
    return "\n".join(lines)


class SlashCommandCompleter(Completer):
    def __init__(self, backend_provider=None) -> None:
        self.backend_provider = backend_provider

    def _backend(self) -> str | None:
        if self.backend_provider is None:
            return None
        return self.backend_provider()

    def get_completions(self, document, complete_event):
        line = document.current_line_before_cursor
        if not line.startswith("/"):
            return
        filtered = filter_slash_commands(line, backend=self._backend())
        for command in ordered_slash_commands_for_palette(filtered, backend=self._backend()):
            yield Completion(
                command.name,
                start_position=-len(line),
                display=command.name,
                display_meta=command.description,
            )
