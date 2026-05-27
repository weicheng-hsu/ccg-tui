from prompt_toolkit.document import Document

from ccg_tui.slash_commands import (
    SlashCommandAction,
    SlashCommandCompleter,
    filter_slash_commands,
    format_slash_command_help,
    parse_slash_command,
    ordered_slash_commands_for_palette,
    slash_commands,
)


def test_slash_commands_have_descriptions():
    commands = slash_commands()

    assert any(command.name == "/help" for command in commands)
    assert any(command.name == "/compact" for command in commands)
    assert any(command.name == "/summarize" for command in commands)
    assert any(command.name == "/handoff" for command in commands)
    assert any(command.name == "/task" for command in commands)
    assert any(command.name == "/capabilities" for command in commands)
    assert all(command.description for command in commands)


def test_filter_slash_commands_lists_all_for_bare_slash_and_filters_prefix():
    all_commands = filter_slash_commands("/")
    memory_commands = filter_slash_commands("/mem")

    assert len(all_commands) == len(slash_commands())
    assert [command.name for command in memory_commands] == ["/memory", "/memories"]
    assert filter_slash_commands("hello") == ()


def test_filter_slash_commands_can_include_backend_native_commands():
    product_only = filter_slash_commands("/review")
    claude_commands = filter_slash_commands("/review", backend="claude")

    assert product_only == ()
    assert [command.name for command in claude_commands] == ["/review"]
    assert "Review code changes." in claude_commands[0].description


def test_slash_commands_dedupe_backend_native_names_with_ccg_precedence():
    commands = [command for command in slash_commands("claude") if command.name == "/status"]

    assert len(commands) == 1
    assert commands[0].description == "Show CCG session and backend status."


def test_slash_command_help_includes_descriptions():
    text = format_slash_command_help()

    assert "Slash Commands" in text
    assert "/help" in text
    assert "Show available CCG slash commands." in text
    assert "/history" in text


def test_slash_command_completer_shows_all_commands_for_slash():
    completions = list(SlashCommandCompleter().get_completions(Document("/"), None))

    assert len(completions) == len(slash_commands())
    assert [completion.text for completion in completions[:9]] == [
        "/handoff",
        "/capabilities",
        "/summarize",
        "/context",
        "/history",
        "/details",
        "/task",
        "/model",
        "/permissions",
    ]
    assert str(completions[0].display_meta_text)


def test_slash_command_palette_order_prioritizes_handoff_groups():
    completions = ordered_slash_commands_for_palette(slash_commands(), backend="codex")

    assert [command.name for command in completions[:12]] == [
        "/handoff",
        "/capabilities",
        "/summarize",
        "/context",
        "/history",
        "/details",
        "/task",
        "/model",
        "/permissions",
        "/status",
        "/help",
        "/quit",
    ]


def test_slash_command_completer_filters_as_user_types():
    completions = list(SlashCommandCompleter().get_completions(Document("/sta"), None))

    assert [completion.text for completion in completions] == ["/status", "/stats"]


def test_slash_command_completer_suggests_task_subcommands():
    completions = list(SlashCommandCompleter().get_completions(Document("/task st"), None))

    assert [completion.text for completion in completions] == ["/task start", "/task status"]


def test_slash_command_completer_uses_current_backend_provider():
    completions = list(SlashCommandCompleter(lambda: "gemini").get_completions(Document("/rew"), None))

    assert [completion.text for completion in completions] == ["/rewind"]
    assert "Rewind Gemini conversation." in str(completions[0].display_meta_text)


def test_slash_command_completer_ignores_non_command_lines_and_arguments():
    completer = SlashCommandCompleter()

    assert list(completer.get_completions(Document("hello /"), None)) == []
    assert list(completer.get_completions(Document("/status now"), None)) == []


def test_parse_slash_command_identifies_product_commands_and_aliases():
    parsed = parse_slash_command("/stats", "claude")

    assert parsed is not None
    assert parsed.action is SlashCommandAction.PRODUCT
    assert parsed.canonical == "/status"


def test_parse_slash_command_translates_core_backend_commands():
    codex_memory = parse_slash_command("/memory", "codex")
    gemini_compact = parse_slash_command("/compact now", "gemini")
    antigravity_compact = parse_slash_command("/compact now", "antigravity")

    assert codex_memory is not None
    assert codex_memory.action is SlashCommandAction.BACKEND
    assert codex_memory.backend_prompt == "/memories"
    assert gemini_compact is not None
    assert gemini_compact.action is SlashCommandAction.BACKEND
    assert gemini_compact.backend_prompt == "/compress now"
    assert antigravity_compact is not None
    assert antigravity_compact.action is SlashCommandAction.BACKEND
    assert antigravity_compact.backend_prompt == "/compress now"


def test_parse_slash_command_handles_model_as_product_command():
    parsed = parse_slash_command("/model", "gemini")

    assert parsed is not None
    assert parsed.action is SlashCommandAction.PRODUCT
    assert parsed.canonical == "/model"


def test_parse_slash_command_handles_permissions_as_product_command():
    parsed = parse_slash_command("/permissions", "codex")

    assert parsed is not None
    assert parsed.action is SlashCommandAction.PRODUCT
    assert parsed.canonical == "/permissions"


def test_parse_slash_command_handles_capabilities_as_product_command():
    parsed = parse_slash_command("/capabilities", "gemini")

    assert parsed is not None
    assert parsed.action is SlashCommandAction.PRODUCT
    assert parsed.canonical == "/capabilities"
    assert parsed.backend_prompt is None


def test_parse_slash_command_identifies_local_utility_commands():
    parsed = parse_slash_command("/handoff claude sonnet continue", "gemini")

    assert parsed is not None
    assert parsed.action is SlashCommandAction.LOCAL
    assert parsed.canonical == "/handoff"
    assert parsed.args == "claude sonnet continue"
    assert parsed.backend_prompt is None


def test_parse_slash_command_identifies_task_command_family_as_local():
    parsed = parse_slash_command("/task start fix resume flow", "codex")

    assert parsed is not None
    assert parsed.action is SlashCommandAction.LOCAL
    assert parsed.canonical == "/task"
    assert parsed.args == "start fix resume flow"
    assert parsed.backend_prompt is None


def test_parse_slash_command_keeps_mcp_as_interactive_backend_command():
    parsed = parse_slash_command("/mcp", "antigravity")

    assert parsed is not None
    assert parsed.action is SlashCommandAction.INTERACTIVE_BACKEND
    assert parsed.backend_prompt == "/mcp"


def test_slash_command_completer_includes_antigravity_native_commands():
    completions = list(SlashCommandCompleter(lambda: "antigravity").get_completions(Document("/plug"), None))

    assert [completion.text for completion in completions] == ["/plugin", "/plugins"]


def test_parse_slash_command_passes_unknown_commands_through():
    parsed = parse_slash_command("/review", "claude")

    assert parsed is not None
    assert parsed.action is SlashCommandAction.PASSTHROUGH
    assert parsed.backend_prompt == "/review"


def test_all_core_commands_are_recognized():
    core_commands = (
        "/help",
        "/quit",
        "/clear",
        "/compact",
        "/model",
        "/plan",
        "/permissions",
        "/status",
        "/capabilities",
        "/copy",
        "/resume",
        "/init",
        "/memory",
        "/mcp",
        "/history",
        "/details",
        "/context",
        "/summarize",
        "/handoff",
        "/task",
    )

    parsed = [parse_slash_command(command, "codex") for command in core_commands]

    assert all(command is not None for command in parsed)
    assert all(command.action is not SlashCommandAction.PASSTHROUGH for command in parsed if command is not None)
