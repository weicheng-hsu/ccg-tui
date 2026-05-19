<p align="center">
  <img src="docs/assets/ccg_tui_logo.png" alt="CCG TUI Logo" width="250" />
</p>

# CCG TUI

CCG TUI is a vendor-agnostic terminal workspace for developers who use multiple
coding agent CLIs. It lets you run Codex, Claude Code, and Gemini CLI from one
consistent interface while CCG owns the transcript, local workflow commands,
resume context, and cross-backend handoff packets.

The project is useful today as a local operator console, with deliberately
conservative boundaries around automation: CCG can advise on backend fit and
prepare handoff context, but it does not silently switch backend, model, or
permissions.

## 🎥 Demo

![CCG TUI demo](docs/assets/ccg-tui-demo.gif)

The demo shows the fullscreen backend picker, explicit model and permission
control, local task state, normalized backend activity details, a summary
checkpoint, advisory capability registry, and a preview-only Claude handoff
packet. It is recorded with the deterministic fake backend, so it can be
regenerated without Codex, Claude, or Gemini credentials:

```bash
uv run python scripts/record_readme_demo.py
```

## 💡 Why It Exists

Modern coding assistants are strong in different situations. CCG TUI keeps
backend choice operational instead of product-specific: start with the CLI that
fits the task, keep a local transcript that is independent of vendor storage,
and hand off bounded working context when another backend should continue.

## ✨ What Works

- Fullscreen `prompt_toolkit` TUI with backend, model, and permissions pickers.
- One-shot prompt mode and a line-by-line fallback UI.
- Backend adapters for `codex`, `claude`, and `gemini`.
- Persistent terminal-backed backend sessions during interactive use.
- Normalized output, activity, failures, vendor session ids, and transcript
  events across supported backends.
- CCG-owned slash commands for status, local history, resume context,
  summaries, capabilities, task boundaries, and handoff preview.
- Backend-native slash-command passthrough with selected compatibility
  translations.
- JSON transcripts under `runtime/transcripts/`.
- Gemini-backed summary checkpoints.
- First-turn local resume context injection.
- Manual cross-backend handoff packet preview/export and explicit
  target-session execution with lineage metadata.
- Advisory backend capability and permission compatibility notes.

## 🚧 Boundaries

The current implementation intentionally keeps these areas out of scope:

- automatic backend switching
- automatic model or permission changes
- vendor-native cross-backend resume
- product-owned MCP orchestration
- product-owned skills orchestration
- subagent orchestration
- cross-session search

## 🚀 Quick Start

Prerequisites:

- Python 3.12+
- `uv`
- one or more supported vendor CLIs on `PATH`: `codex`, `claude`, `gemini`
- vendor-native authentication completed in whichever CLIs you plan to use

Install dependencies:

```bash
uv sync --dev
```

Launch the fullscreen TUI and choose a backend:

```bash
uv run ccg-tui
```

Start directly with a backend:

```bash
uv run ccg-tui --backend codex
uv run ccg-tui --backend claude
uv run ccg-tui --backend gemini
```

Run a single prompt:

```bash
uv run ccg-tui --backend codex --prompt "Summarize this repository"
```

Use the simple fallback UI:

```bash
uv run ccg-tui --simple-ui
```

## ⌨️ Interactive Controls

| Input | Behavior |
| --- | --- |
| `Enter` | Submit the current prompt. |
| `Shift-Enter` | Add a newline when the terminal reports modified Enter. |
| `Esc` then `Enter` | Add a newline fallback for terminals without modified Enter support. |
| `Ctrl-J` | Submit fallback for terminals that do not expose modified Enter cleanly. |
| `F2` | Refresh the conversation pane. |
| `F3` or `/details` | Toggle expanded backend activity metadata. |
| `Esc` or `Ctrl-C` | Exit the current session or cancel a picker. |

## ⚡ Slash Commands

CCG routes slash commands through one registry:

| Command group | Examples | Behavior |
| --- | --- | --- |
| Product commands | `/help`, `/quit`, `/clear`, `/model`, `/permissions`, `/status`, `/capabilities`, `/copy`, `/resume` | Handled by CCG before backend prompt submission. |
| Local workflow commands | `/history`, `/details`, `/context`, `/summarize`, `/handoff`, `/task` | Update local state, show local artifacts, or run local actions. |
| Backend commands | `/compact`, `/plan`, `/init`, `/memory`, `/mcp`, `/review` | Translated or passed through to the active backend when appropriate. |

Common aliases include `/exit` for `/quit`, `/new` and `/reset` for `/clear`,
`/commands` for `/help`, `/stats` and `/cost` for `/status`, and `/continue`
for `/resume`.

Task commands manage local task boundaries without sending prompts to the
backend:

```text
/task start [title...]
/task status
/task close [note...]
```

## 🧠 Models And Permissions

Open the in-TUI pickers:

```text
/model
/permissions
```

Or set values directly:

```text
/model gpt-5.5
/model sonnet
/model gemini-2.5-flash
/permissions ask
/permissions plan
/permissions auto-edit
/permissions full-access
```

Model names are examples. Actual availability is controlled by the installed
vendor CLI and the account or project it is authenticated against.

The default permission preset is `ask`.

| Preset | Codex | Claude | Gemini |
| --- | --- | --- | --- |
| `plan` | `approval_policy=on-request`, `sandbox_mode=read-only` | `permission_mode=plan` | `approval_mode=plan` |
| `ask` | `approval_policy=on-request`, `sandbox_mode=workspace-write` | `permission_mode=default` | `approval_mode=default` |
| `auto-edit` | `approval_policy=never`, `sandbox_mode=workspace-write` | `permission_mode=acceptEdits` | `approval_mode=auto_edit` |
| `full-access` | `approval_policy=never`, `sandbox_mode=danger-full-access` | `permission_mode=bypassPermissions` | `approval_mode=yolo` |

Permission presets are explicit. CCG never widens access automatically while
preparing routing advice, resume context, or handoff packets.

## 🔄 Sessions, Resume, And Handoff

List local transcripts:

```bash
uv run ccg-tui --list-sessions
```

Create a Gemini-backed summary checkpoint:

```bash
uv run ccg-tui --summarize-session <session-id>
```

Resume a local CCG transcript:

```bash
uv run ccg-tui --resume-session <session-id>
uv run ccg-tui --resume-session <session-id> --resume-context off
uv run ccg-tui --resume-session <session-id> --resume-context-turns 4
```

Local resume starts a fresh vendor-native process and appends new turns to the
same CCG transcript. CCG injects a bounded resume-context preamble into the
first new prompt by default while preserving the user's original prompt in the
transcript. A local session remains single-backend; use handoff to start a new
session on another backend.

Preview or export a handoff packet:

```bash
uv run ccg-tui --handoff-session <session-id> --target-backend claude --target-model sonnet --handoff-goal "Continue the implementation"
uv run ccg-tui --handoff-session <session-id> --target-backend gemini --handoff-output runtime/handoffs/packet.txt
```

Start a new target-backend session from that packet:

```bash
uv run ccg-tui --handoff-session <session-id> --target-backend codex --handoff-goal "Continue from this packet" --handoff-execute
```

Handoff is not vendor-native resume. It is an explicit CCG workflow that builds
bounded portable context and creates a new session only when
`--handoff-execute` is supplied.

## 💾 Transcript Storage

Transcripts are stored as JSON files:

```text
runtime/transcripts/<session-id>.json
```

Each persisted transcript includes workspace metadata, backend sessions, turns,
normalized activity events, vendor session ids, failures, recovery metadata,
resume context metadata, task records, summary checkpoints, and lineage edges.

If a backend process stops without a terminal success or failure event, CCG
persists the latest turn as `failed` with `error.kind=interrupted` and
`metadata.recovery.state=interrupted`. Resume and handoff treat partial output
as inspectable but non-authoritative.

## 📄 License

CCG TUI is released under the MIT License. See [LICENSE](LICENSE).
