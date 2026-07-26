# Prompt Expansion

Expand short prompts into richly detailed image generation prompts using a local
LLM (Qwen3-1.7B, ~1.8 GB). The expansion model auto-downloads on first use and
is dropped from memory before diffusion runs.

## Preview Expansion

```bash
# See what expansion produces
mold expand "a cat"

# Multiple variations
mold expand "cyberpunk city" --variations 5

# JSON output
mold expand "a cat" --variations 3 --json
```

## Generate with Expansion

```bash
# Short prompt → detailed prompt → image
mold run "a cat" --expand

# Batch + expand: each image gets a unique expanded prompt
mold run "a sunset" --expand --batch 4

# Disable expansion (overrides config/env default)
mold run "a cat" --no-expand
```

## Native prepared batches

In the desktop and iPhone Create workspaces, the Batch control also sets the
expansion count. Batch 1 keeps the quick **Expand** rewrite and undo, with the
host route frozen through the next Generate or Develop. Batch 2 or greater
uses **Prepare N variations** and opens an inline review workspace before any
generation request is queued. Each prompt can be edited or removed; the whole
set can be regenerated or discarded.

On desktop, a Batch 1 rewrite whose model, family, or host changed is never a
dead Generate click: an immediate recovery notice can **Re-expand and
generate** from the original prompt on the current route, **Generate expanded
prompt anyway** as an explicit current-route override, or **Restore original**.
The notice uses readable model names, larger error copy, and a copy button.

Mold resolves the selected Create host before expansion and keeps that exact
route for every sibling. It rejects responses that do not contain exactly N
non-empty prompts. Changing the source prompt, model, host, or Batch count keeps
the reviewed prompts visible but blocks generation until you refresh or discard
them. A missing expansion model is pulled on the named host without falling
back to another machine. Create keeps that recovery inline for both quick and
prepared expansion: it shows Connecting, Starting, Queued, live percentage,
bytes, current file, ETA, and an explicit Retry expansion action when Ready.
Failed or cancelled pulls can be retried on the same host without losing the
prompt or reviewed set. On iPhone, each pull attempt temporarily leases the
frozen route, joins compatible Models work already in Starting, and releases
on terminal, stale, or superseded outcomes; Retry reacquires the route from the
same immutable recovery record. Editing or removing reviewed prompts cancels a
pending replacement instead of letting it overwrite the newer set.
Each prepared sibling records a durable batch ID and its one-based position in
the Library details panel, together with the source prompt when present.

On iPhone, the concrete route includes the selected host ID, endpoint,
Keychain-supplied API key, and server instance. The touch workspace uses 44pt
actions and 16px editors, confirms a two-to-one collapse, guards deferred source
preprocessing, and restores focus only while the replaced control still owns it.
Models and Create share one mobile download authority, so an expansion pull
cannot open a competing stream or drift to another host. When failures and an
unconfirmed cancellation coexist, the accessible outcome announces both.

## External Backend

Use an OpenAI-compatible API instead of the local LLM:

```bash
mold run "a cat" --expand \
  --expand-backend http://localhost:11434/v1

mold run "a cat" --expand \
  --expand-backend http://localhost:11434/v1 \
  --expand-model llama3
```

## Configuration

Set `MOLD_EXPAND=1` to enable expansion by default.

```toml
[expand]
enabled = true
backend = "local"
model = "qwen3-expand:q8"
temperature = 0.7

# Custom system prompt (placeholders: {WORD_LIMIT}, {MODEL_NOTES})
# system_prompt = "You are an image prompt writer..."

# Per-family tuning
[expand.families.sd15]
word_limit = 50
style_notes = "SD 1.5 uses CLIP-L (77 tokens). Use comma-separated keywords."

[expand.families.flux]
word_limit = 200
style_notes = "Write rich, descriptive natural language with atmosphere."
```

Templates can also be set via `MOLD_EXPAND_SYSTEM_PROMPT` and
`MOLD_EXPAND_BATCH_PROMPT` environment variables.
