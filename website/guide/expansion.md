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

## Desktop prepared batches

In the desktop Generate workspace, the Batch control also sets the expansion
count. Batch 1 keeps the quick **Expand** rewrite and undo, with the host route
frozen through the next Generate. Batch 2 or greater
uses **Prepare N variations** and opens an inline review workspace before any
generation request is queued. Each prompt can be edited or removed; the whole
set can be regenerated or discarded.

Mold resolves the selected Generate host before expansion and keeps that exact
route for every sibling. It rejects responses that do not contain exactly N
non-empty prompts. Changing the source prompt, model, host, or Batch count keeps
the reviewed prompts visible but blocks generation until you refresh or discard
them. A missing expansion model is pulled on the named host without falling
back to another machine. Generate keeps that recovery inline for both quick and
prepared expansion: it shows Connecting, Starting, Queued, live percentage,
bytes, current file, ETA, and an explicit Retry expansion action when Ready.
Failed or cancelled pulls can be retried on the same host without losing the
prompt or reviewed set.
Each prepared sibling records a durable batch ID and its one-based position in
the Gallery details panel.

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
