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
