---
title: Video Generation
---

# Video Generation

mold supports generating video clips using LTX Video and LTX-2 models. LTX-2 distilled models support chaining multiple clips together for longer videos with scene-by-scene direction.

## Single-clip generation

```bash
mold run ltx-2-19b-distilled:fp8 "a cat walks through autumn leaves" --frames 97
```

## Multi-prompt scripts (v2)

Direct any-length video scene-by-scene with a TOML script. Each prompt becomes a stage; each boundary has a `transition` (`smooth`, `cut`, or `fade`).

### Canonical form

```bash
mold run --script shot.toml
mold run --script shot.toml --dry-run   # print stage summary, don't submit
mold chain validate shot.toml            # parse without submitting
```

### Sugar form (uniform smooth chains)

```bash
mold run ltx-2-19b-distilled:fp8 \
  --prompt "a cat walks into the autumn forest" \
  --prompt "the forest opens to a clearing" \
  --prompt "a spaceship lands" \
  --frames-per-clip 97
```

Per-stage transitions or per-stage frames require `--script`.

### Transitions

| Mode | Behavior |
|------|----------|
| `smooth` *(default)* | Motion-tail carryover — prompt change produces a visual morph between scenes. |
| `cut` | Fresh latent, no carryover. If the stage has `source_image`, it's used as an image-to-video seed. |
| `fade` | Cut + post-stitch alpha blend of `fade_frames` (default 8) on each side of the boundary. |

### Example `shot.toml`

```toml
schema = "mold.chain.v1"

[chain]
model = "ltx-2-19b-distilled:fp8"
width = 1216
height = 704
fps = 24
seed = 42
steps = 8
guidance = 3.0
strength = 1.0
motion_tail_frames = 25
output_format = "mp4"

[[stage]]
prompt = "a cat walks into the autumn forest"
frames = 97

[[stage]]
prompt = "the forest opens to a clearing"
frames = 49

[[stage]]
prompt = "a spaceship lands"
frames = 97
transition = "cut"

[[stage]]
prompt = "the cat looks up in wonder"
frames = 97
transition = "fade"
fade_frames = 12
```

### Web composer

The web UI at `/generate` has a **Script** mode toggle. Switch to Script mode for a card-based editor where each stage gets its own prompt, frame count, and transition selector. Supports drag-reorder, per-stage prompt expansion, and TOML import/export.

### TUI script mode

Press `s` from the main TUI hub to open Script mode. Key bindings:

| Key | Action |
|-----|--------|
| `j` / `k` | Navigate stage list |
| `a` / `A` | Add stage after current / at end |
| `d` | Delete current stage (confirm) |
| `J` / `K` | Move stage down / up |
| `t` | Cycle transition (smooth → cut → fade) |
| `i` | Edit prompt |
| `f` | Edit frames |
| `Enter` | Submit chain |
| `Ctrl-S` | Save as TOML |
| `Ctrl-O` | Load TOML |

### Capabilities endpoint

```
GET /api/capabilities/chain-limits?model=<name>
```

Returns per-model caps used by all composer UIs:

```json
{
  "model": "ltx-2-19b-distilled:fp8",
  "frames_per_clip_cap": 97,
  "max_stages": 16,
  "max_total_frames": 1552,
  "fade_frames_max": 32,
  "transition_modes": ["smooth", "cut", "fade"]
}
```
