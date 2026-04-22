# SD3 CLIP-L/G 77-token truncation bug — handoff

> Paste the prompt at the bottom of this file into a fresh Claude Code session
> to pick up where this one left off.

## TL;DR

`sd3.5-large:q8` (and every other `sd3*` family) fails with
`"shape mismatch in broadcast_add, lhs: [1, N, 768], rhs: [1, 77, 768]"`
whenever the prompt tokenises to `N > 77` tokens. Observed three times in
killswitch's server log over the last few hours (seq lengths 130, 131, 132).
The shared SD1.5/SDXL encoder path already truncates to 77 correctly; the
SD3-specific wrapper regressed.

## Repro

Any sd3 family, any prompt long enough to exceed 77 CLIP tokens:

```bash
# Remote:
curl -sS -X POST http://beast:7680/api/generate \
  -H 'content-type: application/json' \
  -d '{"model":"sd3.5-large:q8","prompt":"'"$(python3 -c 'print("a highly detailed " * 30)')"'","width":1024,"height":1024,"steps":20,"guidance":4.0}' | head -c 300
# → HTTP 500 with "shape mismatch in broadcast_add, lhs: [1, 132, 768], rhs: [1, 77, 768]"
```

Same bug on `--local` once an sd3 engine builds.

## Root cause (with file:line)

**`crates/mold-inference/src/encoders/sd3_clip.rs:86-97`** — the
`ClipWithTokenizer::encode_text_to_embedding` method tokenises the prompt
and pads UP to `max_position_embeddings` (77) but never truncates DOWN when
the tokeniser returns more than 77 ids:

```rust
let mut tokens = self.tokenizer.encode(prompt, true)...get_ids().to_vec();
let eos_position = tokens.len() - 1;        // ← overshoots when len > 77
while tokens.len() < self.max_position_embeddings {
    tokens.push(pad_id);                    // ← pads up only
}
let tokens = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
let (_, text_embeddings_penultimate) =
    clip.forward_until_encoder_layer(&tokens, usize::MAX, -2)?;
// …
last_hidden.i((0, eos_position, ..))?     // ← also out-of-bounds when len > 77
```

CLIP's position-embedding table holds exactly 77 entries. When the token
tensor is `[1, 132]`, the internal `embedding + position_embedding` add
hits `lhs: [1, 132, 768], rhs: [1, 77, 768]`, which is the error the user
surfaced. `eos_position = tokens.len() - 1` is also out-of-bounds for the
pooled-output slice when it happens to not panic before the add fires.

**Compare to the correct shared path** at
`crates/mold-inference/src/encoders/clip.rs:105-107`:

```rust
let mut tokens = ...get_ids().to_vec();
// CLIP hard limit: 77 tokens (including BOS/EOS)
tokens.truncate(77);
```

Same CLIP tokeniser, same 77-limit constant — the SD3 wrapper just
forgot to call `truncate`.

**Blast radius.** `ClipWithTokenizer` is used for BOTH SD3's CLIP-L
(`encoders/sd3_clip.rs:232` — `encode_text_to_embedding` on `clip_l`) AND
CLIP-G (`:236` — same method on `clip_g`). Fixing the helper once fixes
both. Every `sd3*` model is affected:

- `sd3.5-large:q8` (observed — three failures)
- `sd3.5-large:fp16`
- `sd3.5-medium:*`
- `sd3-large:*` (base sd3, if still in the manifest)

SD1.5, SDXL, FLUX, Flux.2, Z-Image, LTX-Video, LTX-2, Qwen-Image,
Wuerstchen — all unaffected (different encoders, different truncation
paths already verified).

## The fix (sketch — do not paste blindly)

```rust
let raw_tokens = self.tokenizer.encode(prompt, true)
    .map_err(|e| anyhow::anyhow!("CLIP tokenization failed: {e}"))?
    .get_ids()
    .to_vec();

// Truncate to max_position_embeddings, preserving EOS as the last slot.
// CLIP's pooled output reads from the EOS position; losing EOS breaks
// the pooled branch silently.
let eos_id = *raw_tokens.last().unwrap_or(&pad_id);
let mut tokens = raw_tokens;
if tokens.len() > self.max_position_embeddings {
    tokens.truncate(self.max_position_embeddings);
    *tokens.last_mut().expect("non-empty after truncate") = eos_id;
}
let eos_position = tokens.len() - 1;
while tokens.len() < self.max_position_embeddings {
    tokens.push(pad_id);
}
```

Three subtle points to not miss:

1. **EOS preservation** — CLIP's pooled output pulls from the EOS-position
   hidden state. If you just `tokens.truncate(77)` you lose EOS when the
   raw length exceeds 77; the pooled branch then reads a content token's
   hidden state, which changes output. The shared encoder at
   `encoders/clip.rs:107` gets away with a bare `truncate(77)` because
   that path doesn't compute a pooled-at-EOS output at all — it returns
   only the last-layer `forward(...)` result.
2. **`eos_position` recompute** — it's fine to leave as
   `tokens.len() - 1` after truncate-then-pad, but only because the pad
   step happens after the truncate. If you flip the order, `eos_position`
   lands in the pad region.
3. **Don't silently warn** on overlong prompts unless logging is cheap —
   users may hit this with an expanded prompt that's 80 tokens. Prefer a
   single `tracing::debug!` with the truncation count so the CLI doesn't
   spam on every generation.

## Verification

Weight-free unit test (no candle weights — small synthetic model is fine):

1. Build a `ClipWithTokenizer` against the real SDXL CLIP-L tokenizer JSON
   (it's a file-based tokeniser, no weights needed to tokenise).
2. Run `encode_text_to_embedding` on a 50-token prompt and a 200-token
   prompt in the same test. Assert both return `[1, 77, 768]` penultimate
   and `[768]` pooled (the pooled shape is actually `[768]` after the `i`
   slice at line 104 — confirm).
3. Actually this test path is weight-bearing (`clip.forward_until_encoder_layer`
   needs real CLIP weights). Split into a pure tokeniser-level test that
   just verifies `tokens.len() == 77` after the new truncation logic, with
   `tokens[76] == eos_id` when the raw tokeniser output exceeded 77.

Integration test: hit `http://beast:7680/api/generate` against
`sd3.5-large:q8` with a 300-char prompt after the fix ships. Expected:
200 + image bytes, no shape-mismatch error in `~/.mold/logs/server.log`.

## Render-chain v1 context (the other active thread)

This session just finished **render-chain v1**. The state:

- Branch `feat/render-chain-v1` is pushed to origin, 12 commits ahead of
  `origin/main`. PR not opened yet — the URL stub is
  `https://github.com/utensils/mold/pull/new/feat/render-chain-v1`.
- Last commit on the branch: `766322e fix(cli): pass owned String to
  create_engine in local chain path` — caught only when the CUDA build
  ran on killswitch (Phase 3 feature-matrix check omitted `cuda`/`metal`).
- killswitch is running the new binary at `766322e` as PID 1199380
  (`./target/release/mold serve --bind 0.0.0.0 --gpus 0,1`), logs at
  `~/.mold/logs/server.log`. `MOLD_HOST=http://beast:7680` reaches it.
- Local `main` is parity with `origin/main` at `1410d08`. The chain work
  lives on the feature branch only.
- All four plan phases landed with tests green: `mold-ai-core` 611 pass,
  `mold-ai-inference` lib 588 pass, `mold-ai-server` lib 186 pass (+5
  chain route tests), `mold-ai` 369 unit + 38 integration + 11 runpod
  (+12 chain CLI tests). `cargo fmt/clippy --workspace -- -D warnings`
  clean. Website `bun run fmt:check/verify/build` clean.

This SD3 CLIP-L/G truncation bug is **NOT** related to render-chain — it
fires on the single-clip SD3 path (`mold_server::gpu_worker::process_job`).
Fixing it is an independent fix that can land on `main` directly, or
stack on top of `feat/render-chain-v1` and go out as part of the chain
PR. Your call.

## Branch / commit layout

```
6211182 (origin/feat/render-chain-v1~1) docs(chain): …
766322e (origin/feat/render-chain-v1, HEAD of feat branch) fix(cli): pass owned String …
1410d08 (origin/main, local main) fix(flux): surface city96-GGUF …
```

---

## The prompt

Paste from here into a fresh Claude Code session:

---

I need to fix a CLIP-L/G prompt-truncation bug in the mold repo
(`/Users/jeffreydilley/github/mold`) that currently makes every `sd3*`
model (SD 3.5 Large q8 confirmed) return HTTP 500 with
`"shape mismatch in broadcast_add, lhs: [1, N, 768], rhs: [1, 77, 768]"`
for any prompt that tokenises to more than 77 CLIP tokens.

## Read first

1. `tasks/sd3-clip-77-truncation-handoff.md` — full diagnosis, file:line
   cites, repro curl, fix sketch, verification plan. Read end-to-end.
2. `crates/mold-inference/src/encoders/sd3_clip.rs:60-131` — the buggy
   function (`ClipWithTokenizer::encode_text_to_embedding`).
3. `crates/mold-inference/src/encoders/clip.rs:100-115` — the shared
   path that correctly truncates. Contrast to see the regression.

## Status on entry

- Branch `main` locally and on origin at `1410d08`.
- `feat/render-chain-v1` is on origin at `766322e` (12 commits ahead of
  main). It's unrelated to this bug but is a parallel in-flight PR;
  ignore unless you're explicitly asked to stack the fix on it.
- killswitch (BEAST, dual-3090) is running `766322e` as
  `mold serve --bind 0.0.0.0 --gpus 0,1`. Reproduce against it via
  `MOLD_HOST=http://beast:7680`. Log tail:
  `ssh killswitch@192.168.1.67 "tail -f ~/.mold/logs/server.log"`.
- `CLAUDE.md` claims `mold-inference`/`mold-server` have
  `[lib] test = false` — **stale, tests run normally**. Verified in
  render-chain v1's Phase 1d–4 landings.
- Pre-existing clippy warnings unrelated to this bug (do NOT fix):
  `manual_repeat_n` in `mold-core/src/download.rs:1451`,
  `field_reassign_with_default` in `mold-core/src/placement_test.rs:167`.

## What you're doing

Fix the bug per the handoff's "The fix" section. Short commit, targeted
test, verify on killswitch. One atomic commit, `fix(sd3): truncate CLIP
token sequences to 77 with EOS preserved` (or similar).

Do NOT push unless the user asks. Do NOT stack on `feat/render-chain-v1`
unless asked — land on `main` as a standalone fix so it can merge
independently of the render-chain PR.

Verify with:

```bash
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
cargo test -p mold-ai-inference --lib
cargo test -p mold-ai-core
# + a new unit test you add for the truncation helper
```

Then optionally rebuild on killswitch and retry the repro curl against
`http://beast:7680` to confirm the user-surfaced symptom is gone.

## Process

- Use `superpowers:systematic-debugging` — the root cause is already
  in the handoff, but the skill will keep you honest about verifying.
- Use `superpowers:test-driven-development` — write a failing test for
  the 200-token case first, then fix, then watch it go green.
- Use `superpowers:verification-before-completion` before claiming done.

If you discover the bug is broader than the handoff claims (e.g. it
affects another encoder I missed), stop and re-scope rather than
silently widening the fix.
