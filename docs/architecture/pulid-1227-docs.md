# Staged agent-doc deltas for #1227

`CLAUDE.md` (and its `AGENTS.md` symlink), `README.md`,
`crates/mold-cli/src/skill/SKILL.md`, and the `website/` navigation are edited
by several PuLID issues at once, and every one of those edits lands on the same
few paragraphs. So #1227 stages its deltas here instead of racing for those
lines; whoever lands last applies this file and deletes it.

Nothing here is new policy. Each entry is an existing sentence that #1227 made
factually wrong, plus the sentence that replaces it.

---

## 1. `CLAUDE.md` — the "PuLID face extraction" bullet

**Title.** `- **PuLID face extraction is a CPU `candle-onnx` path, and the
embedding it produces is RAW.**` is no longer true of the runtime: `candle-onnx`
parses the pinned files at load and is otherwise only the parity oracle.
Replace the title with:

> **PuLID face extraction is a CPU path over resident candle modules, and the
> embedding it produces is RAW.**

**Point (1).** Replace, in full:

> (1) `candle-onnx` is CPU-only by construction — `get_tensor` places every
> initializer on `Device::Cpu` and `Gemm` builds `alpha`/`beta` there — so
> `IdentityExtractor::load` REFUSES a non-CPU device rather than demoting
> silently, and `simple_eval` re-materializes initializers every call, which is
> why the qualifying latency number is warm-repeated (halcyon p95 415.7 ms /
> plato p95 1574.5 ms against a 2.0 s budget; plato's 1.27x margin is the one to
> watch, and `pulid_face_probe bench` re-measures instead of re-arguing).

with:

> (1) The two graphs are **weight containers, not a runtime**: `identity/
> onnx_weights.rs` reads their initializers ONCE at load through the same
> authenticated bounded read, and `identity/scrfd_net.rs` /
> `identity/arcface_net.rs` run them as ordinary resident `candle-core`/
> `candle-nn` modules (#1227). `candle_onnx::simple_eval` survives only as
> `reference_forward`, the parity oracle — never a production path, because it
> re-materializes every initializer on `Device::Cpu` on every call. The
> `WeightTape` consumes parameters in **graph order** with a shape assertion on
> each, so a reordered or substituted graph fails at load naming the op index,
> and `finish()` refuses a graph carrying parameters nobody ran. Both modules
> take a device, so `IdentityExtractor::load`'s CPU refusal is now the
> ADMISSION-ORDERING contract (extraction runs before the scheduler leases a
> device) rather than an evaluator limitation — `docs/architecture/
> pulid-perf.md` §3 designs the phase that lifts it. The qualifying latency is
> still warm-repeated, 5 warmups / 20 runs, and `pulid_face_probe bench`
> re-measures instead of re-arguing: `--compare` runs both evaluators
> alternately in one loop, `--full` adds the EVA tower and the IDFormer, and
> `--regress-against` checks a p95 against the committed per-host baselines.
> **The re-materialization was not the cost centre** — measured 1.04x, not the
> 25% #1227 set out to find, because a 261 MB copy is cheap next to 23 GFLOP of
> f32 convolution. §4 records the numbers, and the EVA02-CLIP tower, unmeasured
> until #1227, is **79%** of a real extraction (2,237 ms of 2,840 ms p50 on
> halcyon) against the face stack's 13%.

**Point (2)** stays, with one clause added at the end, because the workaround is
now oracle-only:

> … until the one-condition candle fix lands. Since #1227 that rewrite only
> affects `simple_eval`, i.e. the parity oracle; the resident port reads the
> `Resize` nodes' target size from the tensor it is adding to.

**Point (5)** stays verbatim — the authenticated bounded read is unchanged and
is still the only way an extractor is constructed.

## 2. `CLAUDE.md` — the "Identity is extracted once per request" bullet

One sentence to add after "Because extraction completes and releases its ~1.4 GB
before a device is leased…":

> The ~1.4 GB peak is unchanged, but its composition moved: the recognizer's
> 261 MB is now resident tensors rather than a retained `ModelProto` plus a
> per-call copy of it, so the transient half of that stage is gone.

## 3. `crates/mold-cli/src/skill/SKILL.md`

No change. The skill documents user-facing CLI surface, and #1227 adds no flag,
env var, endpoint, or model.

## 4. `README.md`, `website/` navigation

No change, same reason. `website/guide/configuration.md` gains no row: #1227
reads no new environment variable.
