# mold — Peer Review

**Reviewer:** Bender  
**Date:** 2026-03-12  
**Commit:** d508b7a  

---

## Summary

Core inference pipeline is working (GGUF loading, device split, VAE decode). The scaffolding is solid. However there are several critical correctness bugs, zero tests, and a handful of medium-priority issues that need addressing before this is production-ready.

---

## Critical

### 1. `MoldClient::generate` is completely broken
**File:** `crates/mold-core/src/client.rs`

The client calls `.json::<GenerateResponse>()` to parse the response, but the server returns raw `image/png` bytes with `Content-Type: image/png`. They are fundamentally incompatible. The CLI (`mold generate`) will always fail with a JSON parse error.

**Fix:** `generate()` must return `Vec<u8>` (raw bytes) and the caller reconstructs `GenerateResponse`.

### 2. Panics produce empty 500s with no diagnostics
**File:** `crates/mold-server/src/routes.rs`

The generate handler uses `map_err(|e| ...)` which only catches `Err`. CUDA OOM and other runtime panics (e.g., VAE decode at 1024×1024) cause tokio to catch the panic and return `500 ""` with no body and no log. Already bit us multiple times.

**Fix:** Wrap the blocking inference call in `tokio::task::spawn_blocking` + `std::panic::catch_unwind` and map panics to proper error responses. Also add `tracing::error!` in the map_err closure.

### 3. Generate handler holds the mutex for the entire inference duration
**File:** `crates/mold-server/src/routes.rs`

`state.engine.lock().await` is held for the full 8–30s generation. Any concurrent request will queue behind it (tokio::sync::Mutex is fair). This is arguably intentional (GPU is single-threaded), but it means a hung generation blocks all other requests including `/health`. The lock should be released before blocking I/O where possible, and the inference should run on `spawn_blocking`.

### 4. No request validation
**File:** `crates/mold-core/src/types.rs`, `crates/mold-server/src/routes.rs`

`GenerateRequest` has no bounds checking. A request with `width: 0` or `steps: 0` produces cryptic candle panics instead of a 422. Minimum viable validation:
- `width` and `height`: must be > 0, must be multiples of 16 (FLUX requirement), and capped at 768 (current OOM limit)
- `steps`: must be >= 1
- `prompt`: must not be empty

---

## High Priority

### 5. CLIP prompt not truncated to 77 tokens
**File:** `crates/mold-inference/src/engine.rs`

CLIP has a hard maximum of 77 tokens. The code passes raw token IDs without truncation. Long prompts will silently produce incorrect output or panic inside CLIP's attention mechanism.

```rust
// Add before building input_ids:
tokens.truncate(77);
```

### 6. `is_schnell` detected by string match on model name
**File:** `crates/mold-inference/src/engine.rs`

`self.model_name.contains("schnell")` is fragile. A model named `"my-schnell-dev"` would incorrectly use the schnell schedule; a GGUF named `"flux1-schnell-Q8_0.gguf"` would only match if the model_name also contains "schnell". Should use an explicit enum `ModelFamily::Schnell | ModelFamily::Dev` resolved at load time.

### 7. T5 config hardcoded as raw JSON string
**File:** `crates/mold-inference/src/engine.rs`

The T5-XXL config is embedded as a 400-char JSON string literal. If candle's `t5::Config` gains or removes fields this silently breaks (or panics on deserialization). Should be a typed `const` or built from a struct literal, not deserialized from a string that can't be checked at compile time.

### 8. `unsafe` mmap with no path existence check
**File:** `crates/mold-inference/src/engine.rs`

All four `VarBuilder::from_mmaped_safetensors` calls are `unsafe` and produce unreadable errors if the path doesn't exist. Add a simple pre-check:

```rust
if !path.exists() {
    bail!("model file not found: {}", path.display());
}
```

### 9. Default CLI width/height is 1024 but 1024 causes OOM
**File:** `crates/mold-cli/src/main.rs`

`--width` and `--height` default to 1024 in the CLI, but 1024×1024 panics (VAE OOM) on the current setup. The default should match the safe operating range: **768**.

### 10. `candle` pinned to `branch = "main"`
**File:** `crates/mold-inference/Cargo.toml`

`branch = "main"` is non-reproducible — a `cargo update` on a different day can silently pull in a breaking candle commit. Pin to a specific `rev = "..."`.

---

## Medium Priority

### 11. Dead error types: `MoldError` and `InferenceError` are unused
**Files:** `crates/mold-core/src/error.rs`, `crates/mold-inference/src/error.rs`

Both error enums are defined but the actual code uses `anyhow::Error` throughout. Either use them (makes error handling more structured and allows `match` on error variants) or remove them to reduce confusion.

### 12. Dead API client methods hit non-existent endpoints
**File:** `crates/mold-core/src/client.rs`

`MoldClient::load_model()` and `unload_model()` call `/api/models/load` and `/api/models/{model}` which don't exist in the router. They'll always return 404. Either implement the routes or remove the methods.

### 13. `mold serve` CLI command is a silent stub
**File:** `crates/mold-cli/src/commands/serve.rs`

Prints a warning and exits with Ok(()). Users who run `mold serve` get no actual server. Should either call `mold_server::run_server()` directly (it's a library crate, this is already possible) or return an explicit error.

### 14. `mold pull` simulates a download
**File:** `crates/mold-cli/src/commands/pull.rs`

Runs a fake progress bar and exits. Fine as a placeholder, but the fake progress bar is confusing — it implies the model was downloaded when nothing happened.

### 15. Config TOML parse errors swallowed silently
**File:** `crates/mold-core/src/config.rs`

```rust
Err(_) => Config::default(),  // silently ignores parse error
```

Should `tracing::warn!` or at minimum `eprintln!` so users know their config is being ignored.

### 16. `hf-hub` and `uuid` are unused dependencies
**Files:** `crates/mold-inference/Cargo.toml`, `crates/mold-core/Cargo.toml`

`hf-hub` is pulled into mold-inference but never called (no download code exists). `uuid` is in mold-core but not used in any visible code. Both increase compile time and binary size for no benefit.

### 17. GPU info always null
**File:** `crates/mold-server/src/routes.rs`

`ServerStatus.gpu_info` is always `None`. `nvidia-smi` or candle's device API can provide this. Low priority but `mold ps` showing "GPU: not detected" is misleading.

---

## Minor / Style

### 18. Redundant `.map_err(anyhow::Error::from)` in `FluxTransformer::denoise`
`flux::sampling::denoise` returns `candle_core::Result<Tensor>` which auto-converts via `?`. The explicit `.map_err(anyhow::Error::from)` is unnecessary.

### 19. `t5_emb.clone()` in the non-quantized path wastes a copy
In the non-quantized branch: `(t5_emb.clone(), clip_emb.clone(), img.clone())` — the originals are never used again, so move instead of clone.

### 20. `LoadedFlux.cpu` device stored unnecessarily
`Device::Cpu` is a zero-cost value (`Device::Cpu` is a unit variant). Storing it in `LoadedFlux` adds no value — just use `Device::Cpu` inline during generate.

### 21. `model_registry::find_model` is dead code

### 22. `LoadModelRequest` in types.rs is orphaned
Only used by dead client methods.

### 23. No README.md at repo root
CLAUDE.md exists but GitHub shows no README on the repo landing page.

---

## Test Coverage: Zero

No tests exist anywhere. Minimum required:

| What | Where | Priority |
|------|-------|----------|
| `OutputFormat` parsing (png, jpeg, jpg, invalid) | `mold-core` | High |
| `GenerateRequest` validation bounds | `mold-core` | High |
| `Config::load_or_default` with missing/corrupt file | `mold-core` | High |
| `ModelPaths::resolve` env var precedence over config | `mold-core` | Medium |
| API route unit tests (mock engine, check HTTP codes) | `mold-server` | High |
| `model_registry::known_models` returns expected entries | `mold-inference` | Low |

---

## Fixes Implemented

See commits following this review.
