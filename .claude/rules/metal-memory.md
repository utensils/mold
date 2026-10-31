---
paths:
  - "crates/mold-core/src/metal_memory.rs"
  - "crates/mold-cli/src/commands/metal_memory*.rs"
  - "studio/api/metalMemory.ts"
  - "studio/components/DevicePanel.vue"
  - "website/guide/metal-memory.md"
  - "docs/metal-memory-policy-*.md"
  - "crates/mold-inference/src/weight_loader.rs"
  - "crates/mold-server/src/memory_preflight.rs"
  - "crates/mold-server/src/gpu_worker.rs"
---

# macOS Metal memory policy

Moved from the root CLAUDE.md; loaded only when working under the paths above.

`mold_core::metal_memory::MetalMemorySnapshot` is the single budget authority:
capacity is min(Metal recommendation, positive uint32 sysctl MiB, RAM minus
max(15%, 8 GiB)); incremental headroom also charges existing Metal allocations
and live Mach free+inactive. Hardware RAM totals and CUDA attribution remain
separate. Native read-only sampling uses the memoized Candle Metal device;
failed supported probes block admission, absent optional sysctl can use the
recommendation. Reclaim credits are bounded and post-drop guards resample after
pool release. Do not restore a RAM-only fallback or elevate a server.
`mold system metal-memory` routes before config/DB startup and always targets
this machine; status is read-only, set/reset require explicit root invocation.
`--persist` owns only the fixed root LaunchDaemon, never an executable from a
user path. Shared DevicePanel reads optional host telemetry; remote clients have
no kernel mutation control. See `website/guide/metal-memory.md` and the reviewed
`docs/metal-memory-policy-plan.md` for contracts and validation limits.

Metal computes in F32, so a stored F16 or BF16 weight widens on load. That goes
through `weight_loader::load_stored_tensor`, which reads and casts on the CPU and
uploads once at exact size; any new Metal weight backend must call it too. Never
restore upload-then-cast: candle's pool rounds the cast output up to the next
power of two and keeps the upload until the next queue wait, which took a 12.3 GB
Z-Image checkpoint to 89.7 GiB. Only lossless widening (`widening_is_lossless`)
leaves the device kernel; narrowing casts keep it, because their rounding can move
pixels. A failed render must still sweep. A failed command buffer surfaces from
the wait that collects it, and that wait returns before the pool sweep, so the
worker calls `release_pooled_metal_memory` after every generation that returns,
and that function retries the wait once. Without this, admission reads zero
headroom and idle reclaim finds nothing to evict, because the requested model is
exempt. Z-Image and FLUX.2 admission charge dense half-precision transformers at
their widened size (`metal_transformer_widening_bytes`) and charge nothing extra
under block offload (`resident_transformer_widening_bytes`).
