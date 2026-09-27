---
paths:
  - "crates/mold-core/src/metal_memory.rs"
  - "crates/mold-cli/src/commands/metal_memory*.rs"
  - "studio/api/metalMemory.ts"
  - "studio/components/DevicePanel.vue"
  - "website/guide/metal-memory.md"
  - "docs/metal-memory-policy-*.md"
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
