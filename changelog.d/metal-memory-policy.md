- **Respect macOS Metal memory limits.** GPU admission and memory planning use
  Metal's working-set recommendation, an explicit `iogpu.wired_limit_mb` limit,
  and live host headroom while preserving installed RAM in hardware inventory.
  Device telemetry exposes the effective budget separately from system RAM.
- **Explicit local Metal administration.** `mold system metal-memory` inspects,
  sets or resets this Mac's wired limit with verified readback; optional
  `--persist` installs or removes Mold's owned boot policy without elevating
  the inference server.
