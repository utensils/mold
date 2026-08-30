- **Video families now default to FlashAttention.** Wan and LTX-2 resolve their
  attention backend through a new per-family policy: with the FlashAttention v2
  kernel compiled in, an unset `MOLD_ATTN` reaches a video DiT as `flash`
  instead of the hand-rolled math path. Measured **2.1x** on the Wan DiT
  (158.4 s to 75.3 s, `wan22-t2v-a14b:q5` 53 frames at 832x480 on an RTX 4090).
  Image families are unchanged and still default to `math`, so an archived
  still renders the same bytes it always did; `MOLD_ATTN=math` restores the old
  arithmetic everywhere.
- **Wan step caching is on by default.** `MOLD_WAN_STEP_CACHE` now defaults to
  `auto` rather than `off`. First-block residual reuse was already measured at
  **1.85x** on the non-distilled tiers (605.6 s to 327.4 s, `wan22-t2v-a14b:q8`
  33 frames at 832x480, 20 steps) and already refuses itself on distilled
  adapters and schedules under 12 steps, so the default only engages where it
  was qualified. `MOLD_WAN_STEP_CACHE=off` disables it.
