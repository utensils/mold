---
paths:
  - "apps/macos/**"
---

# Native Mac interaction contracts

Generate submission feedback is separate from the canvas following an earlier
job. Start feedback synchronously on the press, acknowledge only actual backend
acceptance, surface preparation/refusal failures, and fence late replies by the
latest press ID. Persistent acceptance text describes the acceptance event; it
must not claim a job is still running after that job completes.

Prompt editor height remembers the preferred size but clamps the rendered height
to available space. Keep the Generate action row outside scrolling controls.
Inspector headings have one full-width disclosure action with accessory buttons
as siblings; controls must fit both narrow and wide inspector widths. Test
rendered layout limits, including full UInt64 seed values.

Job Details derives actions and status from the current queue entry, preserving
host/instance/job identity fences for async responses. Hide stale running progress
when a job leaves that state. Explain control actions in plain English with
`.help`; do not claim a setting's write store is its effective value's origin.
See `apps/macos/docs/tooltip-audit.md` for the reviewed inventory and verification.
