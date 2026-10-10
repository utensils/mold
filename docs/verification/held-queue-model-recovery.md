# Held model recovery verification

The production report concerned a missing reviewed H3 Turbo adapter beside an
installed base checkpoint. The saved batch child had no missing-model error
code. The repair hint therefore inspects the current acquisition destinations,
without classifying the saved error message or reading source media.

## Regression coverage

Focused Rust checks cover the missing H3 adapter, complete and unknown recipes,
custom paths including a wrong component slot, per-model inspection reuse,
held/retryable eligibility, and actual queue listing/detail responses. Explicit
null negatives veto stale legacy errors; omitted catalog hints retain their
existing typed-error recovery. OpenAPI
references and scoped core/server Clippy are checked separately.

Shared native tests cover tolerant hint decoding, exact-model and explicit
retry permission, invalid hints suppressing legacy fallback, acquisition
revalidation before and after consent, and successful retry after downloading
clears the hint. A captured legacy row cannot override a fresh invalid hint.
The iOS layout contract stacks active model downloads at every Dynamic Type
size; inactive rows retain their existing adaptive layout.

Vue tests cover hint validation, credential-specific recovery state, exact-host
action dispatch, no periodic hint probes, queue/detail/host/activity controls,
separate keyboard activation, and responsive recovery action layout. Existing
license, companion-ticket and cancellation tests remain in use.

## Native UAT

Standalone loopback fixtures supplied harmless model and queue metadata. The
actual iOS Simulator app displayed Download and Retry, Move to and Failure
Details together. Activation sent an exact-model acquisition, then one
instance/batch-scoped retry after the synthetic ticket completed. Cancelling
from Models sent a ticket-specific DELETE and did not retry the held job.

The exact long “MiniMax H3 Ref2VA Turbo 4-step (rank 21)” title and recipe ID
were visually checked in light appearance. Identity and progress use the full
row width; Cancel remains separate. Normal and maximum accessibility text sizes
were checked, including scrolling to all recovery controls. The actual macOS
app displayed the same diagnostic and recovery/transfer/details actions.

Evidence is retained locally in `/tmp/mold-held-download-uat/`, including
`UAT.md`, request logs, and screenshots. This is Simulator evidence, not a
physical iPhone claim. Mac acquisition was covered by shared tests and iOS
interaction; Mac visual UAT was read-only. Original native preferences were
restored and apps stopped after verification.

## Web and Tauri UAT

The actual web UI exercised the same synthetic download-to-retry sequence.
Desktop, 390px phone and 200% zoom views were inspected. Recovery controls wrap
independently of the model title, and the desktop action keeps an intrinsic
width. Saved error details remain available while recovery runs.

The actual Tauri iOS app exercised an exact-model download and one subsequent
retry on the owning fixture host. This exposed contradictory generic retry
copy, which now yields to the available Download and Retry action. Held mobile
actions use separate grid rows so the recovery label can use the card width.

The actual Tauri desktop app completed the same synthetic acquisition/retry.
Final accessibility inspection exposed separate Download and Retry, Details and
Cancel controls. Activating Details opened the diagnostic drawer with zero
mutation requests. The held row is a group rather than a button containing
actions. Final mobile grid styling was additionally checked using the production
component and stylesheet at 390 CSS pixels; this last CSS-only proof used a
browser after the Simulator transaction test.

No production download, generation, transfer or job retry was performed for
this UI work. The separate authorized LTX decoder qualification is documented
in `ltx25-vae-spatial-auto.md`.
