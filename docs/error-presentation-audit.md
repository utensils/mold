# Error presentation audit and implementation plan

## Findings

The reported queue error is an execution-plan refusal with an H3 memory budget,
not a claim about the phone’s RAM. It reaches both native apps through
`QueueEntry.waitDescription`, while iOS also repeats it through `QueueHeldActions`.
HTTP refusals and rendering streams carry raw server reasons too. Native app
`Error+Sentence` helpers duplicate policy. Web, Tauri desktop and mobile share
`studio/lib/errors.ts`, but queue details still bypass that formatter.

| Surface                            | Existing boundary                                   | Planned handling                                                                   |
| ---------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Server HTTP errors                 | `ApiError.into_response`                            | Format once, log the original with its code                                        |
| Server queue and batch projections | wire serialization                                  | Format error/held reasons without changing stored diagnostics or state             |
| Server generation SSE              | `SseErrorEvent.message`                             | Same shared Rust formatter; keep retained flag and codes                           |
| Other server job statuses          | error fields in shared wire types                   | Same serializer for downloads, chain, mesh and upscale failures                    |
| Native Mac and iOS                 | MoldClient + app sentence helpers                   | One shared Foundation formatter, including old-server queue/batch/download replies |
| Web, Tauri desktop and mobile      | studio error utilities and queue details            | Same contract fixtures and concise presentation; retain raw clipboard diagnostics  |
| Local native export/input failures | app Error helpers and direct localized descriptions | Shared formatter; preserve already actionable short system messages                |
| Scheduler telemetry                | typed blocked reasons and resource estimates        | Preserve structured values; no presentation parsing changes to scheduling          |

## Plan

1. Pin the screenshot, memory units, runtime failures, actionable validation and
   unknown diagnostics with failing cross-language contract tests.
2. Add a server-owned formatter in mold-core. At wire boundaries only, format
   known memory refusals with decimal GB/MB, explicitly label the estimate and
   available budget, and show the shortfall. This is a budget estimate, not a
   guarantee that freeing that exact amount will make the render fit.
3. Keep raw diagnostics in server logs and durable records. Preserve codes,
   retryability, lifecycle and licence payloads. Never reclassify scheduling
   or admission from friendly prose.
4. Add one native MoldClient fallback and one studio fallback for older servers.
   Share wording through `docs/contracts/user-errors.json`. Short actionable
   validation stays intact; implementation traces become concise log guidance.
5. Remove the duplicate held error from the iOS row and let the reason use the
   full available row width. Job details retain the summary and controls.
6. Validate shared Swift/Rust/studio contracts, native builds and iPhone queue
   UAT, including accessibility sizes. Obtain independent peer review after
   implementation, address findings, publish the PR and verify exact-head CI.

## Boundaries

This changes presentation only. Stored failures and error codes retain their
original diagnostic meaning. The compatibility parser recognizes explicit
byte-unit tokens; it never reformats seeds, dimensions, counts or arbitrary
numbers. Unknown short actionable sentences pass through; unknown long or
multiline diagnostics direct the user to logs. Typed scheduler telemetry and
licence consent flows retain their existing authority.

## Review findings addressed

Independent review found four issues: portable mesh stages must retain raw
errors, GPU quarantine/cooldown advice must survive summarization, download SSE
needs the same formatter as snapshots, and local native failures need local logs
and wording. Regression coverage now includes all four. Short LoRA and missing
video-tool failures retain specific recovery guidance. Native private OSLog
fields protect paths/URLs/credentials while publishing the error domain/code.

## Verification

The screenshot’s original byte amounts yield a **248.34 MB** shortfall. iPhone
Simulator UAT proved a single visible/spoken held reason at extra-small, Large
and AX5, with no raw byte values or execution-plan diagnostic in the view tree.
The loopback fixture performed no generation. Rust/Swift/TypeScript fixtures
cover byte units, graphics/system/shared pools, runtime failures, restart,
cooldown, unsupported hardware, LoRA/video-tool recovery, and plain validation.

### Validation evidence

- Core: 1,937 tests passed (one ignored); shared Swift: 1,227 tests passed.
- Studio: all 2,310 tests passed; targeted web generation/error tests and
  desktop error/queue tests passed (47 desktop assertions).
- Native macOS build, clean native iOS build and iPhone queue UAT passed.
  Both native architecture lints and core/server Clippy passed.
- Web and Tauri desktop production builds passed.
- Broader frontend suites expose existing image-input test failures on main:
  the unchanged ImagePickerModal baseline fails the same three upload mocks;
  unchanged MobileSourceControls, IdentityWell and IPC tests reproduce three
  more existing failures. These are outside error presentation.
- The broad local server suite encounters two-second feeder timeouts; relevant
  HTTP presentation, wire serialization and persistence regressions are tested
  separately. CI results are tracked on the exact PR head.

Independent review approved the final implementation after regression fixes,
including a second review of GB-based scheduler refusals and named hosts.
