# Metal memory policy: independent review

Reviewer: Claude Fable 5.1 (`claude-fable-5-1`), via local Claude Code CLI,
2026-09-05 UTC. Scope: original plan, base `c6b925473`, no implementation.

Fable found the total/incremental model sound but requested corrections before
implementation. Findings and dispositions:

| Finding                                                                                      | Disposition                                                                                                      |
| -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Scheduler uses resource sampler/registry and a separate terminal-classification ceiling      | Accepted; explicitly route both through policy                                                                   |
| Registry sysinfo and inference Mach available RAM differ                                     | Accepted; use Mach free + inactive for policy in both                                                            |
| In-engine free accessor is the critical choke point                                          | Accepted; clamp it, not only discovery                                                                           |
| Lazy discovery plus unknown-budget refusal can deadlock startup                              | Accepted eager device probe; retain explicit refusal for actual supported-probe failure rather than RAM fallback |
| Metal allocated count includes retained pool and should not activate legacy CUDA attribution | Accepted; separate additive policy field, sweep before post-drop sample                                          |
| Per-site total/incremental choices need a concrete table                                     | Accepted in revised plan                                                                                         |
| Stable H3 numbers and tests change                                                           | Accepted; injected samples and revised expectations                                                              |
| Privileged group must return before Config/DB/log initialization                             | Accepted, including status                                                                                       |
| Sysctl width, safety floor and registered daemon reset must be specified                     | Accepted; u32 MiB, existing floor, fixed-label bootout                                                           |
| Read-only status, lenient client parsing and manual docs obligations                         | Accepted                                                                                                         |
| Split accounting and privileged administration into two PRs                                  | Declined to honor user's explicit one-branch, one-PR requirement; isolate milestones in commits                  |

No kernel setting or inference job was changed/run for the review. Raw reviewer
output is retained locally in ignored `tmp/metal-memory-review/plan-fable.txt`.

## Intermediate administration review

Claude Fable reviewed the local administration implementation on 2026-09-04
(`claude --model fable --effort high`, read-only, no MCP servers). Accepted:

- Extract kernel/boot-policy orchestration behind injectable traits; cover
  bootout failure, kernel failure after bootout, persistence rollback and reset.
- A foreign boot-policy file is a warning for live-only changes, while persistent
  mutations still refuse it.
- Distinguish unsupported (false) from failed reads (null) in top-level status.
- Lock the trusted directory descriptor without leaving a lock artifact.
- Freeze the exact v1 plist in a golden fixture so future changes must preserve
  or explicitly migrate existing owned files.
- Gate the macOS directory constant and improve rollback/recovery messages.

Two proposals were intentionally not adopted. A mismatched kernel readback can
also mean another administrator changed the setting; blindly restoring the old
value would overwrite that action. Report requested/observed/previous values
and require inspection instead. Conditional rollback is reserved for a value
still equal to Mold's verified write. Native writes stay in the explicit CLI
administration module; moving a mutation helper into read-only inference
telemetry would weaken the architectural boundary without improving the ABI.
The conservative exact not-loaded launchctl classification is retained; other
errors must never be mistaken for an absent service. A stale unverified service
gets concrete inspection/recovery commands rather than automatic deletion.

## Full implementation review and corrections

Claude Fable 5.1 reviewed `c6b925473..bdacbb0f0`, also inspecting the follow-up
`86ff8c7dc`. Two high-severity findings were accepted:

- Metal's pre-existing `vram_in_use_bytes` stub returned zero, making every
  resident cache credit zero. Native allocated-byte deltas now populate that
  ledger. Load baselines first release unused Candle pool buffers so recycled
  allocations cannot erase the measured footprint. Injected regression tests
  fail on the old stub and pass with the real observation; warm admission
  accepts the recorded footprint once and still refuses a subsequently reduced
  kernel policy.
- Candle's constructor indexes its device list with `swap_remove`, which panics
  for a missing ordinal. Validate the native device count before invoking that
  constructor, returning an ordinary error without poisoning the shared cache.
  An injected constructor proves missing ordinals never enter Candle.

Also accepted: preserve simultaneous kernel/device/host probe errors; mark the
pre-telemetry Metal budget unavailable; preserve CUDA's original no-worker
capacity fallback; and read the latest capacity directly for terminal
classification instead of building an entire canonical device snapshot.
Stable and streaming helper tests now use injected policy observations. The
native read-only qualification test is explicitly ignored in automated suites;
no native GPU inference, kernel mutation or boot-policy installation was run.

## Follow-up verdict

Claude Fable 5.1 reviewed `86ff8c7dc..0b93dcbc1` and confirmed all six findings
closed in substance, with no blocking issue. Two low-severity observations:

- Accepted a second unused-buffer sweep before the post-load sample, keeping
  load-time dtype-conversion temporaries out of resident cache credit.
- The native server guard remains tested through its pure observation adapter.
  Broader scheduler tests now supply actual Metal policy fixtures and exercise
  first-observation readiness, transient external pressure, recovery/dispatch
  and avoiding a duplicate host-memory reservation. A core regression pins
  conservative recommendation handling after an increase or reset. These do
  not claim end-to-end privileged or model-workload qualification.

The full scheduler run caught two old fixtures that supplied no Metal policy;
those fixtures are now updated to the new required authority rather than
weakening the fail-closed production behavior. A redundant ordinal cast found
by clippy was also removed.

## Final closure

Claude Fable 5.1 reviewed `0b93dcbc1..32152e7bf` and returned **No findings**.
The reviewer rechecked both the pool sweep and constructor against this
branch's actual Candle pin `744ae3b83cfac18db28107a353c449cc9b80d4ec`, confirming
the preceding closures at that revision. The final validation refresh passed
CPU workspace and Metal all-targets clippy, core policy tests and injected
inference tests. A stale test-comment description was corrected afterward;
there are no further implementation changes or outstanding review findings.
