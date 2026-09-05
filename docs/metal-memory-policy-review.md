# Metal memory policy: independent review

Reviewer: Claude Fable 5.1 (`claude-fable-5-1`), via local Claude Code CLI,
2026-09-05 UTC. Scope: original plan, base `c6b925473`, no implementation.

Fable found the total/incremental model sound but requested corrections before
implementation. Findings and dispositions:

| Finding | Disposition |
| --- | --- |
| Scheduler uses resource sampler/registry and a separate terminal-classification ceiling | Accepted; explicitly route both through policy |
| Registry sysinfo and inference Mach available RAM differ | Accepted; use Mach free + inactive for policy in both |
| In-engine free accessor is the critical choke point | Accepted; clamp it, not only discovery |
| Lazy discovery plus unknown-budget refusal can deadlock startup | Accepted eager device probe; retain explicit refusal for actual supported-probe failure rather than RAM fallback |
| Metal allocated count includes retained pool and should not activate legacy CUDA attribution | Accepted; separate additive policy field, sweep before post-drop sample |
| Per-site total/incremental choices need a concrete table | Accepted in revised plan |
| Stable H3 numbers and tests change | Accepted; injected samples and revised expectations |
| Privileged group must return before Config/DB/log initialization | Accepted, including status |
| Sysctl width, safety floor and registered daemon reset must be specified | Accepted; u32 MiB, existing floor, fixed-label bootout |
| Read-only status, lenient client parsing and manual docs obligations | Accepted |
| Split accounting and privileged administration into two PRs | Declined to honor user's explicit one-branch, one-PR requirement; isolate milestones in commits |

No kernel setting or inference job was changed/run for the review. Raw reviewer
output is retained locally in ignored `tmp/metal-memory-review/plan-fable.txt`.
