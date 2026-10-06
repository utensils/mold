# Temporary build-first CI

Owner-requested on October 6, 2026, pending better runners. PRs provide basic
formatting, workflow syntax, dependency identity, locked Rust compilation and
relevant frontend/native app compilation. This temporarily reduces regression
coverage: optional GPU feature coverage comes from shipping builds, and full
unit, accessibility, instrumentation and coverage gates are deferred.

Nightlies still build the artifacts they publish. Signing, notarization,
TestFlight processing validation, packaged artifact checks and publication remain
active. Publishers depend on their artifact build rather than unrelated test jobs.
No test implementation was deleted; local test commands remain available.

Observed timings on release main `11462906`:

| Lane | Runtime |
| --- | --- |
| Rust union lint/tests | 26m23s |
| Rust default lint/tests | 16m52s |
| Coverage | 13m43s |
| Desktop native lint/tests | 15m19s |
| Windows installer including redundant lint/tests | 66m22s |
| Linux AppImage including redundant lint/tests | Over 2h |
| Signed desktop macOS distribution | 88m02s, including notarization |
| Native iOS hosted accessibility legs | 60–90m plus runner queue |

These are observations, not promised speedups. Actual release compilation and
Apple service processing can still take substantial time. The Docker candidate
matrix has a 120-minute per-target budget and ran for over two hours; it duplicated
the tagged publication matrix. Automatic candidate triggers and release-plz's
three-hour precursor wait are paused. Manual Docker validation remains available;
tagged Docker publication still emits verified digest/provenance metadata.
Weekly/manual MSRV and tagged Nix cache publication remain independent.

## Restore after runner improvements

1. Restore the blocks marked `PAUSED 2026-10-06` in CI, Desktop, native macOS,
   native iOS, Tauri iOS, Android, Docker validation, MSRV and release-plz workflows.
   Remove the temporary basic `rust`, `desktop-rust` and native iOS `check` jobs
   before uncommenting their original definitions to avoid duplicate job keys.
2. Restore the original nightly `needs` lists and Android emulator architecture.
   Restore Docker automatic triggers and the release-plz precursor wait together.
3. Remove the temporary mode branch at the top of `ci-routing-contract.sh`, then
   run its preserved comprehensive assertions. Restore native full-suite docs.
4. Run actionlint and the relevant full contracts before merging the restoration.
   Benchmark runner/queue times before making comprehensive tests blocking again.

The release monitor in Codex is paused independently of GitHub Actions. This
change does not restart it.
