# Lane F5 · Sparkle 2 updater

Worktree `.claude/worktrees/agent-a3a3506bad474d934`, branch
`worktree-agent-a3a3506bad474d934`, cut from `a5172d65`.

This lane builds a feature rather than fixing findings, so the ledger is one
row per piece of PLAN "F5".

| Piece | Status | Commit | Test |
| --- | --- | --- | --- |
| Sparkle 2 as a pinned SPM dependency, `SUFeedURL` / `SUPublicEDKey` / `SUScheduledCheckInterval` | done | `c039341f` | `theBundlesDefaultFeedIsTheStableOne` |
| Release fails closed on a placeholder key or an off-allowlist feed | done | `81492994` | `scripts/tests/sparkle-key-gate.sh` |
| Sparkle's helpers signed innermost-first, Downloader keeps its entitlements | done | `81492994` | `scripts/tests/sparkle-signing-order.sh` |
| Channel → feed, allowlist, unknown value falls back to stable alone | done | `db9feeca` | `theFeedsAreAFixedHTTPSAllowlist`, `theFeedsAreTheOnesTheWorkflowPublishes`, `anUnrecognisedStoredChannelIsStableAlone`, `theDelegateRereadsTheChannelOnEveryCheck` |
| No updater in Debug / under `MOLD_NATIVE_FRESH` / in the test host | done | `db9feeca` | `onlyAPlainReleaseLaunchMayReplaceItself`, `thisProcessHasNoUpdater` |
| "Check for Updates…" under About Mold, declared once; Settings ▸ General ▸ Updates | done | `db9feeca` | `checkForUpdatesIsDeclaredOnce` |
| Composition root split so three lanes fit under the 150-line rule | done | `a7af5a6d` | the whole app suite (566) stays green |
| Reset leaves the update channel alone (cross-lane) | done | `4aeb4d33` | `everyPersistedPreferenceIsEitherResetOrDeliberatelyKept` |
| `make appcast`, `CFBundleVersion` from the commit count | done | `d26e0fd2` | `scripts/tests/appcast-recipe.sh` |
| `macos-native-distribution.yml` + publish order, verbatim | done | `947888a6` | — (owed: its first run) |
| README updater + release sections | done | this commit | — |

## What is OWED, exactly

Three things this lane cannot do, and nobody should read the green gates as
covering them.

1. **The owner's key step.** `generate_keys` must be run once on James's Mac;
   the PRIVATE half goes into the `MOLD_NATIVE_SPARKLE_KEY` repository secret
   and the PUBLIC half replaces `MOLD_SPARKLE_PUBLIC_ED_KEY` in
   `apps/macos/project.yml`. Until then **every Release build fails** at
   `scripts/assert-sparkle-key.sh` — deliberately: a Release that cannot
   verify an update must not exist. Debug is unaffected and `make lint`,
   `make test` and `make build` are all green without it. Exact commands are
   in README ▸ Updates ▸ "The one-time owner step".
2. **A signed build updating itself from a test appcast.** Nothing here has
   ever run an actual update. What is proved is the feed choice, the gate, the
   menu, the signing ORDER and the key check; what is not is Sparkle
   installing over a real notarized bundle. The first end-to-end check should
   be: build two versions with different commit counts, publish the older,
   point the newer's feed at a local file, and watch the install.
3. **The publish workflow's first run.** `macos-native-distribution.yml` and
   `macos-native-publish.yml` have never executed. Both YAML files parse and
   `scripts/tests/ci-routing-contract.sh` still passes, but the RUNTIME
   unknowns are real and named here so the first run is read as a first run:
   the engine step (`make signed` builds `rust/mold-macos-ffi` with cargo on
   `macos-26`, which no CI job has ever done — `macos-native.yml` is
   deliberately remote-only), the Developer ID identity string matching the
   imported certificate, and `generate_appcast` reading a notarized DMG.

## Decisions worth the reviewer's attention

- **Two FEEDS, not Sparkle channels.** Sparkle can tag items with
  `sparkle:channel` inside one feed, and that is right when a beta is a subset
  of the same stream. Nightly is not: it must not be reachable from the stable
  pointer at all, which is also how the Tauri app publishes it. So
  `--channel` is deliberately NOT passed to `generate_appcast`, and the
  appcast-recipe test asserts that — a tagged item is one a default-channel
  updater silently ignores.
- **The feed is chosen by the delegate, not `SUFeedURL`.** `setFeedURL` is
  deprecated in Sparkle 2 because a feed written into user defaults outlives
  the app that wrote it. `feedURLString(for:)` is asked at every check, so the
  picker takes effect immediately and nothing persists on Sparkle's side. The
  test drives the SECOND and THIRD change, not just the first.
- **Absent, not disabled.** In Debug / UAT / tests there is no
  `SPUStandardUpdaterController` at all, so the menu item and the Settings
  group do not exist. A greyed-out "Check for Updates…" in a dev build would
  be a promise about a feed that build must never read.
- **`UpdaterActivation` is a pure function of three booleans**, exercised over
  all eight combinations, precisely because a test wrapped in `#if DEBUG` runs
  nothing in the other configuration. `BuildFlagsTests` is the precedent.
- **Not sandboxed, confirmed from Sparkle's own documentation.** The
  `SUEnableInstallerLauncherService` key and the
  `com.apple.security.temporary-exception.mach-lookup.global-name` pair are on
  Sparkle's *sandboxing* page and exist so a sandboxed app can reach its own
  installer. `ENABLE_APP_SANDBOX` is `NO`, so neither is added. The one thing
  from that page that DOES apply is the signing order, and the exception it
  makes for `Downloader.xpc`'s own entitlements.
- **`CFBundleVersion` is `git rev-list --count HEAD`.** Sparkle compares it
  across BOTH channels. A consequence worth stating: a nightly is always
  "newer" than the last release, so moving back to Stable waits for a release
  cut after that commit. The README says so and the Settings footer says so.

## Cross-lane edits

- `Sources/Mold/Settings/PreferencesReset.swift` — one entry
  (`"updateChannel"`) in `kept`, plus the sentence that explains it. Commit
  `4aeb4d33`, kept separate so the integrator can sequence it.
  `PreferencesResetTests` reads the sources and fails on any key written to
  the suite that neither list names, so this was not optional.
- `Sources/Mold/MoldApp.swift` was split at the coordinator's instruction
  (commit `a7af5a6d`). **The upscale lane's two stores now go in
  `AppStores.swift`**: a `let` in the property list, a line in `init()` at the
  marker `// NEW STORES GO HERE`, and — if a pane needs to read them — one
  `.environment()` at the marker in `AppStores+Environment.swift`.
  `MoldApp.swift` is 71 lines and needs no edit from that lane at all.

## Findings judged wrong

None; this lane had no review findings to verify.

## The one real obstacle, recorded

`xcodebuild -resolvePackageDependencies` HANGS in an agent shell on Sparkle's
BINARY artifact. Sparkle's `Package.swift` is a `binaryTarget` whose
XCFramework is a GitHub release asset; the git fetch and checkout complete,
then SwiftPM's own downloader never returns (killed at 81 minutes, and again
at a 5-minute hard timeout). `curl` of the same URL takes 0.36 s and the bytes
match `Package.swift`'s checksum exactly, and `git ls-remote` answers with no
prompt, so it is not the network, the URL, the version, credentials or a lock.
Priming SwiftPM's artifact cache with that download
(`~/Library/Caches/org.swift.swiftpm/artifacts/https___github_com_sparkle_project_Sparkle_releases_download_2_10_0_Sparkle_for_Swift_Package_Manager_zip`)
made resolution succeed immediately, and everything since has built and tested
normally. Nothing was vendored. CI runners have an ordinary network and should
not hit this; if a fresh clone does, that one `curl` + `cp` is the workaround.

## Gates

- `make lint` — green; `MoldApp.swift` no longer appears in `lint-size` at all.
- `cd Packages/MoldClient && swift test` — 637 tests in 28 suites, passed.
- App bundle, under the shared lane lock — 566 tests in 85 suites, passed
  (558 before; this lane adds 8).
- `scripts/tests/{linkage-fixup,sparkle-key-gate,sparkle-signing-order,appcast-recipe}.sh`
  — all pass, and each has a negative arm that fails on the bug it pins.
- `scripts/tests/ci-routing-contract.sh` — still PASS with the two new
  workflows.
- Both new workflow files parse as YAML.
- `scripts/assert-sparkle-key.sh` and `scripts/fix-macos-native-linkage.sh`
  were both run against the REAL enlarged `Mold.app`: the key gate refuses the
  placeholder, and the linkage check passes over 8 Mach-O files (Sparkle's
  framework, `Autoupdate`, `Updater.app` and the two XPC services included).
- No helper sub-agents were spawned.
