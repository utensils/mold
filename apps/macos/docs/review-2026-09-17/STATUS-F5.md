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
| `macos-native-distribution.yml` + publish order, verbatim | done, then reworked | `947888a6`, `7c00b10c` | `native-workflows-parse.sh` |
| README updater + release sections | done | this commit | — |

## Round two — the adversarial review's findings

The security of the update path was found clean. Publishing was not.

| # | Finding | Status | Commit | Test |
| --- | --- | --- | --- | --- |
| F5#1 | HIGH — nightly could not build: `make dmg` named the DMG after `MARKETING_VERSION`, the workflow copied the nightly name | fixed | `7c00b10c` | `scripts/tests/release-names.sh` |
| F5#2 | HIGH — the stable feed used `releases/latest/`, an alias this app does not own | fixed, by the owner's design change | `7c00b10c` | `release-names.sh` (no native workflow may `gh release create`) |
| F5#3 | MED — the prune failed the job when there was nothing to prune | fixed | `7c00b10c` | `scripts/tests/prune-native-nightly.sh` |
| F5#4 | MED — the ledger's cross-lane instruction missed a fourth hunk | fixed below, and marked in the source | `64d56480` | — |
| F5#5 | MED-LOW — every nightly reported `0.1.0`; the appcast assertion was dead for nightly | fixed | `7c00b10c` | `release-names.sh`; the workflow now asserts all three with `&&` |
| F5#6 | LOW — Sparkle's own defaults keys escape `PreferencesReset`'s claim | classified in the comment | `64d56480` | the existing completeness test still passes |
| F5#7 | LOW — the bundle guard was decorative | fixed: it is the fourth `UpdaterActivation` condition | `64d56480` | `theBundleQuestionIsAskedWhereItCanHold`, `onlyAPlainReleaseLaunchMayReplaceItself` (16 combinations) |
| F5#8 | LOW — a channel change waited for the next scheduled check | fixed: `resetUpdateCycle()` | `64d56480` | — (it is one Sparkle call; nothing here can observe its scheduler) |
| F5#9 | NIT — the key gate proves shape, not identity | deferred, deliberately | — | — |

**F5#9 is deferred because the answer does not exist yet.** Pinning the key's
first bytes needs the owner's key to have been generated. It belongs in the
same sitting as the one-time key step, and the README's step is where to add
it.

### What the new publishing design is

The native app ships **alongside** the Tauri one, on the channels that already
exist, and never creates a release of its own:

- **Stable** — `release.yml`'s new `build-macos-native-dmg` calls the reusable
  builder beside `build-desktop-dmg`; the DMG and `mold-native-appcast.xml`
  join the same `softprops/action-gh-release` `files:` list. So
  `releases/latest/download/mold-native-appcast.xml` is correct *because*
  release-plz owns that pointer.
- **Nightly** — `macos-native.yml`'s `publish-macos-native-nightly` rides
  desktop.yml's rolling `latest` prerelease, in the SAME concurrency group
  (`desktop-nightly-publication`) because both clobber assets there.
- `macos-native-publish.yml`, `check-publish-head.sh` and the
  `macos-native-v*` tag namespace are **deleted**.

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
3. **The publishing jobs' first real run.** None of them has ever executed,
   and none CAN from this branch: `release.yml` runs on `v*` tags and
   `macos-native.yml`'s nightly jobs on a push to `main`, and this branch is
   never merged. They are correct by READING — mirrored line for line against
   the desktop jobs they sit beside, cited in place — and by
   `scripts/tests/native-workflows-parse.sh`, which parses all four workflows,
   asserts the new jobs exist by name AND that the desktop ones are untouched,
   and runs `bash -n` over every inline `run:` block in the two this app owns.
   `scripts/tests/release-names.sh` ties the Makefile's artifact names to the
   workflow's expectations, and `prune-native-nightly.sh` exercises the prune's
   selection without GitHub.

   What the first run will exercise that nothing here can:
   - **The engine step.** `make signed` builds `rust/mold-macos-ffi` with
     cargo on `macos-26`. No CI job has ever done that — `macos-native.yml`'s
     `check` is deliberately remote-only — so the cargo build, the fdk-aac C
     dependency and the 26.0 deployment target are all first-time on a runner.
   - **The Developer ID identity string** matching the certificate imported
     into the ephemeral keychain.
   - **`generate_appcast` reading a notarized DMG**, and the three assertions
     over the feed it writes.
   - **Nightly:** whether the rolling `latest` prerelease already exists when
     the native job runs (it bows out with a notice rather than creating one),
     and the anonymous propagation waits.
   - **Stable:** that the artifact lands as `artifacts/Mold-native-<v>.dmg` and
     `artifacts/updates/mold-native-appcast.xml` under `release-version`'s
     `merge-multiple: true` download, which is what the `files:` entries name.

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
  (commit `a7af5a6d`). **The instruction below replaces an earlier one in this
  ledger that said `MoldApp.swift` needed no edit at all. It was wrong**, and
  following it would have compiled, rendered, and left `ActivityStore` never
  started at launch (review F5#4). The upscale lane has FOUR hunks in the old
  `MoldApp.swift`, and the fourth one does not live in `AppStores`.

  For the integrator, taking the upscale lane's work onto this shape:

  1. **Discard its `MoldApp.swift` diff wholesale.** All four hunks will
     conflict, because `a7af5a6d` MOVED the region and cherry-pick does not
     follow moves.
  2. `AppStores.swift` — two `let`s in the property list; in `init()`, at the
     `// NEW STORES GO HERE` marker, `UpscaleStore(hosts:models:library:)` and
     `ActivityStore(hosts:)`. Both dependencies are already above it
     (`library`, then `models`).
  3. `AppStores+Environment.swift` — two `.environment()` at its marker.
  4. **`MoldApp.swift`, the line marked `THE LAUNCH START`:**
     `if NSApp.isActive { stores.heartbeat.start(); stores.activity.start() }`.
     That line is the one hunk of the old composition root that did not move,
     and `applicationDidBecomeActive` has already fired by the time it runs —
     a store that watches the activation notifications itself still needs its
     first start from here.

  Generate-controls lane: nothing in F5 touches `MoldAppDelegate`'s quit path,
  so a `flush()` added to `applicationShouldTerminate` will not conflict.

- `.github/workflows/release.yml` — three lines and a release-note sentence:
  the `build-macos-native-dmg` job, its entry in `release-version`'s `needs`,
  and `artifacts/updates/mold-native-appcast.xml` in the `files:` list. The
  desktop and CLI jobs are untouched, and `native-workflows-parse.sh` asserts
  that. Commit `7c00b10c`, at the owner's explicit direction.

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
- App bundle, under the shared lane lock — 567 tests in 85 suites, passed
  (558 before; this lane adds 9).
- `scripts/tests/{linkage-fixup,sparkle-key-gate,sparkle-signing-order,appcast-recipe,release-names,prune-native-nightly,native-workflows-parse}.sh`
  — all pass, all wired into `make test`, and each has a negative arm that
  fails on the bug it pins.
- `scripts/tests/ci-routing-contract.sh` — still PASS with the two new
  workflows.
- Both new workflow files parse as YAML.
- `scripts/assert-sparkle-key.sh` and `scripts/fix-macos-native-linkage.sh`
  were both run against the REAL enlarged `Mold.app`: the key gate refuses the
  placeholder, and the linkage check passes over 8 Mach-O files (Sparkle's
  framework, `Autoupdate`, `Updater.app` and the two XPC services included).
- No helper sub-agents were spawned.
