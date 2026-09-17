# Lane F · Engine, FFI, release

Worktree branch `worktree-agent-a785dbfd79fc3d3b9`, cut from `feat/macos-native-app` @ `46af16fb`.
Reports read: `05-shell-engine-release.md` (everything not tagged `[settings]`) and
`04-queue-models-machines.md` L3.

## Findings

| id                                                     | status    | commit     | test                                                                                             |
| ------------------------------------------------------ | --------- | ---------- | ------------------------------------------------------------------------------------------------ |
| H1 · keyless engine + permissive CORS                   | fixed     | `946bfa34` | `EngineLaunchTests` (5); `mold-macos-ffi` `the_embedded_cors_origin_is_one_no_page_can_present`    |
| H2 · the FFI crate does not build                       | not-a-bug | —          | fixed before the wave by `34cfa484` (PLAN "Deviations"); `cargo check --features metal` is clean   |
| H3 · `/nix/store` linkage ships in the DMG              | fixed     | `bf8c8016` | `scripts/tests/linkage-fixup.sh` (run by `make test`)                                              |
| H4 · `make release` can ship an engine-less app         | fixed     | `bf8c8016` | `scripts/assert-embedded-engine.sh`; no automated test — see "Owed"                                |
| M1 · `ALIVE` stays true on a panic; nothing polls it    | fixed     | `857c05ec`, `f7fb7983` | `a_panicking_engine_thread_still_reports_dead`; `MoldEngine+Lifecycle.watch()`        |
| M2 · Start is enabled but inert after a failure         | fixed     | `f7fb7983` | `EngineLifecycleTests.startIsOfferedOnlyWhereItCanWork`                                            |
| M3 · `.running` published before the listener binds     | fixed     | `f7fb7983` | `EngineLifecycleTests.runningWaitsForTheEngineToAnswer`, `…aPortHeldBySomethingElseIsNamed…`       |
| M4 · `setenv` runs beside the app's threads             | fixed     | `f7fb7983` | none possible — see "Owed"                                                                        |
| M5 · a corrupt home pointer silently relocates          | fixed     | `f6e0740e` | `MoldHomePointerTests` (4); `EngineLaunchTests.aDamagedHomePointerRefuses…`                        |
| M6 · no interlock against a second engine on one home   | fixed     | `f7fb7983` | `EngineLifecycleTests.aSecondEngineOnOneHomeIsRefusedByName`                                       |
| M7 · quit allows 8 s of a 45 s drain; lease left behind | fixed     | `51cbe4c3`, `857c05ec` | `EngineLifecycleTests.theQuitBudgetIsTheServersOwn` (reads the Rust constant)         |
| M8 · the engine's SIGTERM handler strands the app       | fixed     | `f7fb7983` | none possible — see "Owed"                                                                        |
| M9 · `mold_engine_join` can block unbounded             | fixed     | `857c05ec` | `the_final_join_gives_up_rather_than_hanging_the_app`                                              |
| M10 · `macos-dev` kills the user's Tauri Mold           | fixed     | `a0737821` | none (devshell script)                                                                            |
| M14 · nested code signed with the app's entitlements    | fixed     | `a0737821` | none; the two exemptions are KEPT with the check each owes written into the file                  |
| M15 / 04-L3 · blanket `NSAllowsArbitraryLoads`          | kept      | `a0737821` | — the narrow form is a REGRESSION here; reasoning in `Info.plist`                                 |
| M16 · arm64/macOS 26 floor and no updater unstated      | fixed     | `73e9b528` | —                                                                                                 |
| M17 · `ENGINE_TARGET` hardcodes one Mac's volume        | fixed     | `bf8c8016` | —                                                                                                 |
| L1 · `apps/macos` is in no CI workflow                  | fixed     | `73e9b528` | `.github/workflows/macos-native.yml` — unrun, see "Owed"                                          |
| L2 · signing applies app entitlements to nested code    | fixed     | `a0737821` | `linkage-fixup.sh` covers the nested-Mach-O half                                                  |
| L3 · notarization result never inspected                | fixed     | `39a5b901` | —                                                                                                 |
| L4 · release version a duplicated literal `0.1.0`       | fixed     | `39a5b901` | `make -n dmg` names `Mold-0.1.0.dmg` from `project.yml`                                           |
| L8 · repo-wide ignore patterns for one app              | fixed     | `bf8c8016` | `git check-ignore -v` on all three                                                                |
| L9 · `NSBonjourServices` absent while `mdns` is on      | not-a-bug | —          | `NSBonjourServices` gates `NWBrowser`/`NetService`; `mdns-sd` is raw multicast UDP and no Swift source uses either API. Runtime check still owed |

## Adversarial review (`review/REVIEW-F.md`), second pass

| id                                                        | status  | commit     | test                                                                                  |
| --------------------------------------------------------- | ------- | ---------- | -------------------------------------------------------------------------------------- |
| F1 CRITICAL · the join strips a LIVE writer's lease        | fixed   | `37a00e6c` | `joining_never_reaches_into_the_gallery_authority` (source contract — see the note)      |
| F1b · a timed-out stop reported the engine stopped         | fixed   | `bb7627a2` | `EngineLifecycleTests.startIsOfferedOnlyWhereItCanWork` (`.stopping` can never start)    |
| F2 HIGH · quit hard-kills during `.starting` / `.stopping` | fixed   | `bb7627a2` | `EngineLifecycleTests.quittingWaitsForEveryStateWithAnEngineThreadInIt`                   |
| F11 · the panel's 45 s against a 50 s wait                 | fixed   | `bb7627a2` | `EngineLifecycleTests.theQuitBudgetIsTheServersOwn` (both new assertions)                 |
| F5 MED · the probe is an attempt COUNT, not a deadline     | fixed   | `7bfe6830` | `EngineLifecycleTests.theProbeGivesUpOnTheClockRatherThanAfterNAttempts`                  |
| F3 HIGH · the interlock probes a port nothing binds        | fixed   | `e62c96b1` | `a_held_lease_names_the_writer_and_is_never_taken_from_it` + 3; `the_lease_file_name_is_the_servers_own` |
| F4 HIGH · "This Mac" offers Pair a Phone…                  | fixed   | `3236270a` | `EngineLifecycleTests.thisMacsEngineIsNotSomethingAPhonePairsWith`                        |
| F6 MED · `pkill -f` matches nothing                        | fixed   | `df67a518` | none (devshell + Makefile); both now launch absolutely                                   |
| F7 MED · Release ships Debug's library-validation exemption | fixed  | `df67a518` | `xcodebuild -showBuildSettings` per configuration (Debug/Release files differ)            |
| F8 LOW · `SIG_IGN` inherited by `ffmpeg` children          | fixed   | `df67a518` | none possible — POSIX inheritance; the key is simply gone                                |
| F9 LOW · the one-shot migration runs at every launch       | documented | (README in the ledger commit) | confirmed idempotent: DB sentinel + `OnceLock` (`config_sync.rs:646-659`) |
| F10 LOW · Gatekeeper never asserted                        | fixed   | `df67a518` | `spctl --assess` after the staple, with no `|| true`                                     |
| CORS framing / ATS wording nits                            | fixed   | `df67a518` | — comments only                                                                          |

**F1's test is a source contract on purpose.** `release_gallery_writer_leases`
only touches leases this process REGISTERED, and nothing outside `mold-server`
can register one — so a behavioural test would have passed against the bug. The
test extracts `join`'s body and fails on any call into `gallery_authority`.

**F3 is a documented reversal of 05-M6's policy, decided from the code.** The
detection is now real (the gallery writer lease, read without writing, tested
against a held / released / absent lease) but the answer is a WARNING rather
than a refusal: `queue_journal.rs:203-213` designs for "two servers sharing one
`MOLD_HOME`", an unadopted queue is REPORTED as an orphan rather than silently
stranded, and CLAUDE.md states the writer lease is shared by design. Refusing
would break a setup mold supports. `EngineInterlock`'s doc carries all three
citations. This also removes F3's `EngineRelaunch` sub-finding: an advisory
cannot be tripped by the relaunch it would have blocked.

**F3 adds the sixth C function.** `mold_engine_home_writer_pid`, read-only,
`catch_unwind`-guarded, deliberately NOT `gallery_authority::storage_status`
(which takes the bookkeeping flock and so writes under
`.mold-batch-transactions`). The lease file's name is spelled twice — there and
in `gallery_authority`, which keeps it private — and pinned by a test that reads
the server's own source.

## Commits, in order

`946bfa34` H1 · `857c05ec` M1/M9 (+M7's lease) · `f6e0740e` M5 · `f7fb7983` M1/M2/M3/M4/M6/M8 ·
`51cbe4c3` M7 · `bf8c8016` H3/H4/M17/L8 · `a0737821` M14/L2/M10/M15 · `73e9b528` L1/M16 ·
`39a5b901` L3/L4 · `c899a5d5` ledger.

Second pass: `37a00e6c` F1 · `bb7627a2` F1b/F2/F11 · `7bfe6830` F5 · `e62c96b1` F3 ·
`3236270a` F4 · `df67a518` F6/F7/F8/F10 · this ledger (with F9's README sentence).

## Gates run

(Figures are after the second pass.)

- `make lint` — green (the three type-size advisories are pre-existing and unchanged; no file
  over 150 lines).
- `cd Packages/MoldClient && swift test` — 626 tests, 26 suites, green.
- app bundle `xcodebuild test` under the shared lock — 550 tests, 81 suites, green.
- `cargo check --features metal` in `rust/mold-macos-ffi` — clean.
- `cargo test --features metal` in `rust/mold-macos-ffi` — 10 tests, green.
- `xcodebuild -showBuildSettings` for both configurations — Debug takes
  `scripts/Mold.Debug.entitlements` and Release `scripts/Mold.entitlements`, and Debug still
  resolves `SWIFT_ACTIVE_COMPILATION_CONDITIONS = DEBUG`.
- `scripts/tests/candle-single-identity.sh` — PASS, still 5 candle packages on the one fork rev
  after the two new dependencies (`tracing`, and `http` as a dev-dependency).
- `scripts/tests/linkage-fixup.sh` — green, including the arm that must refuse.
- The `#if MOLD_EMBEDDED_ENGINE` arm was type-checked by temporarily defining the flag in
  `Engine.xcconfig` with no staticlib: compiles with no errors, fails only at the expected
  undefined symbols. Every remote-only test run leaves that arm uncompiled, so this is the
  only thing standing between the engine code and a broken `make engine`.

## Owed — exact checks this lane could NOT run

Nothing here is claimed as verified.

1. **H1 end to end.** `make engine && make build`, start the engine, then:
   `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:<port>/api/status` → `401`;
   with `-H "X-Api-Key: $(python3 -c 'import json;print(json.load(open("'"$HOME"'/Library/Application Support/io.utensils.mold.native/secrets.json"))["local-engine-api-key"])')`
   → `200`; and `curl -si -H 'Origin: http://evil.example' …/api/status | grep -i access-control-allow-origin`
   → the embedded sentence, never `*`.
2. **H3 on a real product.** `make engine && CONFIG=Release make build && otool -L
   build/Release/Mold.app/Contents/MacOS/Mold | grep /nix/store` → matches BEFORE
   `scripts/fix-macos-native-linkage.sh`, nothing after. The script is proven against
   synthesised load commands, not against a build of the engine.
3. **H4's assertion.** `make signed` on a Mac with `MOLD_SIGN_IDENTITY`; then again after
   `make engine-clean` and confirm it REFUSES naming `MOLD_EMBEDDED_ENGINE`.
4. **M4's residual race.** Not testable: `setenv` beside a concurrent `getenv` is a data race,
   not a behaviour. The window is narrowed to `@main`'s `init`; AppKit already has threads by
   then. Closing it means `run_server` taking the home and key as arguments.
5. **M7 / M8 at runtime.** With the engine running: `kill -TERM <pid>` quits the app cleanly
   (M8) and the "Finishing…" panel appears with Quit Now (M7); afterwards
   `ls "$MOLD_HOME"/**/.mold-gallery-writer.lease` finds nothing.
6. **M14 / F7's two remaining entitlements.** `disable-library-validation` is GONE from Release;
   `allow-jit` and `allow-unsigned-executable-memory` stay, each recording its own check in
   `scripts/Mold.entitlements`. Both need a SIGNED, hardened build and a local Metal render,
   which this lane has neither the identity nor the engine for. Run
   `allow-unsigned-executable-memory`'s check first: if a signed render works without it, it goes
   whatever happens to `allow-jit`. Debug's copy owes its own check in
   `scripts/Mold.Debug.entitlements`.
6b. **F6 at runtime.** `macos-dev` twice in a row: the second run must reap the first, and
   `ps -Ao args | grep Mold` must still show Mold Desktop if it was open.
6c. **F3's advisory on a real home.** Start `mold serve` on this Mac's `$MOLD_HOME`, then press
   Start in Settings ▸ This Mac: the pane must name that server's pid and the engine must still
   start. Stop it and press Start on a fresh launch: no advisory.
7. **CI.** `.github/workflows/macos-native.yml` has never run — this branch is never pushed to
   `main` and Actions cannot be exercised from here. `macos-26` is a real hosted image (two
   workflows in this repo already use it); `brew install xcodegen` on that image is the one
   step most likely to need adjusting.
8. **L9.** A signed bundle on a clean macOS 26 install must still advertise and discover over
   mDNS after the local-network prompt.

## Cross-lane edits

- `Packages/MoldClient/…/MoldHome+Pointer.swift` — NEW file in MoldClient (allowed by the task);
  `MoldHome.swift` itself is untouched, which is why the refusal is a separate query rather than
  folded into `resolve`.
- `Sources/Mold/MoldApp.swift` (lifecycle wiring only, as owned) and
  `Support/MoldAppDelegate.swift` (owned): six lines total.
- `.gitignore`, `flake.nix` (`macos-*` commands), `.github/workflows/macos-native.yml` — all named
  in the task.
- `Sources/Mold/Machines/MachinesPane+Sections.swift` — ONE line in the menus lane's file
  (`pairing(_:)` now asks `MoldEngine.isPairable`), F4. The reason and its test live in
  `Sources/Mold/Engine/`, so a cherry-pick conflict here resolves to that single call.
- **Not edited, and someone should**: 04-L3's other half asks for a sentence in the host editor
  when a key is set on a plain-`http` non-loopback address. `Shell/HostEditor.swift` is Lane E's.
- **Behaviour another lane should know about**: "This Mac" is now a KEYED host, so anything
  branching on `auth_required` or on a host having a key will start treating the local engine as
  keyed. The one such surface found — the Machines pane's pairing section — is fixed above;
  `HostStore+Editing`, `MachinesSettings.isManaged` and `HostEditor` were checked and are clean
  (the key is never written under `remote-api-key.<uuid>`).

## Judged wrong / not fixed as written

- **H2** was already fixed by the wave's own step 0 (`34cfa484`), which dropped the `version`
  requirements entirely rather than teaching `sync-release-pr.sh` a fourth root. Verified by a
  clean `cargo check`.
- **M15** asks for `NSAllowsLocalNetworking` plus a documented reason for anything broader.
  Adding it would BREAK the product requirement: on macOS 10.12+ Apple ignores
  `NSAllowsArbitraryLoads` whenever `NSAllowsLocalNetworking` is present, and a Tailscale host at
  `100.64/10` is not "local" to ATS. The blanket is kept and the whole reasoning, including the
  two real consequences, is written into `Info.plist`.
- **L9** is a non-issue as analysed (raw multicast UDP, not the Bonjour APIs `NSBonjourServices`
  gates), though the runtime check the report asks for is still owed.
- **F3's suggested fix** (an exclusive `flock` on a new `.mold-engine-owner` file under
  `$MOLD_HOME`, held for the engine's life) was not taken. It would be a second, mold-unaware
  lock beside the one mold already keeps for exactly this question, and it encodes the refusal
  policy the code says is wrong. The lease is read instead — and read-only.
- **F1's fallback suggestion** ("if the early release must stay, move it after
  `reply(toApplicationShouldTerminate:)`") was not taken either: the call is simply gone, which
  is the report's own first choice and leaves no path on which it can fire.
