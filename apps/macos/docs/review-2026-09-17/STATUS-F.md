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

## Commits, in order

`946bfa34` H1 · `857c05ec` M1/M9 (+M7's lease) · `f6e0740e` M5 · `f7fb7983` M1/M2/M3/M4/M6/M8 ·
`51cbe4c3` M7 · `bf8c8016` H3/H4/M17/L8 · `a0737821` M14/L2/M10/M15 · `73e9b528` L1/M16 ·
`39a5b901` L3/L4 · this ledger.

## Gates run

- `make lint` — green (the three type-size advisories are pre-existing and unchanged).
- `cd Packages/MoldClient && swift test` — 626 tests, 26 suites, green.
- app bundle `xcodebuild test` under the shared lock — 547 tests, 81 suites, green.
- `cargo check --features metal` in `rust/mold-macos-ffi` — clean.
- `cargo test --features metal` in `rust/mold-macos-ffi` — 4 tests, green.
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
6. **M14's two entitlements.** Each records its own check in `scripts/Mold.entitlements`. Both
   need a SIGNED, hardened build and a local Metal render, which this lane has neither the
   identity nor the engine for. Neither was removed on reasoning alone.
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
- **Not edited, and someone should**: 04-L3's other half asks for a sentence in the host editor
  when a key is set on a plain-`http` non-loopback address. `Shell/HostEditor.swift` is Lane E's.
- **Behaviour another lane should know about**: "This Mac" is now a KEYED host. Anything that
  branches on `auth_required` or on a host having a key — the Machines pane's pairing section,
  for one — will start treating the local engine as keyed. `LocalEngineSettings` says why pairing
  it is still pointless (a phone cannot reach loopback here), but the Machines pane's own gate is
  not this lane's file.

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
