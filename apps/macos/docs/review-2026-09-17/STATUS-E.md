# Lane E · Secrets + Settings

Worktree branch `worktree-agent-a8466b6972565508c`, based on `feat/macos-native-app` @ `bc96d415`.
Report read: `05-shell-engine-release.md` (the `[settings]` items, plus every LOW about Settings,
Shell preferences or host editing).

## Findings

| id | status | commit | test |
|---|---|---|---|
| decision · keys out of the Keychain | done | `c944d82a` | `SecretStoreTests` (11, port of `secrets.rs`'s own suite incl. `secretsFileIsOwnerOnly`) |
| decision · one-time Keychain migration | done | `f11a4779` | `HostSecretsTests.everyKeychainKeyMovesToTheFileAndTheItemGoes`, `.theMoveHappensOnce`, `.oneUnreadableItemLosesNeitherTheOthersNorItself` |
| H5 · a list write mirrors an absent key as a delete | fixed | `f11a4779` | `HostSecretsTests.savingTheMachineListNeverTouchesAStoredKey`, `.aKeyIsClearedOnlyWhenSomebodyAsks` |
| H6 · removing a machine is unconfirmed and destroys its key | fixed | `c1399dfa` | `MachinesSettingsTests.removingAMachineAsksFirstAndOnlyThenRemovesIt`, `.theQuestionNamesTheKeyItDestroys` |
| M11 · curated panes edit env-locked rows | fixed | `32b732cc` | `SettingRowTests.anEnvOwnedRowIsReadOnlyAndSaysWhy`, `.anEnvOwnedRowWithNoNamedVariableStillExplainsItself` |
| M12 · Return on an empty secret clears the credential | fixed | `3e7244f6` | `AdvancedTableEditorTests.anEmptySecretFieldNeverClearsTheStoredCredential` |
| M13 · Reset does not keep its own promise | fixed | `0322a6ea` | `PreferencesResetTests` (both) |
| L5 · an emptied machine list is re-seeded | fixed | `e63828c8` | `HostPersistenceTests` (3) |
| L6 · `nan`/`inf` reports the whole machine as failed | fixed | `6229234c` | `AdvancedTableEditorTests.aNonFiniteNumberRevertsRatherThanFailingTheMachine` |
| UAT hooks ship in Release | fixed | `67184017` | `NativeUATTests` (3) |
| context menus (explicit goal) | done | `05ed1032` | `RowActionTests` (7) |
| README key sentences | done | `98cdd1e6` | — |
| — file-size floor after the above | done | `3fd8a1b4` | — |

Not this lane, and marked so for Lane F: **H1**, **H2**, **H3**, **H4**, **M1–M10**, **M14–M17**,
**L1** (CI), **L2**, **L3**, **L4**, **L8**, **L9** are engine / FFI / release / CI. **L7**
(`keyboardShortcut` on a `Menu`, `Shell/MoldCommands.swift`) is a shell menu item nobody's lane
list names; it is untouched here.

## What the fixes are

- **`SecretStore`** (`Packages/MoldClient/Sources/MoldClient/SecretStore*.swift`) is a port of
  `desktop/src-tauri/src/secrets.rs`: flat `{"name": "value"}` under
  `~/Library/Application Support/io.utensils.mold.native/secrets.json`, `0600` set on the temp file
  BEFORE an atomic `rename(2)` (`replaceItemAt` would carry the OLD file's mode onto the new one),
  an unparseable file parked once as `secrets.json.corrupt`, one `Mutex` around the whole
  read-modify-write, names limited to `remote-api-key.<host uuid>` and `local-engine-api-key`.
  Errors are thrown. `localEngineAPIKey(environment:)` is there for **Lane F** with
  `local_server_api_key`'s precedence: non-empty `MOLD_API_KEY` → stored → a fresh UUID stored
  before it is returned.
- **`LegacyKeychain`** replaces `Keychain.swift` and is a migration READER only. Its `Source` is
  injected, so the rules are tested without a real keychain. A read that is `unreadable` (as opposed
  to `absent`) leaves the done-flag off, so the next launch retries rather than abandoning that key.
- **M13's pin** reads the app's own sources for every string handed to `AppStorageSuite.defaults`
  and fails on one that neither `PreferencesReset.keys` nor `.kept` names — including one spelled in
  a way it cannot resolve. That is what makes a fifth omission impossible rather than unlikely.
- **The UAT gate** is `Support/NativeUAT.swift`: one case per hook, `#if DEBUG` in one place.
  `NativeUATTests` fails if a `MOLD_NATIVE_` literal appears anywhere else in `Sources/Mold` outside
  a comment. Measured on the built binaries: the Debug dylib carries all eight names, the Release
  executable carries **none**.
- **Context menus**: `RowAction` + `RowActionMenu` (destructive last, behind a divider, whatever
  order declared). Machines rows get Edit… / Check Now / Copy Address / Set as Default / Remove…
  (This Mac's engine keeps every item and disables the three it cannot do); Advanced and curated
  config rows get Copy Key / Copy Value / Copy `<env var>` / Reset to Default — a secret's value is
  never copyable; an Accounts provider gets Clear `<name>` Token on exactly `offersClear`.

## Cross-lane

- `Sources/Mold/Shell/RootView.swift` (2 lines) and `Sources/Mold/Support/HostStore.swift` (1 line)
  now ask `NativeUAT` instead of `ProcessInfo` directly. No behaviour change in Debug. (`67184017`)
- `Sources/Mold/Shell/ProviderSection.swift` gains a `.contextMenu` and a 4-line `menu` property.
  Nobody's lane list names this file; it is Accounts' own row. (`05ed1032`)
- `Packages/MoldClient/Sources/MoldClient/StoredHost.swift` — two doc-comment sentences that said
  the key travels through the Keychain. Lane A owns the file; this is comment text only. (`f11a4779`)
  Its test file's own such sentence, the same way, in this lane's last commit.

## Requests for Lane F (do not act on these here)

1. **`make uat` should empty the throwaway secrets directory** the way it already empties the
   throwaway prefs domain — one line beside the `defaults delete`:
   `rm -rf "$HOME/Library/Application Support/io.utensils.mold.native.fresh"`.
2. **`MOLD_EMBEDDED_ENGINE` is shadowed in Debug.** `make gen` writes
   `SWIFT_ACTIVE_COMPILATION_CONDITIONS = DEBUG` into the PROJECT's Debug configuration
   (`project.pbxproj`, one occurrence), and a project-level build setting outranks the project's
   xcconfig — so `Engine.xcconfig`'s `$(inherited) MOLD_EMBEDDED_ENGINE` never reaches a Debug
   build. Release has no project-level value and is unaffected. Verified by reading the generated
   pbxproj, not by building an engine. This compounds 05-H4.
3. `localEngineAPIKey()` is on `SecretStore.shared`; a Release build can never be pointed at the
   throwaway directory (its `MOLD_NATIVE_FRESH` read is `#if DEBUG`, spelled package-side because
   MoldClient cannot import the app).

## Nothing judged wrong

Every finding this lane took reproduced exactly as the report described it.

## Verification

`make lint` green (three pre-existing large-type advisories only, none of them this lane's).
`swift test` in `Packages/MoldClient`: **425** passed. Full app bundle `xcodebuild test`: **406** in
62 suites passed. A Release build was made once to prove the `#else` arms compile and to count the
hook strings in each binary; its output was deleted.
