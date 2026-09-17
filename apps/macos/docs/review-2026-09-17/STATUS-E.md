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
| — file-size floor after the above | done | `3fd8a1b4`, `b8cd2ba7` | — |

### Adversarial review (`review/REVIEW-E.md`), second pass

| id | status | commit | test |
|---|---|---|---|
| E1 HIGH · a retried migration overwrites a NEWER key | fixed | `f352e654` | `HostSecretsTests.aRetriedMigrationKeepsTheKeyTheUserJustTyped`, `.anItemGoesOnlyAfterTheFileProvesItHasTheKey` |
| E2 MED-HIGH · an UNREADABLE secrets.json is clobbered | fixed | `c62ebe7e` | `SecretStoreTests.anUnreadableFileIsNeverWrittenOver` |
| E3 MED · a failed write leaves the document world-readable | fixed | `a853ce5d` | `SecretStoreTests.aTemporaryFileIsOwnerOnlyFromItsFirstByte`, `.aFailedWriteLeavesNoPlaintextBehind` |
| E4 · `localEngineAPIKey` has no caller | intentional | — | see below |
| E5 · the cross-lane list was incomplete | fixed | this file | — |
| E6 · a package test cannot hold in Release | fixed | `0a24db66` | `SecretStoreTests.aFreshRunUsesAThrowawayDirectory` (both configurations run) |
| info · two live writers clobber each other | fixed | `db95f051` | `SecretStoreTests.aSecondWriterDoesNotClobberTheFirst` |
| info · no fsync before the Keychain delete | fixed | `a853ce5d` | — (covered by `writeOwnerOnly`) |
| info · `LegacyKeychain` has no `MOLD_NATIVE_FRESH` gate | fixed | `f300191d` | `HostSecretsTests.aFreshRunNeverTouchesTheRealKeychain` |
| info · a menu that opens EMPTY | fixed | this lane's last commit | `RowActionTests.aRowWithNoApplicableActionCarriesNoMenu`, `.everyOtherSurfaceAlwaysHasSomethingToOffer` |

**E1** is the one that could have cost somebody a key: the file now WINS over the Keychain item,
because it is newer by construction — nothing but a deliberate save puts a value there — and the
item is deleted only once `persistedValue` (a fresh read from disk, never the cache that would
answer with what we meant to write) says the file holds it. **E2** is a refusal rather than a park,
because whatever stopped the read will stop the rename: reads AND writes throw, so the blank
key field an unreadable store produces can no longer become a delete. **E3** creates the temp file
`0600` at creation via `FileManager.createFile(atPath:contents:attributes:)`, inside the `do` that
cleans up, and `fsync`s it before the rename.

**E4 is deliberate and stays**: `SecretStore.localEngineAPIKey` and the `local-engine-api-key`
allowlist entry have no caller in this lane. Lane F (Wave 2, 05-H1) is the caller, and it needs the
API to exist before it can land. If Lane F slips, this moves with it.

**The empty menu was not cosmetic** and is now the shared type's rule, not a guard at one call
site: `RowAction.offersMenu` is the decision and `.rowActionMenu(_:perform:)` is the only door
(a `View` overload and a `TableRowContent` one, because `TableRow` is not a `View`), so a row with
no applicable action gets NO `.contextMenu` attached — never an empty one, never a disabled
placeholder. Accounts is the surface that has the case: `AccountsRow.menu(named:)` is empty for an
unset token AND for an environment one, which has nothing on this machine to clear. The other
three call sites were swept in the same test: every machine row and every config-entry shape —
env-locked, null secret, empty string, and a curated row with Reset filtered out — still resolves
to at least one action, so none of them can reach the empty case by accident.

One informational note stands, stated rather than silently dropped: `HostPersistence.decode` still
answers `nil` when EVERY element is malformed. The bytes are parked, nothing is lost, and
re-seeding a list that read as nothing is the lesser wrong.

Not this lane, and marked so for Lane F: **H1**, **H2**, **H3**, **H4**, **M1–M10**, **M14–M17**,
**L1** (CI), **L2**, **L3**, **L4**, **L8**, **L9** are engine / FFI / release / CI. **L7**
(`keyboardShortcut` on a `Menu`, `Shell/MoldCommands.swift`) is a shell menu item nobody's lane
list names; it is untouched here.

## What the fixes are

- **`SecretStore`** (`Packages/MoldClient/Sources/MoldClient/SecretStore*.swift`) is a port of
  `desktop/src-tauri/src/secrets.rs`: flat `{"name": "value"}` under
  `~/Library/Application Support/io.utensils.mold.native/secrets.json`, the temp file created
  `0600` AT creation and `fsync`ed before an atomic `rename(2)` (`replaceItemAt` would carry the
  OLD file's mode onto the new one), an unparseable file parked once as `secrets.json.corrupt`, an
  unreadable one refused outright, one `Mutex` and one `flock(2)` around the whole
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

## Cross-lane — every file this lane touched outside its own list

Sequence the picks around these. Everything else the lane touched is in PLAN.md:130-131's own
list (`Support/{Keychain,HostPersistence,HostStore+Editing,AppStorageSuite}.swift`,
`Settings/**`, `Shell/{HostEditor,MachinesSettings,AccountsSettings}.swift`) or is a new file or
a test.

**Lane B — `Generate/**`**

- `Generate/GeneratePane+UAT.swift` (2 hunks, `67184017`): `MOLD_NATIVE_SOURCE_IMAGE` and
  `MOLD_NATIVE_LIBRARY_PICKER` read through `NativeUAT` instead of `ProcessInfo`, so a Release
  build ignores them. `GenerateUAT.envVar` keeps its name and value. No Debug behaviour change.

**Lane D — `Queue/**`, `Machines/**`**

- `Queue/QueuePane+UAT.swift` (1 hunk, `67184017`): `MOLD_NATIVE_QUEUE_FIXTURE`, same change.
- `Machines/PairingSection+UAT.swift` (1 hunk, `67184017`): `MOLD_NATIVE_PAIRING_FIXTURE`, same.

**Lane A — `Packages/MoldClient/**`**

- `StoredHost.swift` (`f11a4779`) and `Tests/MoldClientTests/StoredHostTests.swift` (`d022c847`):
  doc-comment sentences only, naming `SecretStore` where they named the Keychain.
- NEW files, no conflict surface: `SecretStore.swift`, `SecretStore+File.swift`,
  `SecretStore+Local.swift`, `SecretStoreError.swift`, `Tests/…/SecretStoreTests.swift`.

**Unassigned by PLAN.md, but touched here**

- `Support/HostStore.swift` (1 hunk, `67184017`) — `MOLD_NATIVE_HOSTS` through `NativeUAT`.
- `Support/NativeUAT.swift`, `Support/LegacyKeychain.swift` — NEW (the latter replaces the
  deleted `Support/Keychain.swift`).
- `Shell/RootView.swift` (2 hunks, `67184017`) — `MOLD_NATIVE_DESTINATION` through `NativeUAT`.
- `Shell/GeneralSettings.swift` (2 hunks, `0322a6ea`) — the Reset button's two sentences, so
  behaviour and copy agree (05-M13).
- `Shell/ProviderSection.swift` (1 hunk, `05ed1032`) — a `.contextMenu` and a 4-line `menu`
  property; it is the Accounts row `Shell/AccountsSettings.swift` renders.
- NEW in `Shell/`: `RowAction.swift`, `MachineRowActions.swift`, `MachineRemoval.swift`,
  `Clipboard.swift`.

Every `67184017` hunk is the same mechanical substitution
(`ProcessInfo.processInfo.environment["MOLD_NATIVE_…"]` → `NativeUAT.<case>.value()`), so a
conflict there resolves by taking whichever side has the other lane's logic and re-applying the
substitution.

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

Every finding this lane took — from `05-shell-engine-release.md` and from the adversarial review
alike — reproduced exactly as described.

## Verification

`make lint` green (three pre-existing large-type advisories only, none of them this lane's).
`swift test` in `Packages/MoldClient`: **431** passed, and the same suite passes under
`swift test -c release`. Full app bundle `xcodebuild test`: **411** in 62 suites passed. A Release
build was made once to prove the `#else` arms compile and to count the hook strings in each binary
(Debug dylib: all eight; Release executable: none); its output was deleted.

One flake seen once, on a test this lane does not touch and which passes alone and on a re-run of
the whole bundle: `ModelActionsTests.theMenuAndTheContextualMenuCallTheSameThing`. Recorded here
rather than ignored.
