# Lane H — the sidebar's Library section IS the destination

The owner's observation: the sidebar carried a top-level **Library** row (with
Generate, Queue, Models, Machines) AND, directly below it, a **Library**
section whose first row, All Prints, opened the same thing. Photos, Music and
Mail have ONE Library group whose rows are the destinations.

| id | status | commit | test |
| --- | --- | --- | --- |
| H1 No top-level Library row | fixed | `06c19ef8` | `SidebarRowsTests.theLibraryIsNotATopLevelRow` |
| H2 One highlighted row across both groups | fixed | `06c19ef8` | `…exactlyOneDrawnRowIsHighlighted`, `…theHighlightedLibraryRowIsTheScopeOnShow`, `…aMachineRowIsHighlightedOverItsSection` |
| H3 A section row enters the library at its scope | fixed | `06c19ef8` | `…pickingAShelfRowEntersTheLibraryAtThatScope` |
| H4 ⌘2 / View ▸ Library unchanged | fixed (no code change) | `06c19ef8` | `…theDestinationShortcutsAreUnmoved`, `…viewLibraryOpensTheShelfYouLeft` |
| H5 `MOLD_NATIVE_DESTINATION=library` still lands | fixed | `5bd05c38` | `…theUATHookLandsOnAllPrints`, `…aLaunchWithNoHookOpensWhereYouWere` |
| H6 README's sidebar sentence | fixed | this commit | — |

## The `Destination` decision (brief item 3)

**`.library` stays a case; the section's rows select it.** The alternative --
folding `LibraryScope` into the destination -- would have rewritten every
caller that sets `destination = .library` (notification routing, Show in
Library from Generate and from a machine row, `@AppStorage("destination")`,
`Destination.launch` and the hook's raw value, the ⌘1–⌘5 loop, and
`DestinationDetail`) for no behaviour anyone can see, and would have made the
remembered destination and the remembered scope two halves of one persisted
value that can disagree.

The single-highlight rule is made structural a cheaper way: `SidebarRows`
(new, pure, 59 lines) owns BOTH directions of the sidebar's selection.
`selected(destination:scope:machine:)` returns exactly ONE `SidebarRow` for any
state the window can be in -- so two rows cannot be highlighted, by
construction rather than by two bindings agreeing -- and `pick(_:)` says what
choosing a row means. `SidebarRows.destinations` is `Destination.allCases`
minus `.library`: the top group draws that list, the Library section draws the
shelves, and the type that decides which is highlighted is the same one that
decides what is listed. `Sidebar.Row` (private) became the shared `SidebarRow`;
the sidebar view kept its rows, its counts, its drag targets and its menus.

## ⌘2 means "the library, where you left it"

Chosen from what the app already does for the other destinations, not invented:
`destination` itself is remembered across launches (`RootView`'s
`@AppStorage("destination")`), the picked machine is remembered
(`selectedMachine`), and `LibraryNavigation.scope` has ALWAYS been remembered
across launches, deliberately (`LibraryNavigation.swift` -- "nothing about a
click survives a relaunch the way `scope` and `edge` do"). Forcing All Prints
would have been the one destination that forgets. It needed no code change: the
command sets `destination = .library` and says nothing about the scope.

On a library nobody has moved -- a first launch, and every `MOLD_NATIVE_FRESH`
UAT run -- that scope IS `.all`, so ⌘2 and `MOLD_NATIVE_DESTINATION=library`
both land on All Prints. Both halves are pinned:
`viewLibraryOpensTheShelfYouLeft` asserts the fresh domain gives All Prints AND
that a library left on Favourites reopens on Favourites.

Notification routing keeps its own override and is untouched: a notification
sets `scope = .all` first (`NotificationRouting.swift`), because the print it
names may not be in the collection you were last on.

## What else was checked, and left alone

- Every caller that sets `destination = .library` still works, and each now
  highlights the shelf row rather than a duplicate: `applyNotificationRoute`,
  `GeneratePane+Result`'s Show in Library, `MachineRow`'s Show in Library
  (which sets a machine token and keeps the scope), and the mesh viewer's,
  which routes through `GeneratePane`'s same closure.
- Contextual menus are untouched: Recently Deleted keeps its one
  `rowActionMenu` (Empty Trash…), collections keep `CollectionRow`'s, machines
  keep `SidebarMachineActions`. Nothing is bound twice, and the removed row had
  no menu of its own to lose.
- `MoldCommands` was NOT edited. Its View group enumerates
  `Destination.allCases` by index, which the sidebar no longer reads, so
  ⌘1–⌘5 are unmoved -- pinned by `theDestinationShortcutsAreUnmoved` rather
  than left to trust.
- `Destination.launch` gained an injectable twin so the UAT hook's answer is
  testable; the property every caller uses hands in the real environment and
  the real suite, and `MoldApp.swift` (another lane's file) is untouched.

## Tests changed

None: no existing test asserted the sidebar's rows, the selection binding, or
`Destination.launch`. `SidebarRowsTests` (9 tests) is new. `NativeUATTests`
still passes -- the hook is still read only through `NativeUAT`, now through
its `value(in:)` door.

## Cross-lane edits

None.

## Gates

`make lint` clean (the four pre-existing `large`/`large type` advisories only,
none of them this lane's). App-bundle `xcodebuild test`:
`MoldTests/SidebarRowsTests` and `MoldTests/NativeUATTests` green under the
shared lock; package `swift test` green.
