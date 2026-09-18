# Lane J — Machines is a fleet overview

The owner's ask, from a screenshot of a pane that showed ONE machine: "an area
where you can clearly see the network machines and all their stats from a high
level and then add and remove them from there … a top-level view showing all
the connected nodes and what their status is."

So the Machines DESTINATION (⌘5, View ▸ Machines) is now the FLEET, and a
machine's existing page is one push inside it. Nothing about the page itself
changed: it is the same `MachinesPane`, reached from a card instead of being
the whole destination.

| id | status | commit | test |
| --- | --- | --- | --- |
| J1 One card per machine, pure and testable | fixed | `f7beea10` | `MachineCardTests` (15 tests) |
| J2 The card's menu, the grid's keyboard, the launch hook | fixed | `32e00625` | `MachineFleetTests` (10 tests) |
| J3 The Machine menu draws the card's own list | fixed | `d4b37afb` | `MenuBarTests`, `MachineFleetTests` |
| J4 The overview, Nearby, Add/Remove, navigation | fixed | `d4b37afb` | `MachineFleetTests`, `MenuSurfaceTests` |
| J5 `MOLD_NATIVE_MACHINE`, README | fixed | `97f2ca70`, `d4b37afb` | `NativeUATTests` (pin 9 → 10) |

## What a card is

`MachineCard` (pure value) + `MachineCardFigures` (the aggregation) + `MachineFleet`
(the seam to the four stores a machine's page already reads). The view
(`MachineCardView`) renders and decides nothing.

- Name, status dot and the SIDEBAR's own wording for the state
  (`HostStore.Reachability.summary`), a Default badge, This Mac marked as such.
- GPUs collapsed the way a person says them -- "4× NVIDIA L40S", the same
  phrase `ServerStatus.hardware` already builds for the host editor, pinned
  against it by a test -- with one aggregate load figure and one video-memory
  bar summed across the cards.
- System memory, "N queued, M running" and "N installed · X GB" -- both of the
  last two are `MachineFigures`' own sentences, not a second spelling.
- **An absent figure draws no row.** `MachineFigures` prints an em dash on a
  machine's page, where the label is already there; a card has no such frame.
- **A DOWN machine prints no figures.** `MachineStore` deliberately keeps the
  device rows a machine last answered with, and printing them beside a red dot
  claims they are current -- the page itself refuses to (`unreachable`). It
  keeps its place in the grid, dimmed, and says why. CHECKING keeps its figures,
  or the grid would blank once a second during a refresh.
- Sort: default machine first, then `localizedStandardCompare` by name.

## Live, without a second stream

The overview takes NO resource stream. `MachineStore`'s 1 Hz stream is single
by construction and belongs to the machine whose page is open; a fleet of them
would cost a frame per machine per second forever. The overview asks each
machine that is UP for one sample when it appears (`MachineFleet.load`), the
queue keeps itself current from `/api/events` once loaded, and ⌘R / the toolbar
Refresh asks everything again. Machines that are NOT up are not asked at all --
otherwise pressing Back filed one failure per dead machine into the banner.

## Navigation, and the one piece of state

`NavigationStack`'s path IS `selectedMachine`, the preference the sidebar
already writes (`MachineNavigation`): a sidebar machine row opens that
machine's page, a card opens it, and Back empties the preference, which is what
deselects the row. A second piece of state would have been a second answer.

`MachinesDestination` publishes `machineSelection` and `refreshAction` ONCE for
both halves -- two views offering the same focused value in one scene is two
answers to "which machine", decided by whichever SwiftUI asked last. Both
`focusedSceneValue` lines moved off `MachinesPane`, and `MachinesPane`'s own
`machineSelection` builder is gone.

## One menu, three surfaces

`MachineCardActions.offered(isThisMac:isDefault:)` is the declaration: Open,
Check Now, Set as Default, Copy Address, Edit…, ——, Remove…. It feeds the
card's contextual menu, the card's tap, and the Machine menu bar
(`MachineCommands` now draws `RowActionMenu` over it, keeping ⇧⌘R and adding
⌘⌫ for Remove — the app's own "this row leaves" chord, as in the Library and
the Queue). Set as Default is ABSENT on the default machine; Edit… and Remove…
are ABSENT on This Mac, whose address is whatever port the engine bound. Open
takes no chord: Return opens the focused card, and binding Return in the menu
bar would take it from every default button in the app.

`MachineSelection` lost `check`/`setDefault` and gained `offered`/`perform`, so
the menu bar can no longer spell an item the card does not.

## Add and remove

The Add button and a card's Edit… open the SAME `HostEditor` sheet Settings ▸
Machines opens -- the one that normalizes the address, probes it live and
refuses a duplicate. Remove raises the SAME `MachineRemoval.destruction`
confirm, and its action is `HostStore.remove`, which is where forgetting the
key, the watchers and the default-ness already lives (`remove` clears
`defaultMachine` when it removed the default; nothing re-picks one today, and
this lane did not invent that behaviour).

Nearby aggregates the peers of every machine that advertises `canBrowsePeers`,
deduplicated by normalized address, through the same `PeerAction`. Its one item
per row is now `PeerAction.offered(for:)`, read by the machine's page too, so
"Add" and "Add…" cannot drift between the two surfaces.

## Deviation from the brief

The brief asked for `MOLD_NATIVE_DESTINATION=machines:<host-name>`. That
spelling needs `Destination.launch` (in `RootView.swift`) to parse a `:` suffix,
and `RootView.swift` belongs to another lane in this wave. The hook is the
brief's own "+1 `NativeUAT` case" instead: `MOLD_NATIVE_MACHINE=<name>` beside
`MOLD_NATIVE_DESTINATION=machines`, matched on the machine's name without
regard to case, landing on the overview when it matches nothing. The count pin
moved 9 → 10. Nothing outside this feature had to change.

## Cross-lane edits

NOT made, listed as the brief asks (another lane holds `Sidebar.swift` this
wave). ONE line, in `Sources/Mold/Shell/Sidebar.swift`, in `selection`'s
setter:

```swift
case let .destination(item):
    // Picking the Machines destination itself asks for the FLEET, not for
    // whichever machine was last open -- the overview's path IS this
    // preference (`MachineNavigation`).
    if item == .machines { selectedMachine = "" }
    destination = item
```

Without it, the sidebar's top-level **Machines** row is inert while a
machine's page is open: the row's own getter maps the destination back to the
open machine, so the selection snaps straight back. The Back control in the
toolbar still returns to the fleet, which is the primary route, and a card,
the machine rows and the UAT hooks all work as they are. `View ▸ Machines`
(⌘5) likewise keeps the page you left open -- ordinary `NavigationStack`
behaviour, and it would need `MoldCommands.swift` (also another lane's) to
carry `selectedMachine` to change it. Neither is a defect in this lane's own
files; both are one line each in files this lane may not touch.

## What a UAT screenshot must show

1. **Overview** (`MOLD_NATIVE_DESTINATION=machines`): the title "Machines · N",
   a grid of cards, the default machine's card FIRST with its Default badge,
   This Mac marked, a card carrying "4× …" with a load figure and two memory
   bars, and a dimmed card with a reason on any machine that is down.
2. **A card's menu** (right click): Open, Check Now, Set as Default, Copy
   Address, Edit…, a divider, Remove… in red -- and on This Mac's card, only
   the first four.
3. **The detail** (`MOLD_NATIVE_DESTINATION=machines MOLD_NATIVE_MACHINE=workstation`):
   the unchanged machine page, titled with the machine's name, with a Back
   control in the toolbar.
4. **The Add sheet** (Add a Machine… on the overview): the "Add a Machine"
   host editor with Address/Name/API key and its live check line -- the same
   sheet Settings ▸ Machines opens.
