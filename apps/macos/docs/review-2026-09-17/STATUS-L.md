# Lane L — consolidation (type budgets, file budgets, one flaky test)

`make lint` on the lane's tip prints:

```
  large: Sources/Mold/Library/ReuseStore.swift (211 lines)
  large type: HTTPBackend (1402 lines across its files)
```

Nothing else. `ReuseStore.swift` is left ALONE at the integrator's request —
a UAT fix is landing there and a split would collide with it.

## What moved, and why each new type exists

| step | type | before → after | the new type, and its reason |
| --- | --- | --- | --- |
| 1 | `GenerateController` | 603 → 454 | `ExpandStore` — a rewrite is a round trip with its own backend calls, its own staleness fence and its own result, and a submit reads none of it. Composed and `.environment()`'d like every other store. |
| 2 | `LibraryActions` | 657 → 559 | `PrintImport` — a batch with a policy: what one unreadable file in the middle of ten means, what a machine refusing means, one line about the lot when it is done. Named for what it moves, because `LibraryImport` is already the Import menu's model. |
| 2 | `LibraryActions.swift` | 151 → 122 | `save(_:)` moved beside `saveAll` in `+Files`, where the rest of "turning prints into files on this disk" lives. |
| 3 | `LibraryStore` | 779 → 653 → 540 | `GalleryLive` — the live RECONCILER, and the only thing that needs `GalleryEcho` and the `RelistGate`, which moved with it. Then `LibraryTags` — the tag INDEX, and renaming or deleting a tag everywhere, which is one request per machine and not a loop over prints. |
| 4 | `RenderDraft` | 915 → 753 → 498 | `CanvasFit` — a size and a `ResolutionProfile`; it never reads or writes a draft. Then `RenderRequest` — the translation to the wire, which answers questions the draft holds no opinion on (which well ships, what the DESTINATION host understands about identity photos, how a batch of four fans out). |
| 5 | `HTTPBackend` | 1420 → 1402 | NOT restructured (README says why). `EmptyBody` and `CollectionCreate`/`CollectionChange` moved — the first is posted by three route groups, the second two are `MoldBackend`'s vocabulary. |
| 6 | `GeneratePane.swift` | 211 → 137 | `+Run.swift` — pressing Generate: the clip's routing, the one retained picture a chain must be handed first, and the probe both ask. |
| 6 | `LibraryPane.swift` | 156 → 90 | `+Chrome.swift` — what the pane re-reads when the fleet or the shelves move, the chrome it wears, and what is on screen. |
| 8 | `HostStore` | 613 → 588 | `HostFailure` — a different type declared inside `HostStore+Failures.swift`. A miscount, not debt; same rule as `EmptyBody`. |

Every one takes the thing it serves as a PARAMETER — the arrangement
`LibraryMutations` already had with `LibraryStore`. No forwarders, no renamed
files, `TYPE_MAX` untouched. All are pure moves except `CanvasFit`, noted below.

## Judgements worth naming

- **`RenderRequest` did not go on `GenerateRequest`.** The brief offered
  `GenerateRequest(from draft)` first; `GenerateRequest` sums 372, so the 255
  lines would have put IT 27 over. `RenderDraft(reusing:)` stays on the draft
  because what it makes is a draft.
- **`CanvasFit` is not a pure move in one place, deliberately.**
  `sourceExactCanvas` built a throwaway `RenderDraft` purely to call
  `fit(to:)` on it, because the rule was only reachable through a draft. It
  calls the rule directly now, and `fit(to:)` is GONE rather than left as a
  wrapper. 913 package tests green either side.
- **The remaining `HTTPBackend` request bodies stay where they are.** Each is
  one route's own body, and a route's body is part of its route group. Only
  the shared and the public ones moved.
- **`LibraryStore` needed two moves, not one.** `+Live` alone is 125 lines and
  779 − 125 = 654, still over.

## Step 7 — `HeartbeatTests/oneSlowMachineDoesNotHoldUpTheRest`

| what it relied on | what it relies on now |
| --- | --- |
| `await settle { quickBackend.callCount("queue") == 1 }` — two seconds of POLLING, and then an assertion that blamed concurrency when the truth was load | `FakeBackend.entered(_:atLeast:)` — a continuation resumed from inside `record`, so how busy the box is cannot reach it |

The test now states both halves as LATCHES rather than as an order the task
group does not promise: the first machine HAS been asked and nothing has
answered it, and the second is asked anyway.

Two things the fence exposed that a budget had been hiding:

1. **The waiter list has to be under `callsLock`, the same lock `recorded`
   is.** `record` does run off the main actor — that is why `calls` is locked
   at all — so an unsynchronised waiter list loses a wake-up it was entitled
   to. One of the first twenty runs took the full 60-second hang for exactly
   that. Checking whether to park and deciding whom to wake are now one hold
   of one lock.
2. **A missing call must be a sentence, not a hang.** `within:` is a watchdog
   only — never what makes a passing run pass — and it names the route and
   prints the call list. `.timeLimit(.minutes(1))` is the backstop behind it.

Proof:

- Against a deliberately serial `tick()` (the regression this test exists for):
  **FAILS**, `Time limit was exceeded: 60.000 seconds`. Restored after.
- `for i in $(seq 20)`, whole `HeartbeatTests` suite, after the lock fix:
  **20 passed / 0 failed**. (Before the lock fix: 19/20 and 20/20 — the one
  failure is what led to it.)

## Gates

- `make lint` — the two lines quoted at the top, nothing else.
- `MoldClient` package `swift test` — 913 tests, 49 suites, green.
- App bundle `xcodebuild test`, full `MoldTests` — green (under the shared
  mkdir lock).

## Cross-lane edits

- `Sources/Mold/AppStores.swift` + `AppStores+Environment.swift` — one line
  each, at the two markers, for `ExpandStore`.
- `Sources/Mold/Generate/PromptWand*.swift`, `PromptPanel.swift` — read the
  store instead of the controller.
- `Sources/Mold/Library/LibraryPane+Menu.swift`, `LibraryMutations*.swift`,
  `LibraryActions+Organize.swift`, `LibraryPane+Toolbar.swift`,
  `LibraryStore+{Rows,Organization,Editing}.swift` — re-pointed receivers.
- `Sources/Mold/Support/HostStore+Failures.swift` — `HostFailure` extracted.
  (`HostStore+Default.swift`, rewritten on the branch in `d8ebaa63`, is
  untouched.)
- `Packages/MoldClient/…` — 69 call sites re-pointed at `RenderRequest`,
  almost all of them tests.
- `Tests/MoldTests/FakeBackend.swift` — `entered(_:atLeast:within:)` added and
  `record` now takes the waiters it wakes under the existing lock.

## Not done, and why

- `ReuseStore.swift` (211) — the integrator asked for it to be left alone.
- `HTTPBackend` (1,402) — the README states the reason; it is not a threshold
  to raise.
