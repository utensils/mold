import Foundation
import MoldClient
import Testing

@testable import Mold

/// The one authority behind a queue row's four actions, wherever they are
/// drawn -- the glyph buttons, the row's contextual menu, a batch's group
/// buttons and the Queue menu. Pure, so every gate is pinned with no
/// rendered pane, which is also the only way to pin a contextual menu at all:
/// `AXShowMenu` on a SwiftUI `List` row answers -25206.
@MainActor
struct QueueRowActionsTests {
    private func entry(_ state: String, retryable: Bool? = nil) -> QueueEntry {
        FakeFixtures.queueEntry("job-1", state: state, model: "flux-dev:q4", retryable: retryable)
    }

    /// Everything a current machine advertises.
    private let modern = FakeFixtures.capabilities(
        canReorder: true, canCancelAll: true, canPauseJob: true, cooperativeCancellation: true)

    // MARK: - Pause / Resume

    /// **Fails today**: the row's own Pause/Resume glyphs read only the
    /// row's state, while the Queue menu gates the same action on
    /// `can_pause_job` (`QueuePane+Commands.swift:28-31`) -- so on a machine
    /// that predates per-job pause the button is offered where the menu item
    /// is absent, and pressing it fails.
    @Test func pauseAndResumeAreOfferedOnlyWhereTheMachineAdvertisesThem() {
        let older = FakeFixtures.capabilities()
        #expect(QueueRowActions.resolve(entry("queued"), on: modern).pause)
        #expect(QueueRowActions.resolve(entry("paused"), on: modern).resume)
        #expect(!QueueRowActions.resolve(entry("queued"), on: older).pause)
        #expect(!QueueRowActions.resolve(entry("paused"), on: older).resume)
        // An absent block is the same definitive no as an explicit `false`.
        #expect(!QueueRowActions.resolve(entry("queued"), on: nil).pause)
    }

    /// `set_one_queue_job_paused` refuses a running row by name -- "queue job
    /// {id} is already running; only waiting jobs can be paused or resumed"
    /// (`routes.rs:7706-7710`) -- so Pause there was a guaranteed 409.
    @Test func aRunningRowIsNotOfferedPause() {
        #expect(!QueueRowActions.resolve(entry("running"), on: modern).pause)
        #expect(!QueueRowActions.resolve(entry("running"), on: modern).resume)
    }

    // MARK: - Cancel

    /// **Fails today**: `cooperative_cancellation` is decoded and no file in
    /// `Sources/`, `Tests/` or `Packages/` reads it, so the ✕ is offered on
    /// every live row including a running one.
    ///
    /// The server states what its absence means: "Older servers omit this and
    /// clients keep running rows read-only" (`types.rs:11475-11479`). Web
    /// gates on it (`useQueueInspection.ts:153-155`) and re-checks at action
    /// time (`:299-305`).
    @Test func aRunningRowIsCancellableOnlyWhereTheMachineCanStopWorkSafely() {
        #expect(QueueRowActions.resolve(entry("running"), on: modern).cancel)

        let older = FakeFixtures.capabilities(canPauseJob: true)
        #expect(!QueueRowActions.resolve(entry("running"), on: older).cancel)
        #expect(!QueueRowActions.resolve(entry("running"), on: nil).cancel)
    }

    /// Nothing is running on a queued, paused or held row, so
    /// `DELETE /api/queue/:id` clears it on every machine -- including one
    /// too old to say anything about cooperative cancellation.
    @Test func aWaitingHeldOrPausedRowIsCancellableEverywhere() {
        let older = FakeFixtures.capabilities()
        for state in ["queued", "held", "paused"] {
            #expect(QueueRowActions.resolve(entry(state), on: older).cancel)
        }
    }

    /// A settled row has no per-row delete route at all -- retention sweeps
    /// it (`d9421ae0`).
    @Test func aSettledRowOffersNothing() {
        for state in ["complete", "failed", "cancelled"] {
            #expect(QueueRowActions.resolve(entry(state), on: modern) == QueueRowActions())
        }
    }

    // MARK: - Retry

    @Test func retryIsOfferedOnAHeldRowUnlessTheMachineSaidItWouldNotHelp() {
        #expect(QueueRowActions.resolve(entry("held"), on: modern).retry)
        #expect(QueueRowActions.resolve(entry("held", retryable: true), on: modern).retry)
        #expect(!QueueRowActions.resolve(entry("held", retryable: false), on: modern).retry)
        #expect(!QueueRowActions.resolve(entry("queued"), on: modern).retry)
    }

    // MARK: - A batch

    /// A batch offers what ANY of its children do, and the dispatch then asks
    /// each child again -- one waiting and one running child pauses the
    /// waiting one rather than refusing both.
    @Test func aBatchOffersTheUnionOfItsChildren() {
        let rows = [
            FakeFixtures.queueEntry("a", state: "queued"),
            FakeFixtures.queueEntry("b", state: "running"),
        ]
        let group = QueueRowActions.group(rows, on: modern)
        #expect(group.pause)
        #expect(group.cancel)
        #expect(!group.resume)
        #expect(!QueueRowActions.resolve(rows[1], on: modern).pause)
    }

    /// A batch of running children on a machine that cannot stop running work
    /// offers no group cancel at all -- absent, not disabled.
    @Test func aBatchOfRunningChildrenOffersNoCancelWhereTheMachineCannot() {
        let rows = [FakeFixtures.queueEntry("a", state: "running")]
        #expect(!QueueRowActions.group(rows, on: FakeFixtures.capabilities()).cancel)
    }

    // MARK: - The contextual menu

    /// **Fails today**: `menuTitles` is a HAND-MAINTAINED COPY of the row's
    /// `@ViewBuilder`. Reorder or rename an item in the view and this still
    /// passes, and neither list carries a divider, so "destructive last,
    /// behind a divider" -- the rule this whole pass is about -- is pinned
    /// nowhere at all.
    ///
    /// `offered(...)` is now THE list `.rowActionMenu` renders, so asserting
    /// on it asserts on the menu: its order, its gating, its wording, and
    /// which item is destructive.
    @Test func theContextualMenuIsTheListTheViewDraws() {
        let waiting = QueueRowActions.resolve(entry("queued"), on: modern)
            .offered(canMoveUp: true, canMoveDown: true)
        #expect(waiting.map(\.title) == ["Pause Job", "Move Up", "Move Down", "Cancel Job"])
        // Destructive LAST -- which is where `RowActionMenu` draws the
        // divider -- and it is the only one.
        #expect(waiting.map(\.isDestructive) == [false, false, false, true])
        #expect(waiting.map(\.kind) == [.pause, .moveUp, .moveDown, .cancel])

        let held = QueueRowActions.resolve(entry("held"), on: modern).offered()
        #expect(held.map(\.title) == ["Try Again", "Cancel Job"])

        let paused = QueueRowActions.resolve(entry("paused"), on: modern).offered()
        #expect(paused.map(\.title) == ["Resume Job", "Cancel Job"])

        // Nothing offered means NO MENU -- `.rowActionMenu` attaches none, so
        // a right-click on a settled row does not open an empty one.
        let settled = QueueRowActions.resolve(entry("complete"), on: modern).offered()
        #expect(settled.isEmpty)
        #expect(!RowAction.offersMenu(settled))
    }

    /// And it says the same things the Queue menu says, in the same order --
    /// asserted against that menu's own list rather than against a comment
    /// claiming they agree.
    @Test func theRowsMenuAndTheQueueMenuOfferTheSameWords() {
        for state in ["queued", "running", "paused", "held", "complete"] {
            let row = QueueRowActions.resolve(entry(state), on: modern)
            let selection = QueueSelection(
                job: QueueSelection.Job(
                    canPause: row.pause, canResume: row.resume, canRetry: row.retry,
                    canMoveUp: true, canMoveDown: true, canCancel: row.cancel,
                    moveToDestinations: [], pause: {}, resume: {}, retry: {},
                    moveUp: {}, moveDown: {}, cancel: {}, moveTo: { _ in }),
                emptyQueue: nil)
            #expect(selection.offeredTitles
                == row.offered(canMoveUp: true, canMoveDown: true).map(\.title),
                "\(state)")
        }
    }

    /// A batch's own menu names the reach of each item, and is likewise
    /// destructive-last. Retry is never a group action -- it needs a
    /// `QueueAuthority` per row.
    @Test func aBatchsContextualMenuNamesEveryJobItReaches() {
        let rows = [
            FakeFixtures.queueEntry("a", state: "queued"),
            FakeFixtures.queueEntry("b", state: "paused"),
        ]
        let offered = QueueRowActions.group(rows, on: modern)
            .groupOffered(canMoveUp: false, canMoveDown: true)
        #expect(offered.map(\.title)
            == ["Pause Every Job", "Resume Every Job", "Move Down", "Cancel Every Job"])
        #expect(offered.last?.isDestructive == true)

        let held = [FakeFixtures.queueEntry("a", state: "held")]
        #expect(!QueueRowActions.group(held, on: modern).groupOffered()
            .contains { $0.kind == .retry })
    }
}
