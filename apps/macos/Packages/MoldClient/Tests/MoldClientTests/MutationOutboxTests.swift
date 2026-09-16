import Foundation
import Testing

@testable import MoldClient

/// The queue that carries an organization edit to the machines.
///
/// Its whole reason to exist is what happens when a send fails: an optimistic
/// change is already on screen, and the question is whether the screen is now
/// wrong. The answer is not "always" -- a newer edit for the same print may
/// still be in flight, and reverting would undo something the person did after.
@Suite struct MutationOutboxSuite {
    let plato = UUID()
    let hal = UUID()

    private func edit(_ change: PrintChange, _ targets: [MoldHost.ID: [String]]) -> PrintEdit {
        PrintEdit(change: change, targets: targets)
    }

    // MARK: - Queueing

    @Test func anEditBecomesOneEntryPerMachine() {
        var outbox = MutationOutbox()
        let queued = outbox.enqueue(edit(.favorite(true), [plato: ["a.png"], hal: ["b.png"]]))
        #expect(queued.count == 2)
        #expect(Set(queued.map(\.host)) == [plato, hal])
    }

    /// The fence the server applies once. A fresh id per attempt is exactly the
    /// double-apply it exists to prevent, so the id is minted with the entry.
    @Test func everyEntryCarriesItsOwnStableOperationID() {
        var outbox = MutationOutbox()
        let queued = outbox.enqueue(edit(.favorite(true), [plato: ["a.png"], hal: ["b.png"]]))
        #expect(Set(queued.map(\.id)).count == 2)
        #expect(outbox.head(for: plato)?.id == queued.first { $0.host == plato }?.id)
    }

    @Test func anEmptyEditQueuesNothing() {
        var outbox = MutationOutbox()
        #expect(outbox.enqueue(edit(.favorite(true), [:])).isEmpty)
        #expect(outbox.isEmpty)
    }

    // MARK: - One chain per machine

    @Test func eachMachineIsItsOwnQueueInOrder() {
        var outbox = MutationOutbox()
        _ = outbox.enqueue(edit(.favorite(true), [plato: ["a.png"]]))
        _ = outbox.enqueue(edit(.tag("owls", adding: true), [plato: ["a.png"]]))
        _ = outbox.enqueue(edit(.favorite(true), [hal: ["b.png"]]))

        #expect(outbox.head(for: plato)?.change == .favorite(true))
        #expect(outbox.head(for: hal)?.change == .favorite(true))
        outbox.succeeded(outbox.head(for: plato)!.id)
        #expect(outbox.head(for: plato)?.change == .tag("owls", adding: true))
    }

    /// A machine that is behind must never hold up another. They are separate
    /// servers; there is no fleet-wide order to preserve.
    @Test func oneMachineFallingBehindDoesNotBlockTheOther() {
        var outbox = MutationOutbox()
        _ = outbox.enqueue(edit(.favorite(true), [plato: ["a.png"], hal: ["b.png"]]))
        outbox.succeeded(outbox.head(for: hal)!.id)
        #expect(outbox.head(for: hal) == nil)
        #expect(outbox.head(for: plato) != nil)
    }

    // MARK: - Giving up

    @Test func givingUpNamesTheRowsToRepair() {
        var outbox = MutationOutbox()
        let queued = outbox.enqueue(edit(.favorite(true), [plato: ["a.png", "b.png"]]))
        #expect(outbox.failed(queued[0].id) == ["a.png", "b.png"])
        #expect(outbox.isEmpty)
    }

    /// The rule the outbox exists for. A later edit still speaks for "a.png",
    /// so the older failure must not revert it, must not re-read it, and must
    /// not report it -- the screen is showing the NEWER intent, which has not
    /// failed.
    @Test func aRowWithANewerEditPendingIsLeftAlone() {
        var outbox = MutationOutbox()
        let first = outbox.enqueue(edit(.favorite(true), [plato: ["a.png", "b.png"]]))
        _ = outbox.enqueue(edit(.tag("owls", adding: true), [plato: ["a.png"]]))
        #expect(outbox.failed(first[0].id) == ["b.png"])
    }

    @Test func aRowSupersededOnAnotherMachineIsStillRepairedHere() {
        var outbox = MutationOutbox()
        let first = outbox.enqueue(edit(.favorite(true), [plato: ["a.png"]]))
        // Same filename, different machine. Different print entirely.
        _ = outbox.enqueue(edit(.tag("owls", adding: true), [hal: ["a.png"]]))
        #expect(outbox.failed(first[0].id) == ["a.png"])
    }

    @Test func failingAnEntryThatIsNoLongerQueuedRepairsNothing() {
        var outbox = MutationOutbox()
        let queued = outbox.enqueue(edit(.favorite(true), [plato: ["a.png"]]))
        outbox.succeeded(queued[0].id)
        #expect(outbox.failed(queued[0].id).isEmpty)
    }

    // MARK: - Retrying

    @Test func aRetryKeepsTheEntryAtTheHeadAndCountsTheAttempt() {
        var outbox = MutationOutbox()
        let queued = outbox.enqueue(edit(.favorite(true), [plato: ["a.png"]]))
        outbox.retry(queued[0].id)
        #expect(outbox.head(for: plato)?.id == queued[0].id)
        #expect(outbox.head(for: plato)?.attempts == 2)
    }

    /// Whether trying again could plausibly work. Retrying something that
    /// cannot succeed is not resilience -- it is a spinner that never stops.
    @Test func onlyTheFailuresThatCouldPassLaterAreRetried() {
        #expect(MoldClientError.unreachable("offline").isTransient)
        #expect(MoldClientError.http(status: 503, code: nil, message: nil).isTransient)
        #expect(MoldClientError.http(status: 429, code: nil, message: nil).isTransient)
        // A key does not appear by waiting, a 404 does not become a 200, and
        // a reply this build cannot parse will not parse next time either.
        #expect(!MoldClientError.unauthorized.isTransient)
        #expect(!MoldClientError.http(status: 404, code: nil, message: nil).isTransient)
        #expect(!MoldClientError.http(status: 422, code: nil, message: nil).isTransient)
        #expect(!MoldClientError.malformedResponse.isTransient)
    }

    @Test func theWholeChainIsReadableInOrderForReplay() {
        var outbox = MutationOutbox()
        _ = outbox.enqueue(edit(.favorite(true), [plato: ["a.png"]]))
        _ = outbox.enqueue(edit(.tag("owls", adding: true), [plato: ["a.png"]]))
        #expect(outbox.chain(for: plato).map(\.change)
            == [.favorite(true), .tag("owls", adding: true)])
        #expect(outbox.chain(for: hal).isEmpty)
    }

    @Test func machinesWithWorkAreTheOnesToDrain() {
        var outbox = MutationOutbox()
        _ = outbox.enqueue(edit(.favorite(true), [plato: ["a.png"], hal: ["b.png"]]))
        outbox.succeeded(outbox.head(for: plato)!.id)
        #expect(outbox.waiting == [hal])
    }
}
