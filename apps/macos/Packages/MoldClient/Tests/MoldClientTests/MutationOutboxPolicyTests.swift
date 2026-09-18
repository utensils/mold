import Foundation
import Testing

@testable import MoldClient

/// The policy that used to live beside `LibraryStore`'s drain loop: when to
/// send, how long to wait, and when to stop trying. It moved here because it
/// is arithmetic over the queue, not a screen concern, and belongs beside the
/// `Entry` it decides about.
@Suite struct MutationOutboxPolicySuite {
    let workstation = UUID()

    private func edit(_ change: PrintChange, _ targets: [MoldHost.ID: [String]]) -> PrintEdit {
        PrintEdit(change: change, targets: targets)
    }

    @Test func aFreshEntryIsSent() {
        var outbox = MutationOutbox()
        let queued = outbox.enqueue(edit(.favorite(true), [workstation: ["a.png"]]))
        #expect(outbox.next(for: workstation) == .send(queued[0]))
    }

    /// A retry does not resend immediately -- it waits, and for exactly the
    /// entry that failed, not whatever `next` is asked about later.
    @Test func aFailedEntryWaitsThenIsSentAgain() {
        var outbox = MutationOutbox()
        let queued = outbox.enqueue(edit(.favorite(true), [workstation: ["a.png"]]))
        outbox.retry(queued[0].id)
        let retried = outbox.head(for: workstation)!
        #expect(outbox.next(for: workstation) == .wait(MutationOutbox.backoff(after: 1), then: retried))
    }

    /// The fourth failure is where the outbox itself decides enough is
    /// enough -- it removes the entry and names the rows nothing later still
    /// speaks for, the same rule `failed` has always applied.
    @Test func theFourthFailureGivesUpAndNamesWhatNothingSupersedes() {
        var outbox = MutationOutbox()
        let queued = outbox.enqueue(edit(.favorite(true), [workstation: ["a.png", "b.png"]]))
        for _ in 0..<outbox.maxAttempts { outbox.retry(queued[0].id) }

        guard case let .giveUp(entry, orphaned) = outbox.next(for: workstation) else {
            Issue.record("expected .giveUp")
            return
        }
        #expect(entry.id == queued[0].id)
        #expect(orphaned == ["a.png", "b.png"])
        #expect(outbox.isEmpty)
    }

    /// A title change is a PATCH on one print, not a bulk mutation -- it
    /// carries no fence because setting a title twice is setting a title.
    @Test func theWireFormOfATitleChangeIsAPatch() {
        var outbox = MutationOutbox()
        let queued = outbox.enqueue(edit(.title(from: "Old", to: "New"), [workstation: ["a.png", "b.png"]]))

        guard case let .patch(patch, filenames) = queued[0].wire else {
            Issue.record("expected .patch")
            return
        }
        #expect(patch.title == "New")
        #expect(filenames == ["a.png", "b.png"])
    }

    /// The host applies a given operation id once, so a retry after a
    /// dropped connection must reuse it -- a fresh id per attempt is exactly
    /// the double-apply the fence exists to prevent.
    @Test func theOperationIdIsStableAcrossRetries() {
        var outbox = MutationOutbox()
        let queued = outbox.enqueue(edit(.favorite(true), [workstation: ["a.png"]]))
        outbox.retry(queued[0].id)

        guard case let .mutate(mutation) = outbox.head(for: workstation)!.wire else {
            Issue.record("expected .mutate")
            return
        }
        #expect(mutation.operationId == queued[0].id)
    }
}
