import Foundation
import MoldClient
import Testing

@testable import Mold

/// The pane's own pure logic -- the subtitle's words, which group a set of
/// rows draws as, a group action's reach, the drag gesture's translation
/// into a PATCH, and the Empty Queue gate and its confirm -- all pulled out
/// of the view so none of this needs a rendered `List` (design M6 S3).
@MainActor
struct QueuePaneTests {
    private func machine(_ name: String = "plato") -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    // MARK: - Subtitle

    @Test func theSubtitleNeverCountsAHeldRowAsWaiting() {
        let entries = [
            FakeFixtures.queueEntry("a", state: "queued"),
            FakeFixtures.queueEntry("b", state: "held"),
            FakeFixtures.queueEntry("c", state: "held"),
        ]
        #expect(QueueSummary.sentence(entries) == "1 waiting · 2 held")
    }

    @Test func theSubtitleNamesRenderingSeparatelyAndElidesToIdle() {
        let entries = [
            FakeFixtures.queueEntry("a", state: "queued"),
            FakeFixtures.queueEntry("b", state: "running"),
        ]
        #expect(QueueSummary.sentence(entries) == "1 waiting · 1 rendering")
        #expect(QueueSummary.sentence([]) == "Idle")
        #expect(QueueSummary.sentence([FakeFixtures.queueEntry("c", state: "complete")]) == "Idle")
    }

    // MARK: - Grouping

    /// **Fails today** only in spirit -- `QueueGroup.build` already does
    /// this (S2a); pinned again here because it is what `QueuePane` actually
    /// draws a row from.
    @Test func aBatchOfOneIsAPlainRowAndABatchOfFourIsAGroup() {
        let solo = FakeFixtures.queueEntry("solo", state: "queued")
        var children: [QueueEntry] = []
        for index in 1 ... 4 {
            children.append(
                FakeFixtures.queueEntry(
                    "c\(index)", state: "queued", batchId: "batch", batchIndex: index))
        }
        let groups = QueueGroup.build([solo] + children, children: [:])
        #expect(groups.count == 2)
        #expect(groups[0].isExpandable == false)
        #expect(groups[1].isExpandable)
        #expect(groups[1].rows.count == 4)
    }

    // MARK: - Group action

    /// **Fails today**: `QueueStore.act(_:onLiveChildrenOf:host:)` does not
    /// exist yet.
    @Test func aGroupButtonReachesEveryLiveChildAndNoSettledOne() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let queue = QueueStore(hosts: hosts)
        let entries = [
            FakeFixtures.queueEntry("live-1", state: "queued", batchId: "batch"),
            FakeFixtures.queueEntry("live-2", state: "running", batchId: "batch"),
            FakeFixtures.queueEntry("done", state: "complete", batchId: "batch"),
            FakeFixtures.queueEntry("dead", state: "cancelled", batchId: "batch"),
        ]
        let group = QueueGroup.build(entries, children: [:])[0]

        await queue.act(.cancel, onLiveChildrenOf: group, host: plato.id)

        #expect(fake.callCount("cancelJob") == 2)
    }

    // MARK: - Reorder

    /// **Fails today**: `QueuePane.reorderCalls` does not exist, and the
    /// obvious implementation -- the row's index on screen -- would send
    /// `2`, not the reorderable candidate's own position of `1` (design M6
    /// fact 2).
    @Test func movingARowAsksTheStoreForQueueOrdersPositionNotTheScreenIndex() {
        let entries = [
            FakeFixtures.queueEntry("running", state: "running"),
            FakeFixtures.queueEntry("q1", state: "queued"),
            FakeFixtures.queueEntry("q2", state: "queued"),
            FakeFixtures.queueEntry("q3", state: "queued"),
        ]
        let groups = QueueGroup.build(entries, children: [:])
        // Dragging q1 (screen index 1) down past q2, to sit before q3.
        let calls = QueuePane.reorderCalls(
            source: IndexSet(integer: 1), destination: 3, groups: groups, entries: entries)
        #expect(calls.map(\.id) == ["q1"])
        #expect(calls.map(\.position) == [1])
    }

    /// **Fails today**: `QueueStore.reorder(_:on:)` does not exist yet.
    @Test func aBatchMoveIssuesAscendingPatchesThenOnePoll() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let queue = QueueStore(hosts: hosts)

        await queue.reorder([("c1", 0), ("c2", 1)], on: plato.id)

        #expect(fake.reorders.map(\.id) == ["c1", "c2"])
        #expect(fake.reorders.map(\.position) == [0, 1])
        #expect(fake.callCount("queue") == 1)
    }

    // MARK: - Empty Queue

    /// **Fails today**: `QueuePane.emptyQueueTargets` does not exist yet.
    @Test func emptyQueueIsAbsentOnAMachineThatDoesNotAdvertiseIt() {
        let plato = machine("plato")
        let hal = machine("hal9000")
        let capabilities: [MoldHost.ID: Capabilities] = [
            plato.id: FakeFixtures.capabilities(canCancelAll: false),
            hal.id: FakeFixtures.capabilities(canCancelAll: true),
        ]
        let targets = QueuePane.emptyQueueTargets([plato, hal], capabilities: capabilities)
        #expect(targets.map(\.id) == [hal.id])
    }

    @Test func emptyQueueIsAbsentWhenNoCapabilitiesHaveArrivedAtAll() {
        let plato = machine()
        #expect(QueuePane.emptyQueueTargets([plato], capabilities: [:]).isEmpty)
    }

    /// Pins fact 12's whole point: running work is untouched, and the
    /// confirm has to say so.
    @Test func theEmptyQueueConfirmSaysRunningWorkKeepsGoing() {
        let message = QueueEmptyConfirm.message(waiting: 3, paused: 1)
        #expect(
            message
                == "3 waiting and 1 paused job will be cancelled. Anything already rendering keeps going."
        )
    }

    @Test func theEmptyQueueConfirmSingularizesOneOfEach() {
        let message = QueueEmptyConfirm.message(waiting: 1, paused: 1)
        #expect(message.hasPrefix("1 waiting and 1 paused job "))
    }

    // MARK: - Cancel all

    /// **Fails today**: `QueueStore.cancelAll(on:)` does not exist yet.
    @Test func cancelAllCallsCancelAllQueuedOnceThenPolls() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        fake.queueListing = FakeFixtures.queueListing(["job-1"])
        fake.cancelAllAnswer = FakeFixtures.queueCancelResult(3)
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let queue = QueueStore(hosts: hosts)

        await queue.cancelAll(on: plato.id)

        #expect(fake.cancelledAll)
        #expect(fake.callCount("cancelAllQueued") == 1)
        #expect(fake.callCount("queue") == 1)
    }
}
