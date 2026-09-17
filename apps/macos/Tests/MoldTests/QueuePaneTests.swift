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

    // MARK: - Batch keyboard move

    /// **Fails today**: `QueueBatchRow.moveCall` does not exist yet. A
    /// dragged batch's own children land contiguous only when the calls
    /// issue in ASCENDING target order (`QueueOrder.moves`'s own doc) -- this
    /// pins that the keyboard twin reuses the identical translation, not a
    /// second one that could disagree.
    @Test func aBatchMovesFromTheKeyboardAsAscendingCalls() {
        let entries = [
            FakeFixtures.queueEntry("c1", state: "queued", batchId: "b", batchIndex: 0),
            FakeFixtures.queueEntry("c2", state: "queued", batchId: "b", batchIndex: 1),
            FakeFixtures.queueEntry("p1", state: "queued"),
        ]
        let groups = QueueGroup.build(entries, children: [:])
        let batch = groups[0]
        #expect(batch.isExpandable)

        let calls = QueueBatchRow.moveCall(batch, .down, groups: groups, entries: entries)

        #expect(calls.map(\.id) == ["c1", "c2"])
        #expect(calls.map(\.position) == [1, 2])
    }

    @Test func aBatchAtTheTopCannotMoveUp() {
        let entries = [
            FakeFixtures.queueEntry("c1", state: "queued", batchId: "b", batchIndex: 0),
            FakeFixtures.queueEntry("c2", state: "queued", batchId: "b", batchIndex: 1),
            FakeFixtures.queueEntry("p1", state: "queued"),
        ]
        let groups = QueueGroup.build(entries, children: [:])
        #expect(QueueBatchRow.canMove(groups[0], .up, in: groups) == false)
        #expect(QueueBatchRow.canMove(groups[0], .down, in: groups))
    }

    // MARK: - Queue menu

    /// **Fails today**: `QueueSelection` does not exist yet.
    @Test func theQueueMenuOffersOnlyWhatApplies() {
        let nothing = QueueSelection(job: nil, emptyQueue: nil)
        #expect(nothing.offeredTitles.isEmpty)

        let runningJob = QueueSelection.Job(
            canPause: true, canResume: false, canRetry: false, canMoveUp: false, canMoveDown: true,
            canCancel: true, moveToDestinations: [], pause: {}, resume: {}, retry: {}, moveUp: {},
            moveDown: {}, cancel: {}, moveTo: { _ in })
        let running = QueueSelection(job: runningJob, emptyQueue: nil)
        #expect(running.offeredTitles == ["Pause Job", "Move Down", "Cancel Job"])

        let held = QueueSelection.Job(
            canPause: false, canResume: false, canRetry: true, canMoveUp: false, canMoveDown: false,
            canCancel: true, moveToDestinations: [], pause: {}, resume: {}, retry: {}, moveUp: {},
            moveDown: {}, cancel: {}, moveTo: { _ in })
        #expect(QueueSelection(job: held, emptyQueue: {}).offeredTitles == ["Try Again", "Cancel Job", "Empty Queue…"])
    }

    /// **Fails today**: `Job.moveToDestinations` does not exist yet.
    @Test func aHeldSelectionWithAMachineToSendToOffersMoveTo() {
        let destination = TransferStore.TransferDestination(id: UUID(), name: "hal9000", queueDepth: 2)
        let held = QueueSelection.Job(
            canPause: false, canResume: false, canRetry: true, canMoveUp: false, canMoveDown: false,
            canCancel: true, moveToDestinations: [destination], pause: {}, resume: {}, retry: {},
            moveUp: {}, moveDown: {}, cancel: {}, moveTo: { _ in })
        #expect(QueueSelection(job: held, emptyQueue: nil).offeredTitles == ["Try Again", "Move to", "Cancel Job"])
    }

    // MARK: - Fixture

    /// **Fails today**: `QueueStore.seed(from:)` does not exist yet. Every
    /// mutation on a seeded store sends nothing to the fake and reports
    /// through the same funnel a real refusal would (design M6 decision 27).
    @Test func aFixtureQueueRefusesEveryMutation() async {
        let plato = machine()
        let fake = FakeBackend(host: plato)
        let hosts = HostStore(hosts: [plato]) { _ in fake }
        let queue = QueueStore(hosts: hosts)
        let entry = FakeFixtures.queueEntry("job-1", state: "queued")
        let fixture = QueueStore.Fixture(hosts: [
            "plato": .init(queue: FakeFixtures.queueListing(entries: [entry]), batches: nil)
        ])

        queue.seed(from: fixture)
        #expect(queue.entries(on: plato.id).map(\.id) == ["job-1"])
        #expect(queue.isSeeded)

        await queue.cancel(entry, on: plato.id)
        await queue.pause(entry, on: plato.id)
        await queue.resume(entry, on: plato.id)
        await queue.retry(entry, on: plato.id)
        await queue.reorder([("job-1", 0)], on: plato.id)
        await queue.cancelAll(on: plato.id)
        await queue.refresh()

        #expect(fake.calls.isEmpty)
        #expect(hosts.failures.contains { $0.sentence.contains("fixture") })
        // The refresh above never touched the network either -- the seeded
        // row is exactly what was planted, not overwritten with nothing.
        #expect(queue.entries(on: plato.id).map(\.id) == ["job-1"])
    }
}
