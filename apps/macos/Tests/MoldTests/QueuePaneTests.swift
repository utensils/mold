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
    private func machine(_ name: String = "workstation") -> MoldHost {
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

    /// UAT 2026-09-17 #6: a queued row's contextual menu offered Pause Job
    /// and Cancel Job while the Queue menu offered neither, with the row
    /// selected. Every render this app submits is a batch of one, drawn as a
    /// plain row under its BATCH id -- the id the `List` selection carries.
    ///
    /// **Fails today**: the pane looked the selection up among ENTRY ids.
    @Test func aSelectedBatchOfOneResolvesToItsEntryForTheQueueMenu() {
        let solo = FakeFixtures.queueEntry("solo", state: "queued")
        let only = FakeFixtures.queueEntry("only", state: "queued", batchId: "b1", batchIndex: 1)
        let pair = (1 ... 2).map {
            FakeFixtures.queueEntry("p\($0)", state: "queued", batchId: "b2", batchIndex: $0)
        }
        let groups = QueueGroup.build([solo, only] + pair, children: [:])

        #expect(QueueGroup.selectedEntry("solo", in: groups)?.id == "solo")
        #expect(QueueGroup.selectedEntry("b1", in: groups)?.id == "only")
        // The batch's own disclosure row is not a job.
        #expect(QueueGroup.selectedEntry("b2", in: groups) == nil)
        #expect(QueueGroup.selectedEntry("only", in: groups) == nil)
    }

    // MARK: - Group action

    /// **Fails today**: `QueueStore.act(_:onLiveChildrenOf:host:)` does not
    /// exist yet.
    @Test func aGroupButtonReachesEveryLiveChildAndNoSettledOne() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        hosts.capabilities[workstation.id] = FakeFixtures.capabilities(cooperativeCancellation: true)
        let queue = QueueStore(hosts: hosts)
        let group = QueueGroup.build(mixedBatch, children: [:])[0]

        await queue.act(.cancel, onLiveChildrenOf: group, host: workstation.id)

        #expect(fake.callCount("cancelJob") == 2)
    }

    /// The group dispatch asks each child what THIS machine will honour, so
    /// a batch's Cancel on a host that cannot stop running work clears the
    /// waiting child and leaves the running one alone -- rather than sending
    /// a call the machine would refuse (`types.rs:11475-11479`).
    @Test func aGroupCancelSkipsARunningChildTheMachineCannotStop() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        hosts.capabilities[workstation.id] = FakeFixtures.capabilities()
        let queue = QueueStore(hosts: hosts)
        let group = QueueGroup.build(mixedBatch, children: [:])[0]

        await queue.act(.cancel, onLiveChildrenOf: group, host: workstation.id)

        #expect(fake.callCount("cancelJob") == 1)
    }

    /// One batch with a child in each of the four states that matter.
    private var mixedBatch: [QueueEntry] {
        [
            FakeFixtures.queueEntry("live-1", state: "queued", batchId: "batch"),
            FakeFixtures.queueEntry("live-2", state: "running", batchId: "batch"),
            FakeFixtures.queueEntry("done", state: "complete", batchId: "batch"),
            FakeFixtures.queueEntry("dead", state: "cancelled", batchId: "batch"),
        ]
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
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.queueListing = FakeFixtures.queueListing(["job-1"])
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let queue = QueueStore(hosts: hosts)

        await queue.reorder([("c1", 0), ("c2", 1)], on: workstation.id)

        #expect(fake.reorders.map(\.id) == ["c1", "c2"])
        #expect(fake.reorders.map(\.position) == [0, 1])
        #expect(fake.callCount("queue") == 1)
    }

    // MARK: - Empty Queue

    /// **Fails today**: `QueuePane.emptyQueueTargets` does not exist yet.
    @Test func emptyQueueIsAbsentOnAMachineThatDoesNotAdvertiseIt() {
        let workstation = machine("workstation")
        let hal = machine("hal9000")
        let capabilities: [MoldHost.ID: Capabilities] = [
            workstation.id: FakeFixtures.capabilities(canCancelAll: false),
            hal.id: FakeFixtures.capabilities(canCancelAll: true),
        ]
        let targets = QueuePane.emptyQueueTargets([workstation, hal], capabilities: capabilities)
        #expect(targets.map(\.id) == [hal.id])
    }

    @Test func emptyQueueIsAbsentWhenNoCapabilitiesHaveArrivedAtAll() {
        let workstation = machine()
        #expect(QueuePane.emptyQueueTargets([workstation], capabilities: [:]).isEmpty)
    }

    /// A machine holding work is a target even without the bulk route:
    /// clearing a hold is the one action every hold has.
    @Test func aMachineHoldingWorkCanBeEmptiedWithoutTheBulkRoute() {
        let workstation = machine("workstation")
        let hal = machine("hal9000")
        let capabilities: [MoldHost.ID: Capabilities] = [
            workstation.id: FakeFixtures.capabilities(canCancelAll: false),
            hal.id: FakeFixtures.capabilities(canCancelAll: false),
        ]
        let entries = [workstation.id: [FakeFixtures.queueEntry("h", state: "held")]]
        let targets = QueuePane.emptyQueueTargets(
            [workstation, hal], capabilities: capabilities, entries: entries)
        #expect(targets.map(\.id) == [workstation.id])
    }

    /// Pins fact 12's whole point: running work is untouched, and the
    /// confirm has to say so.
    @Test func theEmptyQueueConfirmSaysRunningWorkKeepsGoing() {
        let message = QueueEmptyConfirm.message(.init(waiting: 3, paused: 1))
        #expect(
            message
                == "3 waiting and 1 paused job will be cancelled. Anything already rendering keeps going."
        )
    }

    @Test func theEmptyQueueConfirmSingularizesOneOfEach() {
        let message = QueueEmptyConfirm.message(.init(waiting: 1, paused: 1))
        #expect(message.hasPrefix("1 waiting and 1 paused job "))
    }

    @Test func theEmptyQueueConfirmCountsHeldJobsAndMachines() {
        let one = QueueEmptyConfirm.message(.init(waiting: 3, paused: 1, held: 12))
        #expect(one.hasPrefix("3 waiting, 1 paused and 12 held jobs will be cancelled."))
        let fleet = QueueEmptyConfirm.message(.init(held: 12), machines: 3)
        #expect(fleet.hasPrefix("12 held jobs across 3 machines will be cancelled."))
        #expect(QueueEmptyConfirm.message(.init()).hasPrefix("Nothing is waiting or held"))
    }

    @Test func theConfirmCountsComeFromTheRowsStates() {
        let counts = QueueEmptyConfirm.Counts([
            FakeFixtures.queueEntry("q"), FakeFixtures.queueEntry("p", state: "paused"),
            FakeFixtures.queueEntry("h", state: "held"), FakeFixtures.queueEntry("r", state: "running"),
        ])
        #expect(counts == .init(waiting: 1, paused: 1, held: 1))
    }

    // MARK: - Empty

    private func emptyBench(canCancelAll: Bool, entries: [QueueEntry]) async
        -> (QueueStore, FakeBackend, MoldHost) {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        fake.serverStatus = FakeFixtures.serverStatus(queuePaused: nil)
        fake.capabilityBlock = FakeFixtures.capabilities(canCancelAll: canCancelAll)
        fake.exportBlock = FakeFixtures.exportOptions()
        fake.queueListing = FakeFixtures.queueListing(entries: entries)
        fake.cancelAllAnswer = FakeFixtures.queueCancelResult(1)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        await hosts.refresh(workstation)
        return (QueueStore(hosts: hosts), fake, workstation)
    }

    private var mixed: [QueueEntry] {
        [
            FakeFixtures.queueEntry("q1"), FakeFixtures.queueEntry("p1", state: "paused"),
            FakeFixtures.queueEntry("h1", state: "held"), FakeFixtures.queueEntry("h2", state: "held"),
            FakeFixtures.queueEntry("r1", state: "running"),
        ]
    }

    /// The bug: `DELETE /api/queue` leaves every HELD row, so Empty Queue
    /// used to leave the pane exactly as full of holds as it was.
    @Test func emptyClearsTheWaitingRowsInBulkAndEveryHeldRow() async {
        let (queue, fake, workstation) = await emptyBench(canCancelAll: true, entries: mixed)

        await queue.empty(on: workstation.id)

        #expect(fake.callCount("cancelAllQueued") == 1)
        #expect(fake.cancelledIds == ["h1", "h2"], "running work is never touched")
    }

    /// Only its holds: a waiting row could start between the listing and a
    /// per-row DELETE, and that route cancels running work.
    @Test func aMachineWithoutTheBulkRouteHasOnlyItsHoldsCleared() async {
        let (queue, fake, workstation) = await emptyBench(canCancelAll: false, entries: mixed)

        await queue.empty(on: workstation.id)

        #expect(!fake.cancelledAll)
        #expect(fake.cancelledIds == ["h1", "h2"])
    }

    /// The held row's own × -- `DELETE /api/queue/:id` with the row's id.
    @Test func cancellingAHeldRowSendsItsOwnId() async throws {
        let (queue, fake, workstation) = await emptyBench(canCancelAll: true, entries: mixed)
        await queue.refresh()
        let held = try #require(queue.entries(on: workstation.id).first { $0.id == "h2" })

        await queue.cancel(held, on: workstation.id)

        #expect(fake.cancelledIds == ["h2"])
    }

    @Test func aRefusedHoldIsReportedAndTheRestStillGo() async {
        let (queue, fake, workstation) = await emptyBench(canCancelAll: true, entries: mixed)
        fake.refuses.insert("cancelJob")

        await queue.empty(on: workstation.id)

        #expect(fake.callCount("cancelJob") == 2, "one refusal does not stop the next hold")
        #expect(queue.hosts.failures.contains { $0.sentence.contains("empty its queue") })
    }

    // MARK: - Batch keyboard move

    /// A dragged batch's children have to land CONTIGUOUS and where the drop
    /// was, and `QueueOrder.moves` plans each `PATCH` against the queue the
    /// previous one left behind -- this pins that the keyboard twin reuses
    /// the identical translation, not a second one that could disagree.
    ///
    /// The positions used to be asserted as `[1, 2]`, which is the plan the
    /// OLD single-index-space `moves` produced and which the server lands as
    /// `[p1, c1] ... c2` -- the two children on either side of the row they
    /// were moved past (review 01#1). The contract is the final ORDER, so
    /// that is what this asserts, replayed the way the server resolves it
    /// (`generation_queue.rs:1815-1836`).
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
        #expect(calls.map(\.position) == [2, 2])

        var order = ["c1", "c2", "p1"]
        for call in calls {
            order.remove(at: order.firstIndex(of: call.id)!)
            order.insert(call.id, at: min(call.position, order.count))
        }
        #expect(order == ["p1", "c1", "c2"])
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
        let nothing = QueueSelection(job: nil, emptyQueues: [])
        #expect(nothing.offeredTitles.isEmpty)

        let runningJob = QueueSelection.Job(
            target: .init(host: UUID(), entry: "running"),
            canPause: true, canResume: false, canRetry: false, canMoveUp: false, canMoveDown: true,
            canCancel: true, moveToDestinations: [], pause: {}, resume: {}, retry: {}, moveUp: {},
            moveDown: {}, cancel: {}, moveTo: { _ in })
        let running = QueueSelection(job: runningJob, emptyQueues: [])
        #expect(running.offeredTitles == ["Pause Job", "Move Down", "Cancel Job"])

        let held = QueueSelection.Job(
            target: .init(host: UUID(), entry: "held"),
            canPause: false, canResume: false, canRetry: true, canMoveUp: false, canMoveDown: false,
            canCancel: true, moveToDestinations: [], pause: {}, resume: {}, retry: {}, moveUp: {},
            moveDown: {}, cancel: {}, moveTo: { _ in })
        let empty = QueueSelection.EmptyQueue(id: UUID(), name: "workstation", run: {})
        #expect(QueueSelection(job: held, emptyQueues: [empty]).offeredTitles
            == ["Try Again", "Cancel Job", "Empty Queue…"])
    }

    /// **Fails today**: `Job.moveToDestinations` does not exist yet.
    @Test func aHeldSelectionWithAMachineToSendToOffersMoveTo() {
        let destination = TransferStore.TransferDestination(id: UUID(), name: "hal9000", queueDepth: 2)
        let held = QueueSelection.Job(
            target: .init(host: UUID(), entry: "held"),
            canPause: false, canResume: false, canRetry: true, canMoveUp: false, canMoveDown: false,
            canCancel: true, moveToDestinations: [destination], pause: {}, resume: {}, retry: {},
            moveUp: {}, moveDown: {}, cancel: {}, moveTo: { _ in })
        #expect(QueueSelection(job: held, emptyQueues: []).offeredTitles
            == ["Try Again", "Move to", "Cancel Job"])
    }

    @Test func queueSelectionIdentityChangesWithTheSelectedRow() {
        let host = UUID()
        func job(_ entry: String) -> QueueSelection.Job {
            QueueSelection.Job(
                target: .init(host: host, entry: entry),
                canPause: true, canResume: false, canRetry: false,
                canMoveUp: false, canMoveDown: true, canCancel: true,
                moveToDestinations: [], pause: {}, resume: {}, retry: {},
                moveUp: {}, moveDown: {}, cancel: {}, moveTo: { _ in })
        }

        #expect(job("first") != job("second"))
    }

    @Test func severalEmptyQueuesAreNamedInsteadOfPickingTheFirst() {
        let first = QueueSelection.EmptyQueue(id: UUID(), name: "workstation", run: {})
        let second = QueueSelection.EmptyQueue(id: UUID(), name: "hal9000", run: {})
        let selection = QueueSelection(job: nil, emptyQueues: [first, second])

        #expect(selection.offeredTitles == [
            "Empty Queue on workstation…", "Empty Queue on hal9000…",
        ])
    }

    @Test func severalEmptyQueuesLeadWithAllMachines() {
        let first = QueueSelection.EmptyQueue(id: UUID(), name: "workstation", run: {})
        let second = QueueSelection.EmptyQueue(id: UUID(), name: "hal9000", run: {})
        var ran: [String] = []
        let selection = QueueSelection(job: nil, emptyQueues: [
            .init(id: nil, name: "All Machines") { ran.append("all") }, first, second,
        ])

        #expect(selection.offeredTitles == [
            "Empty Queue on All Machines…", "Empty Queue on workstation…", "Empty Queue on hal9000…",
        ])
        selection.perform(.emptyQueue(nil))
        #expect(ran == ["all"])
    }

    // MARK: - Fixture

    /// **Fails today**: `QueueStore.seed(from:)` does not exist yet. Every
    /// mutation on a seeded store sends nothing to the fake and reports
    /// through the same funnel a real refusal would (design M6 decision 27).
    @Test func aFixtureQueueRefusesEveryMutation() async {
        let workstation = machine()
        let fake = FakeBackend(host: workstation)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        let queue = QueueStore(hosts: hosts)
        let entry = FakeFixtures.queueEntry("job-1", state: "queued")
        let fixture = QueueStore.Fixture(hosts: [
            "workstation": .init(queue: FakeFixtures.queueListing(entries: [entry]), batches: nil)
        ])

        queue.seed(from: fixture)
        #expect(queue.entries(on: workstation.id).map(\.id) == ["job-1"])
        #expect(queue.isSeeded)

        await queue.cancel(entry, on: workstation.id)
        await queue.pause(entry, on: workstation.id)
        await queue.resume(entry, on: workstation.id)
        await queue.retry(entry, on: workstation.id)
        await queue.reorder([("job-1", 0)], on: workstation.id)
        await queue.empty(on: workstation.id)
        await queue.refresh()

        #expect(fake.calls.isEmpty)
        #expect(hosts.failures.contains { $0.sentence.contains("fixture") })
        // The refresh above never touched the network either -- the seeded
        // row is exactly what was planted, not overwritten with nothing.
        #expect(queue.entries(on: workstation.id).map(\.id) == ["job-1"])
    }
}
