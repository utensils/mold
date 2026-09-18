import Foundation
import MoldClient
import Testing

@testable import Mold

/// The driver behind `TransferPlan` -- `TransferStore.transfer(_:from:to:)`
/// -- wired to two independent `FakeBackend`s so the three real calls
/// (`export`, `admit`, `complete`) and their ordering are pinned without a
/// network (design M6 S4).
@MainActor
struct QueueTransferTests {
    private func machine(_ name: String) -> MoldHost {
        MoldHost(name: name, baseURL: URL(string: "http://\(name)")!)
    }

    /// Two machines, each already reporting the instance id a transfer fences
    /// on -- what `HostStore.instanceID(of:)` needs before anything starts.
    private func pair(sourceInstance: String = "src-1", destInstance: String = "dst-1") async
        -> (hosts: HostStore, transfers: TransferStore, source: MoldHost, destination: MoldHost,
            sourceFake: FakeBackend, destFake: FakeBackend)
    {
        let source = machine("workstation")
        let destination = machine("hal9000")
        let sourceFake = FakeBackend(host: source)
        let destFake = FakeBackend(host: destination)
        for fake in [sourceFake, destFake] {
            fake.capabilityBlock = FakeFixtures.capabilities(events: false)
            fake.exportBlock = FakeFixtures.exportOptions()
            // `complete` always polls the source afterward, the same
            // refresh-after-mutation `act` already does -- planted here so
            // that incidental read does not itself report a failure.
            fake.queueListing = FakeFixtures.queueListing([])
        }
        sourceFake.serverStatus = FakeFixtures.serverStatus(instanceId: sourceInstance)
        destFake.serverStatus = FakeFixtures.serverStatus(instanceId: destInstance)
        let hosts = HostStore(hosts: [source, destination]) { host in
            host.name == source.name ? sourceFake : destFake
        }
        await hosts.refresh(source)
        await hosts.refresh(destination)
        let queue = QueueStore(hosts: hosts)
        return (hosts, TransferStore(hosts: hosts, queue: queue), source, destination, sourceFake, destFake)
    }

    private func heldDetail(jobId: String, batchId: String = "b1", clientBatchId: String = "cb1") -> QueueJobDetail {
        let entry = FakeFixtures.queueEntry(jobId, state: "held", batchId: batchId, clientBatchId: clientBatchId)
        let entryJSON = String(data: try! MoldJSON.encoder.encode(entry), encoding: .utf8)!
        return try! MoldJSON.decoder.decode(QueueJobDetail.self, from: Data(#"{"job": \#(entryJSON)}"#.utf8))
    }

    private func landedBatch(
        instanceId: String, clientBatchId: String, states: [BatchChildState] = [.accepted]
    ) -> BatchStatus {
        let rows = states.enumerated()
            .map { #"{"index": \#($0.offset), "job_id": "j\#($0.offset)", "state": "\#($0.element.rawValue)"}"# }
            .joined(separator: ",")
        let json = #"""
        {"id": "b", "client_batch_id": "\#(clientBatchId)", "instance_id": "\#(instanceId)",
         "durable": true, "children": [\#(rows)]}
        """#
        return try! MoldJSON.decoder.decode(BatchStatus.self, from: Data(json.utf8))
    }

    /// **Fails today**: `TransferStore.transfer` does not exist yet.
    @Test func theHappyPathIsExportAdmitCompleteInOrderWithTheDerivedId() async {
        let (hosts, transfers, source, destination, sourceFake, destFake) = await pair()
        let entry = FakeFixtures.queueEntry("job-1", state: "held", batchId: "b1", clientBatchId: "cb1")
        sourceFake.queueJobDetails["job-1"] = heldDetail(jobId: "job-1")
        sourceFake.exportBodies["job-1"] = Data(#"{"prompt":"a cat"}"#.utf8)
        let clientId = QueueTransferID.derive(source: "src-1", jobId: "job-1", destination: "dst-1")
        destFake.admitAnswer = landedBatch(instanceId: "dst-1", clientBatchId: clientId)

        let outcome = await transfers.transfer(entry, from: source.id, to: destination.id)

        #expect(outcome == .sent(sourceRemoved: true, message: "Sent to hal9000. The original was removed from workstation's queue."))
        #expect(sourceFake.calls.filter { ["exportHeldJob", "completeTransfer"].contains($0) }
            == ["exportHeldJob", "completeTransfer"])
        #expect(destFake.calls.contains("admitTransfer"))
        #expect(destFake.transferAdmissions.first?.clientBatchId == clientId)
        #expect(sourceFake.completedTransfers.first?.jobId == "job-1")
        #expect(hosts.failures.isEmpty)
    }

    /// **Fails today**: no identity re-read happens at all.
    @Test func aRestartedMachineAbortsBeforeAnyWrite() async {
        let (hosts, transfers, source, destination, sourceFake, destFake) = await pair()
        // The destination answers with a DIFFERENT instance than the one
        // `hosts` cached -- it restarted between the picker and the click.
        destFake.serverStatus = FakeFixtures.serverStatus(instanceId: "dst-2-after-restart")
        let entry = FakeFixtures.queueEntry("job-1", state: "held", batchId: "b1", clientBatchId: "cb1")

        let outcome = await transfers.transfer(entry, from: source.id, to: destination.id)

        #expect(outcome == .refused("A machine's identity changed. Refresh the machines and try again."))
        #expect(!sourceFake.calls.contains("exportHeldJob"))
        #expect(!destFake.calls.contains("admitTransfer"))
        #expect(hosts.failures.first?.host == source.id)
    }

    /// **Fails today**: `checkPriorAttempt` is never asked at all.
    @Test func aPriorAttemptFoundOnTheDestinationSkipsStraightToComplete() async {
        let (_, transfers, source, destination, sourceFake, destFake) = await pair()
        let entry = FakeFixtures.queueEntry("job-1", state: "held", batchId: "b1", clientBatchId: "cb1")
        sourceFake.queueJobDetails["job-1"] = heldDetail(jobId: "job-1")
        let clientId = QueueTransferID.derive(source: "src-1", jobId: "job-1", destination: "dst-1")
        destFake.batchStatusByClientId[clientId] = landedBatch(instanceId: "dst-1", clientBatchId: clientId)

        let outcome = await transfers.transfer(entry, from: source.id, to: destination.id)

        #expect(outcome == .sent(sourceRemoved: true, message: "Sent to hal9000. The original was removed from workstation's queue."))
        #expect(!sourceFake.calls.contains("exportHeldJob"))
        #expect(!destFake.calls.contains("admitTransfer"))
        #expect(sourceFake.calls.contains("completeTransfer"))
    }

    /// **Fails today**: a 413 is not classified at all.
    @Test func a413SaysTooLargeAndLeavesTheSourceHeld() async {
        let (hosts, transfers, source, destination, sourceFake, destFake) = await pair()
        let entry = FakeFixtures.queueEntry("job-1", state: "held", batchId: "b1", clientBatchId: "cb1")
        sourceFake.queueJobDetails["job-1"] = heldDetail(jobId: "job-1")
        sourceFake.exportBodies["job-1"] = Data(#"{"prompt":"a cat"}"#.utf8)
        destFake.plantedErrors["admitTransfer"] = MoldClientError.http(status: 413, code: nil, message: nil)

        let outcome = await transfers.transfer(entry, from: source.id, to: destination.id)

        // The number is `MAX_REQUEST_BODY_BYTES` (`lib.rs:178`), spelled by
        // the one constant that holds it -- "about 48 MB" matched nothing.
        #expect(outcome == .refused(
            "hal9000 wouldn't take it: the job's media is larger than a machine will accept in one request "
                + "(64 MB). The original is still here."))
        #expect(!sourceFake.calls.contains("completeTransfer"))
        #expect(hosts.failures.first?.sentence.contains(RequestBodyLimit.sentence) == true)
    }

    /// **Fails today**: `completeTransfer` failing is not distinguished from
    /// the whole transfer failing.
    @Test func aFailedCompleteReportsSuccessWithTheCaveat() async {
        let (_, transfers, source, destination, sourceFake, destFake) = await pair()
        let entry = FakeFixtures.queueEntry("job-1", state: "held", batchId: "b1", clientBatchId: "cb1")
        sourceFake.queueJobDetails["job-1"] = heldDetail(jobId: "job-1")
        sourceFake.exportBodies["job-1"] = Data(#"{"prompt":"a cat"}"#.utf8)
        let clientId = QueueTransferID.derive(source: "src-1", jobId: "job-1", destination: "dst-1")
        destFake.admitAnswer = landedBatch(instanceId: "dst-1", clientBatchId: clientId)
        sourceFake.plantedErrors["completeTransfer"] = MoldClientError.http(status: 409, code: nil, message: nil)

        let outcome = await transfers.transfer(entry, from: source.id, to: destination.id)

        #expect(outcome == .sent(
            sourceRemoved: false,
            message: "Sent to hal9000. The original could not be removed; check workstation before retrying it."))
    }

    /// **Fails today**: a child already `failed` is never checked at all.
    @Test func aDestinationChildThatFailedLeavesTheSourceHeld() async {
        let (_, transfers, source, destination, sourceFake, destFake) = await pair()
        let entry = FakeFixtures.queueEntry("job-1", state: "held", batchId: "b1", clientBatchId: "cb1")
        sourceFake.queueJobDetails["job-1"] = heldDetail(jobId: "job-1")
        sourceFake.exportBodies["job-1"] = Data(#"{"prompt":"a cat"}"#.utf8)
        let clientId = QueueTransferID.derive(source: "src-1", jobId: "job-1", destination: "dst-1")
        destFake.admitAnswer = landedBatch(instanceId: "dst-1", clientBatchId: clientId, states: [.failed])

        let outcome = await transfers.transfer(entry, from: source.id, to: destination.id)

        #expect(outcome == .refused(
            "The destination job failed. The original remains held; choose another machine or inspect the destination."))
        #expect(!sourceFake.calls.contains("completeTransfer"))
    }

    /// **Fails today**: `queueJob` is never re-read, so a stale cached row
    /// would still try to export.
    @Test func aSourceThatIsNoLongerHeldNeverReachesTheDestination() async {
        let (_, transfers, source, destination, sourceFake, destFake) = await pair()
        let entry = FakeFixtures.queueEntry("job-1", state: "held", batchId: "b1", clientBatchId: "cb1")
        // The live row has since resumed and is no longer held -- the cached
        // `entry` handed to `transfer` still says `held`, which is exactly
        // what fact 15 says must not be trusted.
        let resumedEntry = FakeFixtures.queueEntry("job-1", state: "running", batchId: "b1", clientBatchId: "cb1")
        let resumedJSON = String(data: try! MoldJSON.encoder.encode(resumedEntry), encoding: .utf8)!
        sourceFake.queueJobDetails["job-1"] =
            try! MoldJSON.decoder.decode(QueueJobDetail.self, from: Data(#"{"job": \#(resumedJSON)}"#.utf8))

        let outcome = await transfers.transfer(entry, from: source.id, to: destination.id)

        #expect(outcome == .refused("This job is no longer held. Refresh the queue before sending it."))
        #expect(!sourceFake.calls.contains("exportHeldJob"))
        #expect(!destFake.calls.contains("admitTransfer"))
    }

    /// **Fails today**: `transferring` does not exist.
    @Test func onlyOneTransferRunsAtATime() async {
        let (_, transfers, source, destination, sourceFake, destFake) = await pair()
        let entry = FakeFixtures.queueEntry("job-1", state: "held", batchId: "b1", clientBatchId: "cb1")
        sourceFake.queueJobDetails["job-1"] = heldDetail(jobId: "job-1")
        transfers.transferring = (entry: "job-0", destination: destination.id)

        let outcome = await transfers.transfer(entry, from: source.id, to: destination.id)

        #expect(outcome == nil)
        #expect(!sourceFake.calls.contains("exportHeldJob"))
        #expect(!destFake.calls.contains("admitTransfer"))
    }

    // MARK: - transferDestinations

    private func generatingCapabilities() -> Capabilities {
        try! MoldJSON.decoder.decode(
            Capabilities.self, from: Data(#"{"queue":{"heterogeneous_batch_max_outputs":4}}"#.utf8))
    }

    /// **Fails today**: `transferDestinations` does not exist. `pair()`'s
    /// default capabilities carry no `queue` block at all, so this is the
    /// ordinary case, not a special one.
    @Test func aMachineThatDoesNotGenerateIsNotADestination() async {
        let (_, transfers, source, _, _, _) = await pair()
        #expect(transfers.transferDestinations(from: source.id).isEmpty)
    }

    /// The positive control: a reachable, generating machine IS offered.
    @Test func aMachineThatIsUpAndGeneratesIsOfferedAsADestination() async {
        let workstation = machine("workstation")
        let hal = machine("hal9000")
        let sourceFake = FakeBackend(host: workstation)
        let destFake = FakeBackend(host: hal)
        sourceFake.serverStatus = FakeFixtures.serverStatus(instanceId: "src-1")
        destFake.serverStatus = FakeFixtures.serverStatus(instanceId: "dst-1")
        sourceFake.capabilityBlock = FakeFixtures.capabilities(events: false)
        destFake.capabilityBlock = generatingCapabilities()
        let hosts = HostStore(hosts: [workstation, hal]) { $0.name == workstation.name ? sourceFake : destFake }
        await hosts.refresh(workstation)
        await hosts.refresh(hal)
        let transfers = TransferStore(hosts: hosts, queue: QueueStore(hosts: hosts))

        #expect(transfers.transferDestinations(from: workstation.id).map(\.name) == ["hal9000"])
    }

    /// A single machine (itself) offers nowhere to send a held job.
    @Test func aSingleUpMachineMeansNoMenu() async {
        let workstation = machine("workstation")
        let fake = FakeBackend(host: workstation)
        fake.serverStatus = FakeFixtures.serverStatus(instanceId: "src-1")
        fake.capabilityBlock = FakeFixtures.capabilities(events: false)
        let hosts = HostStore(hosts: [workstation]) { _ in fake }
        await hosts.refresh(workstation)
        let transfers = TransferStore(hosts: hosts, queue: QueueStore(hosts: hosts))
        #expect(transfers.transferDestinations(from: workstation.id).isEmpty)
    }
}
