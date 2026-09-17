import Foundation
import MoldClient

/// Sending a HELD job to another machine -- the driver behind
/// `TransferPlan`'s pure decisions, the studio's own `sendHeldQueueJob`
/// (`queueTransfer.ts:50-199`) ported through it (design M6 S4).
///
/// Its own type, not a `QueueStore` extension: a two-machine orchestration
/// with its own in-flight state and its own success caption is a concern of
/// its own, not a queue listing's. `TransferStore+Steps.swift` holds
/// everything after the identity check, split out purely for size.
@MainActor
@Observable
final class TransferStore {
    let hosts: HostStore
    /// For the poll after `complete` settles, and nothing else -- the row's
    /// new position or absence is the machine's to state, `QueueStore`'s own
    /// rule for every mutation.
    let queue: QueueStore

    /// The row/destination a transfer is in flight for -- one at a time,
    /// app-wide, or two would derive the same `client_batch_id` and race
    /// their own idempotency fence. `internal(set)`: a test sets this
    /// directly to pin the guard without racing two real transfers.
    internal(set) var transferring: (entry: String, destination: MoldHost.ID)?

    /// A `.sent` outcome's own sentence, shown for ~8 seconds --
    /// `ModelsPane+Actions.report(_:)`'s own treatment for a one-off success.
    /// `.refused` is not captioned here; it already went through
    /// `hosts.report`.
    internal(set) var summary: String?

    init(hosts: HostStore, queue: QueueStore) {
        self.hosts = hosts
        self.queue = queue
    }

    /// A machine that is up and that generates at all. Absent
    /// `heterogeneous_batch_max_outputs` means it does not generate -- one
    /// admission path, so absence is a refusal (`Capabilities.generates`,
    /// decision 19).
    struct TransferDestination: Identifiable, Hashable {
        let id: MoldHost.ID
        let name: String
        let queueDepth: Int?

        var caption: String { queueDepth.map { "\(name) — \($0) queued" } ?? name }
    }

    func transferDestinations(from source: MoldHost.ID) -> [TransferDestination] {
        guard let sourceInstance = hosts.instanceID(of: source) else { return [] }
        return hosts.hosts.compactMap { host -> TransferDestination? in
            guard host.id != source, hosts.isUp(host), hosts.capabilities[host.id]?.generates == true,
                  hosts.instanceID(of: host.id) != sourceInstance
            else { return nil }
            var depth: Int?
            if case let .up(status) = hosts.reachability(of: host) { depth = status.queueDepth }
            return TransferDestination(id: host.id, name: host.name, queueDepth: depth)
        }
    }

    /// `nil` when nothing was attempted (already mid-transfer, a fixture, an
    /// unresolved identity -- each already reported through `hosts`);
    /// otherwise the plan's final word. A `.sent` outcome is captioned onto
    /// `summary`; a `.refused` is already reported and needs nothing more.
    @discardableResult
    func transfer(_ entry: QueueEntry, from source: MoldHost.ID, to destination: MoldHost.ID) async -> TransferOutcome? {
        let destinationName = hosts.name(of: destination) ?? "another machine"
        let verb = "send that job to \(destinationName)"
        guard !queue.refuseIfFixture(source, doing: verb), transferring == nil else { return nil }
        guard let sourceHost = hosts.host(source), let destClient = hosts.backend(for: destination),
              let sourceClient = hosts.backend(for: source),
              let expectedSource = hosts.instanceID(of: source), let expectedDest = hosts.instanceID(of: destination)
        else {
            hosts.report(NoInstanceKnown(), on: source, doing: verb)
            return nil
        }
        transferring = (entry.id, destination)
        defer { transferring = nil }

        let clientBatchId = QueueTransferID.derive(source: expectedSource, jobId: entry.id, destination: expectedDest)
        let context = TransferContext(
            source: source, sourceClient: sourceClient, sourceName: sourceHost.name,
            destClient: destClient, destName: destinationName,
            expectedDest: expectedDest, clientBatchId: clientBatchId, verb: verb)

        do {
            let sourceStatus = try await sourceClient.status()
            let destStatus = try await destClient.status()
            guard case .step(.checkPriorAttempt) = context.next(.verifyIdentities, .identities(
                sourceChanged: sourceStatus.instanceId != expectedSource,
                destinationChanged: destStatus.instanceId != expectedDest))
            else { return report(.refused("A machine's identity changed. Refresh the machines and try again."), context) }
        } catch {
            hosts.report(error, on: source, doing: verb)
            return nil
        }
        let outcome = await afterIdentities(entry, authorityInstance: expectedSource, context: context)
        if case let .sent(_, message) = outcome { caption(message) }
        return outcome
    }

    private func caption(_ message: String) {
        summary = message
        Task {
            try? await Task.sleep(for: .seconds(8))
            if summary == message { summary = nil }
        }
    }
}
