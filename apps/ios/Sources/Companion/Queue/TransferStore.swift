import Foundation
import MoldClient

/// Move to…: a held job sent to another machine in the three calls the Mac
/// and the web app make -- export, admit, complete -- driven by the shared
/// `TransferPlan`, so a retry after a dropped connection resumes rather than
/// duplicating (the client batch id is derived, not minted).
@Observable
final class TransferStore {
    private(set) var transferring: String?

    @ObservationIgnored let hosts: HostStore
    @ObservationIgnored let queue: QueueStore

    init(hosts: HostStore, queue: QueueStore) {
        self.hosts = hosts
        self.queue = queue
    }

    /// Other machines that are up, generate, and are not the same instance.
    func destinations(from source: MoldHost.ID) -> [MoldHost] {
        guard let sourceInstance = hosts.instanceID(of: source) else { return [] }
        return hosts.hosts.filter { host in
            host.id != source && hosts.isUp(host) && hosts.capabilities[host.id]?.generates == true
                && hosts.instanceID(of: host.id) != sourceInstance
        }
    }

    func transfer(_ entry: QueueEntry, from source: MoldHost.ID, to destination: MoldHost.ID) async {
        guard transferring == nil, let sourceHost = hosts.host(source), let destHost = hosts.host(destination),
              let expectedSource = hosts.instanceID(of: source), let expectedDest = hosts.instanceID(of: destination)
        else { return }
        let verb = String(localized: "send that job to \(destHost.name)")
        transferring = entry.id
        defer { transferring = nil }
        let context = Context(sourceClient: hosts.backend(for: sourceHost), destClient: hosts.backend(for: destHost),
                              sourceName: sourceHost.name, destName: destHost.name, expectedDest: expectedDest,
                              clientBatchId: QueueTransferID.derive(source: expectedSource, jobId: entry.id,
                                                                   destination: expectedDest))
        do {
            let sourceStatus = try await context.sourceClient.status()
            let destStatus = try await context.destClient.status()
            guard case .step(.checkPriorAttempt) = context.next(.verifyIdentities, .identities(
                sourceChanged: sourceStatus.instanceId != expectedSource,
                destinationChanged: destStatus.instanceId != expectedDest))
            else { return finish(.refused(String(localized: "A machine's identity changed. Refresh the machines and try again.")), sourceHost, verb) }
            let prior = try await lookup(context.destClient, context.clientBatchId)
            let detail = try? await context.sourceClient.queueJob(id: entry.id)
            let authority = detail?.job.state == .held ? detail?.job.authority(instanceId: expectedSource) : nil
            switch context.next(.checkPriorAttempt, .priorAttempt(prior, sourceHeld: authority != nil)) {
            case let .outcome(outcome): return finish(outcome, sourceHost, verb)
            case .step(.complete): return finish(await complete(authority, context), sourceHost, verb)
            case .step(.export): break
            default: return
            }
            guard let authority else { return }
            let portable = try await context.sourceClient.exportHeldJob(authority)
            do {
                let admitted = try await context.destClient.admitTransfer(
                    clientBatchId: context.clientBatchId, portable: portable, destinationInstance: expectedDest)
                if case let .outcome(outcome) = context.next(.admit, .admitResult(.landed(admitted))) {
                    return finish(outcome, sourceHost, verb)
                }
            } catch {
                // An admit that failed ambiguously may still have landed: ask.
                if case let .outcome(outcome) = context.next(.admit, .admitResult(TransferPlan.classifyAdmitFailure(error))) {
                    return finish(outcome, sourceHost, verb)
                }
                let confirmed = try await lookup(context.destClient, context.clientBatchId)
                if case let .outcome(outcome) = context.next(.confirmAfterAmbiguousAdmit, .confirmed(confirmed)) {
                    return finish(outcome, sourceHost, verb)
                }
                guard confirmed != nil else { return }
            }
            finish(await complete(authority, context), sourceHost, verb)
        } catch {
            hosts.report(sourceHost, doing: verb, error)
        }
        await queue.poll(source)
        await queue.poll(destination)
    }

    private func complete(_ authority: QueueAuthority?, _ context: Context) async -> TransferOutcome? {
        guard let authority else {
            return .sent(sourceRemoved: true, message: String(localized: "Sent to \(context.destName)."))
        }
        var removed = true
        do { try await context.sourceClient.completeTransfer(authority) } catch { removed = false }
        guard case let .outcome(final) = context.next(.complete, .completed(removed: removed)) else { return nil }
        return final
    }

    private func lookup(_ client: any MoldBackend, _ id: String) async throws -> BatchStatus? {
        do { return try await client.batchStatus(clientBatchId: id) } catch let error as MoldClientError
            where TransferPlan.isNotFound(error) { return nil }
    }

    private func finish(_ outcome: TransferOutcome?, _ source: MoldHost, _ verb: String) {
        switch outcome {
        case let .sent(_, message): queue.summary = message
        case let .refused(message): hosts.report(source, doing: verb, Refusal(message))
        case nil: break
        }
    }

    private struct Refusal: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }

    private struct Context {
        let sourceClient: any MoldBackend
        let destClient: any MoldBackend
        let sourceName: String
        let destName: String
        let expectedDest: String
        let clientBatchId: String

        func next(_ step: TransferPlan.Step, _ outcome: TransferPlan.Outcome) -> TransferPlan.Result {
            TransferPlan.next(after: step, outcome: outcome, clientBatchId: clientBatchId,
                              destinationInstance: expectedDest, destinationLabel: destName, sourceLabel: sourceName)
        }
    }
}
