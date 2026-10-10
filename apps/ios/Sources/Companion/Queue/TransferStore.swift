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
                && (hosts.capabilities[source]?.queue?.transferIdentity == nil
                    || hosts.capabilities[host.id]?.queue?.transferIdentity
                        != hosts.capabilities[source]?.queue?.transferIdentity)
        }
    }

    func transfer(_ entry: QueueEntry, from source: MoldHost.ID, to destination: MoldHost.ID) async {
        guard transferring == nil, let sourceHost = hosts.host(source), let destHost = hosts.host(destination),
              let expectedSource = hosts.instanceID(of: source), let expectedDest = hosts.instanceID(of: destination)
        else { return }
        let verb = String(localized: "send that job to \(destHost.name)")
        transferring = entry.id
        defer { transferring = nil }
        var context = Context(
            sourceClient: hosts.backend(for: sourceHost), destClient: hosts.backend(for: destHost),
            sourceName: sourceHost.name, destName: destHost.name, expectedDest: expectedDest,
            destinationTransferIdentity: hosts.capabilities[destination]?.queue?.transferIdentity
                ?? expectedDest,
            clientBatchId: QueueTransferID.derive(source: expectedSource, jobId: entry.id,
                                                                   destination: expectedDest))
        let reservedProtocol = hosts.capabilities[source]?.queue?.preRenderTransfer == true
        if reservedProtocol
            && (hosts.capabilities[destination]?.queue?.transferIdentity?.isEmpty != false)
        {
            return finish(
                .refused("Update or refresh destination; durable transfer identity unavailable."),
                sourceHost, verb)
        }
        if reservedProtocol && hosts.capabilities[destination]?.queue?.preRenderTransfer != true {
            return finish(
                .refused(
                    "Update the destination server before moving waiting jobs; it must support safe transfer recovery."
                ), sourceHost, verb)
        }
        guard queue.beginTransfer(entry, on: source) else { return }
        defer { queue.endTransfer(entry, on: source) }
        do {
            let sourceStatus = try await context.sourceClient.status()
            let destStatus = try await context.destClient.status()
            guard case .step(.checkPriorAttempt) = context.next(.verifyIdentities, .identities(
                sourceChanged: sourceStatus.instanceId != expectedSource,
                destinationChanged: destStatus.instanceId != expectedDest))
            else { return finish(.refused(String(localized: "A machine's identity changed. Refresh the machines and try again.")), sourceHost, verb) }
            if reservedProtocol,
                let reservation = try await context.sourceClient.transferReservation(id: entry.id)
            {
                guard reservation.destinationTransferIdentity == context.destinationTransferIdentity
                else {
                    return finish(
                        .refused(
                            "The original is reserved for another machine. Retry that destination to reconcile acceptance."
                        ), sourceHost, verb)
                }
                context.clientBatchId = reservation.transferId
            } else if reservedProtocol {
                context.clientBatchId = UUID().uuidString.lowercased()
            }
            let prior = try await lookup(context.destClient, context.clientBatchId)
            let detail = try? await context.sourceClient.queueJob(id: entry.id)
            let authority =
                detail.map {
                    QueueTransferEligibility.allows(
                        $0.job.state, reservedProtocol: reservedProtocol)
                        ? $0.job.transferAuthority(
                            instanceId: expectedSource, reservedProtocol: reservedProtocol) : nil
                } ?? nil
            if reservedProtocol, let prior, let authority,
                let recovered = await restoreTerminalFailure(
                    prior, authority: authority, context: context)
            {
                return finish(recovered, sourceHost, verb)
            }
            switch context.next(.checkPriorAttempt, .priorAttempt(prior, sourceHeld: authority != nil)) {
            case let .outcome(outcome): return finish(outcome, sourceHost, verb)
            case .step(.complete): return finish(await complete(authority, context), sourceHost, verb)
            case .step(.export): break
            default: return
            }
            guard var authority else { return }
            if reservedProtocol {
                authority.transferId = context.clientBatchId
                authority.destinationTransferIdentity = context.destinationTransferIdentity
            }
            var reservation = QueueTransferReservationRequest(
                authority: authority, transferId: context.clientBatchId,
                destinationTransferIdentity: context.destinationTransferIdentity)
            if reservedProtocol { try await context.sourceClient.reserveTransfer(reservation) }
            let portable: Data
            do { portable = try await context.sourceClient.exportHeldJob(authority) } catch {
                if reservedProtocol { try? await context.sourceClient.releaseTransfer(reservation) }
                throw error
            }
            do {
                if reservedProtocol { try await context.sourceClient.sealTransfer(reservation) }
                let admitted = try await context.destClient.admitTransfer(
                    clientBatchId: context.clientBatchId, portable: portable, destinationInstance: expectedDest)
                if reservedProtocol,
                    let recovered = await restoreTerminalFailure(
                        admitted, authority: authority, context: context)
                {
                    return finish(recovered, sourceHost, verb)
                }
                if case let .outcome(outcome) = context.next(.admit, .admitResult(.landed(admitted))) {
                    return finish(outcome, sourceHost, verb)
                }
            } catch {
                if reservedProtocol {
                    let aborted = try await context.destClient.abortDestinationTransfer(
                        .init(
                            transferId: context.clientBatchId,
                            destinationTransferIdentity: context.destinationTransferIdentity))
                    guard aborted.transferId == context.clientBatchId,
                        aborted.destinationTransferIdentity == context.destinationTransferIdentity
                    else { throw MoldClientError.malformedResponse }
                    if let receipt = aborted.abortReceipt {
                        reservation.abortReceipt = receipt
                        try await context.sourceClient.releaseTransfer(reservation)
                        return finish(
                            .refused(
                                "\(error.localizedDescription) The original was restored on \(sourceHost.name)."
                            ), sourceHost, verb)
                    } else {
                        guard
                            let accepted = try await lookup(
                                context.destClient, context.clientBatchId),
                            accepted.instanceId == expectedDest
                        else { throw MoldClientError.malformedResponse }
                        if let recovered = await restoreTerminalFailure(
                            accepted, authority: authority, context: context)
                        {
                            return finish(recovered, sourceHost, verb)
                        }
                        switch context.next(.admit, .admitResult(.landed(accepted))) {
                        case let .outcome(outcome): return finish(outcome, sourceHost, verb)
                        case .step:
                            return finish(await complete(authority, context), sourceHost, verb)
                        }
                    }
                }
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
            if reservedProtocol {
                queue.summary =
                    "Retry Move to \(destHost.name) to reconcile acceptance. An unresolved original stays reserved."
            }
            hosts.report(sourceHost, doing: verb, error)
        }
        await queue.poll(source)
        await queue.poll(destination)
    }

    private func restoreTerminalFailure(
        _ batch: BatchStatus, authority: QueueAuthority, context: Context
    ) async -> TransferOutcome? {
        guard batch.instanceId == context.expectedDest,
            batch.clientBatchId == context.clientBatchId, batch.durable == true,
            batch.children.count == 1,
            [.failed, .cancelled].contains(batch.children[0].state)
        else { return nil }
        do {
            let aborted = try await context.destClient.abortDestinationTransfer(
                .init(
                    transferId: context.clientBatchId,
                    destinationTransferIdentity: context.destinationTransferIdentity))
            guard aborted.transferId == context.clientBatchId,
                aborted.destinationTransferIdentity == context.destinationTransferIdentity,
                let receipt = aborted.abortReceipt
            else { throw MoldClientError.malformedResponse }
            var request = QueueTransferReservationRequest(
                authority: authority, transferId: context.clientBatchId,
                destinationTransferIdentity: context.destinationTransferIdentity)
            request.abortReceipt = receipt
            try await context.sourceClient.releaseTransfer(request)
            return .refused(
                "The destination job failed or was cancelled. The original was restored; choose another machine or retry."
            )
        } catch {
            return .refused(
                "Destination terminal failure could not be reconciled. Retry the same destination; the original remains reserved."
            )
        }
    }

    private func complete(_ authority: QueueAuthority?, _ context: Context) async -> TransferOutcome? {
        guard var authority else {
            return .sent(sourceRemoved: true, message: String(localized: "Sent to \(context.destName)."))
        }
        authority.transferId = context.clientBatchId
        authority.destinationTransferIdentity = context.destinationTransferIdentity
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
        let destinationTransferIdentity: String
        var clientBatchId: String

        func next(_ step: TransferPlan.Step, _ outcome: TransferPlan.Outcome) -> TransferPlan.Result {
            TransferPlan.next(after: step, outcome: outcome, clientBatchId: clientBatchId,
                              destinationInstance: expectedDest, destinationLabel: destName, sourceLabel: sourceName)
        }
    }
}
