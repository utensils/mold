import Foundation
import MoldClient

/// Everything after `TransferStore.transfer`'s identity check -- split out
/// purely for size. Not `private`: `transfer` calls `afterIdentities` from
/// the other file, and `private` does not cross a file boundary.
extension TransferStore {
    /// `checkPriorAttempt` onward: the destination lookup, the source's own
    /// live authority (fact 15 -- read now, never the cached row), and what
    /// the plan decides from their combination.
    func afterIdentities(
        _ entry: QueueEntry, authorityInstance: String, context: TransferContext
    ) async -> TransferOutcome? {
        let priorBatch: BatchStatus?
        do { priorBatch = try await lookup(context.destClient, context.clientBatchId) } catch {
            hosts.report(error, on: context.source, doing: context.verb); return nil
        }
        let authority: QueueAuthority?
        do {
            let detail = try await context.sourceClient.queueJob(id: entry.id)
            authority = detail.job.state == .held ? detail.job.authority(instanceId: authorityInstance) : nil
        } catch let error as MoldClientError where priorBatch != nil && TransferPlan.isNotFound(error) {
            authority = nil
        } catch {
            hosts.report(error, on: context.source, doing: context.verb); return nil
        }

        switch context.next(.checkPriorAttempt, .priorAttempt(priorBatch, sourceHeld: authority != nil)) {
        case let .outcome(outcome): return report(outcome, context)
        case .step(.complete): return await complete(authority: authority, context: context)
        case .step(.export): break
        default: return nil
        }
        guard let authority else { return nil }
        return await exportAndAdmit(authority, context: context)
    }

    func exportAndAdmit(_ authority: QueueAuthority, context: TransferContext) async -> TransferOutcome? {
        let portable: Data
        do { portable = try await context.sourceClient.exportHeldJob(authority) } catch {
            hosts.report(error, on: context.source, doing: context.verb); return nil
        }
        let readyToComplete: Bool
        do {
            let admitted = try await context.destClient.admitTransfer(
                clientBatchId: context.clientBatchId, portable: portable, destinationInstance: context.expectedDest)
            switch context.next(.admit, .admitResult(.landed(admitted))) {
            case let .outcome(outcome): return report(outcome, context)
            case .step: readyToComplete = true
            }
        } catch {
            switch await confirmAfterAmbiguousAdmit(error, context: context) {
            case let .outcome(outcome): return report(outcome, context)
            case .readyToComplete: readyToComplete = true
            case nil: return nil
            }
        }
        guard readyToComplete else { return nil }
        return await complete(authority: authority, context: context)
    }

    enum Confirmation { case outcome(TransferOutcome), readyToComplete }

    /// After `admit` fails ambiguously, looks the destination up again --
    /// nothing found is a hard refusal here, never a second export/admit
    /// attempt (`checkPriorAttempt`'s "not found" means something else).
    func confirmAfterAmbiguousAdmit(_ error: Error, context: TransferContext) async -> Confirmation? {
        switch context.next(.admit, .admitResult(TransferPlan.classifyAdmitFailure(error))) {
        case let .outcome(outcome): return .outcome(outcome)
        case .step(.confirmAfterAmbiguousAdmit): break
        default: return nil
        }
        let confirmed: BatchStatus?
        do { confirmed = try await lookup(context.destClient, context.clientBatchId) } catch {
            hosts.report(error, on: context.source, doing: context.verb); return nil
        }
        switch context.next(.confirmAfterAmbiguousAdmit, .confirmed(confirmed)) {
        case let .outcome(outcome): return .outcome(outcome)
        case .step: return confirmed == nil ? nil : .readyToComplete
        }
    }

    func complete(authority: QueueAuthority?, context: TransferContext) async -> TransferOutcome? {
        // A `nil` authority means the source was already gone when checked --
        // an earlier attempt's `complete` already ran.
        guard let authority else {
            return report(.sent(
                sourceRemoved: true,
                message: "Sent to \(context.destName). The original was already removed from \(context.sourceName)'s queue."),
                context)
        }
        var removed = true
        do { try await context.sourceClient.completeTransfer(authority) } catch { removed = false }
        let outcome = context.next(.complete, .completed(removed: removed))
        await queue.poll(context.source)
        guard case let .outcome(final) = outcome else { return nil }
        return report(final, context)
    }

    /// `nil` for a definite 404 ("nothing landed yet"); anything else is a
    /// real failure of the whole transfer, never "proceed as if empty."
    func lookup(_ client: any MoldBackend, _ clientBatchId: String) async throws -> BatchStatus? {
        do {
            return try await client.batchStatus(clientBatchId: clientBatchId)
        } catch let error as MoldClientError where TransferPlan.isNotFound(error) {
            return nil
        }
    }

    /// `.refused` is reported here and needs nothing more from the caller;
    /// `.sent` is only handed back -- `transfer` is the one place that
    /// captions it.
    func report(_ outcome: TransferOutcome, _ context: TransferContext) -> TransferOutcome {
        if case let .refused(message) = outcome {
            hosts.report(TransferRefusal(message), on: context.source, doing: context.verb)
        }
        return outcome
    }
}
