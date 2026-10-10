import Foundation
import MoldClient

// What one row asks its machine for. Split from `QueueStore.swift` past the
// file-size advisory -- that file now holds the state and the one listing
// READ, and this one every WRITE to a single row.
//
// Fixture refusal remains owned by `QueueStore+Fixture.swift`; stale actions
// are refused before sending a mutation.
@MainActor
extension QueueStore {

    func isActing(_ entry: QueueEntry, on host: MoldHost.ID) -> Bool {
        acting[host]?.contains(entry.id) == true
    }

    /// All rendered actions use the same current-host and current-job authority.
    func actions(for entry: QueueEntry, on host: MoldHost.ID) -> QueueRowActions {
        guard let machine = hosts.host(host), hosts.isUp(machine), !isActing(entry, on: host),
              let current = entries(on: host).first(where: { $0.id == entry.id }),
              current.state == entry.state else { return QueueRowActions() }
        var result = QueueRowActions.resolve(current, on: hosts.capabilities[host])
        result.retry = result.retry && hosts.instanceID(of: host).map { current.authority(instanceId: $0) != nil } == true
        if case .prose(_, retryable: false) = hold(for: current, on: host) { result.retry = false }
        return result
    }

    func actions(for entries: [QueueEntry], on host: MoldHost.ID) -> QueueRowActions {
        entries.reduce(into: QueueRowActions()) { result, entry in
            let row = actions(for: entry, on: host)
            result.pause = result.pause || row.pause
            result.resume = result.resume || row.resume
            result.retry = result.retry || row.retry
            result.cancel = result.cancel || row.cancel
        }
    }

    func canTransfer(_ entry: QueueEntry, on host: MoldHost.ID) -> Bool {
        guard let machine = hosts.host(host), hosts.isUp(machine), !isActing(entry, on: host),
              let current = entries(on: host).first(where: { $0.id == entry.id }), current.state == entry.state,
              QueueTransferEligibility.allows(current.state, reservedProtocol: hosts.capabilities[host]?.queue?.preRenderTransfer == true), let instance = hosts.instanceID(of: host) else { return false }
        return current.transferAuthority(instanceId: instance, reservedProtocol: hosts.capabilities[host]?.queue?.preRenderTransfer == true) != nil
    }

    func cancel(_ entry: QueueEntry, on host: MoldHost.ID) async {
        guard !refuseIfFixture(host, doing: "cancel that job"), actions(for: entry, on: host).cancel else { return }
        await act(entry, on: host, doing: "cancel that job") { client in
            if entry.state == .held { _ = try await client.cancelHeldJob(id: entry.id) }
            else { try await client.cancelJob(id: entry.id) }
        }
    }

    func pause(_ entry: QueueEntry, on host: MoldHost.ID) async {
        guard !refuseIfFixture(host, doing: "pause that job"), actions(for: entry, on: host).pause else { return }
        await act(entry, on: host, doing: "pause that job") { try await $0.pauseJob(id: entry.id) }
    }

    func resume(_ entry: QueueEntry, on host: MoldHost.ID) async {
        guard !refuseIfFixture(host, doing: "resume that job"), actions(for: entry, on: host).resume else { return }
        await act(entry, on: host, doing: "resume that job") { try await $0.resumeJob(id: entry.id) }
    }

    func retry(_ entry: QueueEntry, on host: MoldHost.ID) async {
        guard !refuseIfFixture(host, doing: "retry that job"), actions(for: entry, on: host).retry,
              let current = entries(on: host).first(where: { $0.id == entry.id }),
              current.batchId == entry.batchId, current.clientBatchId == entry.clientBatchId,
              current.batchIndex == entry.batchIndex, let instance = hosts.instanceID(of: host),
              let authority = current.authority(instanceId: instance) else { return }
        await act(entry, on: host, doing: "retry that job") { try await $0.retryJob(authority) }
    }

    private func act(_ entry: QueueEntry, on host: MoldHost.ID, doing verb: String,
                     _ body: (any MoldBackend) async throws -> Void) async {
        guard !refuseIfFixture(host, doing: verb) else { return }
        guard let machine = hosts.host(host), hosts.isUp(machine), !isActing(entry, on: host),
              let client = hosts.backend(for: host) else { return }
        let instance = hosts.instanceID(of: host)
        acting[host, default: []].insert(entry.id)
        defer { acting[host]?.remove(entry.id) }
        guard hosts.host(host) == machine, hosts.instanceID(of: host) == instance,
              entries(on: host).contains(where: { $0.id == entry.id && $0.state == entry.state }) else { return }
        do {
            try await body(client)
            hosts.succeeded(on: host)
        } catch {
            hosts.report(error, on: host, doing: verb)
        }
        // Fallback (d): the row's new position, or that it left the queue
        // entirely, is the server's to state.
        await poll(host)
    }
}
