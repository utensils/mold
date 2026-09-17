import Foundation
import MoldClient

// What one row asks its machine for. Split from `QueueStore.swift` past the
// file-size advisory -- that file now holds the state and the one listing
// READ, and this one every WRITE to a single row.
//
// `NoInstanceKnown`, `NotADurableBatchChild` and the fixture's own
// `FixtureRefusal` live in `QueueStore+Fixture.swift`.
@MainActor
extension QueueStore {

    func cancel(_ entry: QueueEntry, on host: MoldHost.ID) async {
        await act(entry, on: host, doing: "cancel that job") { try await $0.cancelJob(id: entry.id) }
    }

    func pause(_ entry: QueueEntry, on host: MoldHost.ID) async {
        await act(entry, on: host, doing: "pause that job") { try await $0.pauseJob(id: entry.id) }
    }

    func resume(_ entry: QueueEntry, on host: MoldHost.ID) async {
        await act(entry, on: host, doing: "resume that job") { try await $0.resumeJob(id: entry.id) }
    }

    /// Only meaningful for a held job that is a durable batch child, and only
    /// with the host's instance id -- retrying against a host that has
    /// restarted would aim at nothing. The identity comes from `HostStore`,
    /// which already holds it, so this store need not keep a second copy.
    func retry(_ entry: QueueEntry, on host: MoldHost.ID) async {
        guard !refuseIfFixture(host, doing: "retry that job") else { return }
        guard let instance = hosts.instanceID(of: host) else {
            hosts.report(NoInstanceKnown(), on: host, doing: "retry that job")
            return
        }
        guard let authority = entry.authority(instanceId: instance) else {
            hosts.report(NotADurableBatchChild(), on: host, doing: "retry that job")
            return
        }
        await act(entry, on: host, doing: "retry that job") {
            try await $0.retryJob(authority)
        }
    }

    private func act(_ entry: QueueEntry, on host: MoldHost.ID, doing verb: String,
                     _ body: (any MoldBackend) async throws -> Void) async {
        guard !refuseIfFixture(host, doing: verb) else { return }
        guard let client = hosts.backend(for: host) else { return }
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
