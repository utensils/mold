import Foundation
import MoldClient

/// Queued work across every machine, and the actions on it.
@MainActor
@Observable
final class QueueStore {
    private let hosts: HostStore
    private(set) var byHost: [MoldHost.ID: [QueueEntry]] = [:]
    private(set) var instanceIDs: [MoldHost.ID: String] = [:]
    private(set) var isLoading = false
    private(set) var failure: String?

    init(hosts: HostStore) {
        self.hosts = hosts
    }

    func refresh() async {
        isLoading = true
        defer { isLoading = false }
        await withTaskGroup(of: (MoldHost.ID, [QueueEntry], String?).self) { group in
            for host in hosts.hosts {
                let client = hosts.backend(for: host)
                group.addTask {
                    async let listing = try? await client.queue()
                    async let status = try? await client.status()
                    return (host.id, (await listing)?.merged ?? [], (await status)?.instanceId)
                }
            }
            for await (id, entries, instance) in group {
                byHost[id] = entries
                if let instance { instanceIDs[id] = instance }
            }
        }
    }

    var all: [QueueEntry] { byHost.values.flatMap(\.self) }

    func entries(on host: MoldHost.ID) -> [QueueEntry] { byHost[host] ?? [] }

    // MARK: - Actions

    func cancel(_ entry: QueueEntry, on host: MoldHost.ID) async {
        await act(entry, on: host) { try await $0.cancelJob(id: entry.id) }
    }

    func pause(_ entry: QueueEntry, on host: MoldHost.ID) async {
        await act(entry, on: host) { try await $0.pauseJob(id: entry.id) }
    }

    func resume(_ entry: QueueEntry, on host: MoldHost.ID) async {
        await act(entry, on: host) { try await $0.resumeJob(id: entry.id) }
    }

    /// Only meaningful for a held job, and only with the host's instance id --
    /// retrying against a host that has restarted would aim at nothing.
    func retry(_ entry: QueueEntry, on host: MoldHost.ID) async {
        guard let instance = instanceIDs[host] else {
            failure = "This machine hasn't said which run it is; refresh and try again."
            return
        }
        await act(entry, on: host) { try await $0.retryJob(entry, instanceId: instance) }
    }

    private func act(_ entry: QueueEntry, on host: MoldHost.ID,
                     _ body: (any MoldBackend) async throws -> Void) async {
        guard let client = hosts.backend(for: host) else { return }
        do {
            try await body(client)
            failure = nil
        } catch {
            failure = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }
}
