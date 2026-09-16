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

    init(hosts: HostStore) {
        self.hosts = hosts
    }

    func refresh() async {
        isLoading = true
        defer { isLoading = false }
        await withTaskGroup(of: (MoldHost.ID, Result<QueueListing, Error>, String?).self) { group in
            for host in hosts.hosts {
                let client = hosts.backend(for: host)
                group.addTask {
                    async let listing: Result<QueueListing, Error> = {
                        do { return .success(try await client.queue()) }
                        catch { return .failure(error) }
                    }()
                    async let status = try? await client.status()
                    return (host.id, await listing, (await status)?.instanceId)
                }
            }
            for await (id, result, instance) in group {
                switch result {
                case let .success(listing):
                    byHost[id] = listing.merged
                    hosts.succeeded(on: id, doing: "list its queue")
                case let .failure(error):
                    // A machine that cannot answer keeps the rows it last
                    // showed -- blanking them says something happened to the
                    // jobs, when what happened is a bad connection.
                    hosts.report(error, on: id, doing: "list its queue")
                }
                if let instance { instanceIDs[id] = instance }
            }
        }
    }

    var all: [QueueEntry] { byHost.values.flatMap(\.self) }

    func entries(on host: MoldHost.ID) -> [QueueEntry] { byHost[host] ?? [] }

    // MARK: - Actions

    func cancel(_ entry: QueueEntry, on host: MoldHost.ID) async {
        await act(entry, on: host, doing: "cancel that job") { try await $0.cancelJob(id: entry.id) }
    }

    func pause(_ entry: QueueEntry, on host: MoldHost.ID) async {
        await act(entry, on: host, doing: "pause that job") { try await $0.pauseJob(id: entry.id) }
    }

    func resume(_ entry: QueueEntry, on host: MoldHost.ID) async {
        await act(entry, on: host, doing: "resume that job") { try await $0.resumeJob(id: entry.id) }
    }

    /// Only meaningful for a held job, and only with the host's instance id --
    /// retrying against a host that has restarted would aim at nothing.
    func retry(_ entry: QueueEntry, on host: MoldHost.ID) async {
        guard let instance = instanceIDs[host] else {
            hosts.report(NoInstanceKnown(), on: host, doing: "retry that job")
            return
        }
        await act(entry, on: host, doing: "retry that job") {
            try await $0.retryJob(entry, instanceId: instance)
        }
    }

    private func act(_ entry: QueueEntry, on host: MoldHost.ID, doing verb: String,
                     _ body: (any MoldBackend) async throws -> Void) async {
        guard let client = hosts.backend(for: host) else { return }
        do {
            try await body(client)
            hosts.succeeded(on: host)
        } catch {
            hosts.report(error, on: host, doing: verb)
        }
    }
}

/// Not a network failure -- the host just hasn't told us which run it is yet.
/// `report` still names the machine, because "refresh and try again" is what
/// the person needs whichever produced the sentence.
private struct NoInstanceKnown: LocalizedError {
    var errorDescription: String? {
        "This machine hasn't said which run it is; refresh and try again."
    }
}
