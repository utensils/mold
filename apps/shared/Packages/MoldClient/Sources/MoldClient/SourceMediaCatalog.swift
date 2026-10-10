import Foundation
import Observation

/// A transient projection of publication-owned members. It owns no retained pins.
@MainActor @Observable
public final class SourceMediaCatalog {
    public struct Row: Identifiable, Hashable, Sendable {
        public let owner: LibraryEntry
        public let member: RetainedSourceMedia.Member
        public let route: MoldHost?
        public let instance: String?
        let publication: UUID
        public var id: String { "\(publication)/\(owner.hostID)/\(owner.print.filename)/\(member.memberId)" }
    }
    public private(set) var rows: [Row] = []
    public private(set) var failures: [PrintID: String] = [:]
    public private(set) var loading = false
    public private(set) var completed = 0
    public private(set) var total = 0
    private var generation = UUID()
    public init() {}

    public func load(owners: [LibraryEntry], backend: (UUID) -> (any MoldBackend)?,
                     route: (UUID) -> MoldHost? = { _ in nil },
                     instance: (UUID) -> String? = { _ in nil }) async {
        let token = UUID(); generation = token
        rows = []; failures = [:]; completed = 0; total = owners.count; loading = true
        defer { if generation == token { loading = false } }
        // Small batches bound metadata work; all pages are scanned, including search.
        for start in stride(from: 0, to: owners.count, by: 25) {
            guard !Task.isCancelled, generation == token else { return }
            let page = Array(owners[start ..< min(start + 25, owners.count)])
            for (host, entries) in Dictionary(grouping: page, by: \.hostID) {
                let capturedRoute = route(host), capturedInstance = instance(host)
                do {
                    guard let client = backend(host) else { throw MoldClientError.unreachable("Connect this machine to read its source media.") }
                    let result = try await client.retainedSourceMediaInventories(filenames: entries.map { $0.print.filename })
                    guard !Task.isCancelled, generation == token else { return }
                    guard route(host) == capturedRoute, instance(host) == capturedInstance,
                          result.instanceId == nil || capturedInstance == nil || result.instanceId == capturedInstance else {
                        throw MoldClientError.unreachable("The machine changed while reading sources. Retry to refresh its inventory.")
                    }
                    for owner in entries {
                        guard let inventory = result.inventories[owner.print.filename] else {
                            failures[owner.id] = "The machine did not return this print's source inventory."; continue
                        }
                        if inventory.availability == .available {
                            rows += inventory.members.map { Row(owner: owner, member: $0, route: capturedRoute, instance: capturedInstance, publication: token) }
                        } else if inventory.availability != .unavailableLegacy || RetainedSourceMedia.disclosable(owner.print.metadata) {
                            failures[owner.id] = RetainedSourceMedia.disclosure(inventory.availability) ?? "Source media unavailable."
                        }
                    }
                } catch {
                    guard !Task.isCancelled, generation == token else { return }
                    for owner in entries { failures[owner.id] = error.localizedDescription }
                }
                completed += entries.count
            }
        }
    }
}
