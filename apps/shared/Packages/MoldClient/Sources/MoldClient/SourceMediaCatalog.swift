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
        public var id: String { "\(owner.hostID)/\(owner.print.filename)/\(member.memberId)" }
        public let authority: SourceMediaAuthority
        public init(owner: LibraryEntry, member: RetainedSourceMedia.Member, route: MoldHost?, instance: String?) {
            self.owner = owner; self.member = member; self.route = route; self.instance = instance
            authority = .init(owner: owner, route: route, instance: instance)
        }
    }
    public struct Source: Identifiable, Sendable {
        public let id: String
        public let usages: [Row]
        public let searchKey: String
        public var roles: String { Array(Set(usages.map { $0.member.sourceLabel })).sorted().joined(separator: " · ") }
    }
    public private(set) var revision = 0
    @ObservationIgnored private var contentEpoch = 0
    private var hostContentRevisions: [UUID: Int] = [:]
    private var ownerContentRevisions: [PrintID: Int] = [:]
    public func contentRevision(for owner: PrintID) -> Int {
        max(hostContentRevisions[owner.host, default: 0], ownerContentRevisions[owner, default: 0])
    }
    @ObservationIgnored private var activeAuthorities: Set<SourceMediaAuthority> = []
    public private(set) var rows: [Row] = []
    public private(set) var sources: [Source] = []
    public private(set) var failures: [PrintID: String] = [:]
    public private(set) var loading = false
    public private(set) var completed = 0
    public private(set) var total = 0
    @ObservationIgnored private var generation = UUID()
    @ObservationIgnored private var cached: [PrintID: Cached] = [:]
    @ObservationIgnored private var presentation: [PrintID: LibraryEntry] = [:]
    @ObservationIgnored private var staleDisplay: [PrintID: [Row]] = [:]
    @ObservationIgnored private var detailCached: [SourceMediaAuthority: Cached] = [:]
    private struct Cached {
        let authority: SourceMediaAuthority
        let inventory: RetainedSourceMedia.Inventory
    }
    public init() {}

    /// A captured gallery row loses access when its exact inventory is revoked.
    public func contains(_ row: Row) -> Bool {
        _ = contentRevision(for: row.owner.id)
        // Observe membership pruning too (another window can change browser scope).
        _ = rows.count
        guard let hit = cached[row.owner.id], hit.authority == row.authority,
              hit.inventory.availability == .available else { return false }
        return hit.inventory.members.contains(row.member)
    }

    private func revoke(_ owner: PrintID) {
        cached[owner] = nil
        staleDisplay[owner] = nil
        detailCached = detailCached.filter { $0.key.owner != owner }
        contentEpoch &+= 1
        ownerContentRevisions[owner] = contentEpoch
    }

    /// Revalidation can keep immutable, same-authority pixels on screen while
    /// captured members lose permission to perform original/reuse operations.
    public func invalidate(host: UUID, filename: String? = nil, preservingDisplay: Bool = false) {
        let affected: (PrintID) -> Bool = { $0.host == host && (filename == nil || $0.filename == filename) }
        invalidateCache(affected, preservingDisplay: preservingDisplay)
        contentEpoch &+= 1
        if let filename { ownerContentRevisions[.init(host: host, filename: filename)] = contentEpoch }
        else {
            hostContentRevisions[host] = contentEpoch
            ownerContentRevisions = ownerContentRevisions.filter { $0.key.host != host }
        }
        revision &+= 1
    }

    /// Explicit Refresh uses this after its pixel-cache barrier. One published
    /// epoch restarts all scoped tiles, including a previously cancelled load.
    public func invalidate(owners: Set<PrintID>, preservingDisplay: Bool = true) {
        guard !owners.isEmpty else { return }
        invalidateCache({ owners.contains($0) }, preservingDisplay: preservingDisplay)
        contentEpoch &+= 1
        for owner in owners { ownerContentRevisions[owner] = contentEpoch }
        revision &+= 1
    }

    private func invalidateCache(_ affected: (PrintID) -> Bool, preservingDisplay: Bool) {
        // An event outside a filtered scan must not silently abort that scan:
        // its scoped task key correctly does not change for unrelated owners.
        if activeAuthorities.contains(where: { affected($0.owner) }) {
            generation = UUID()
            if loading { loading = false }
        }
        if preservingDisplay {
            for (owner, members) in Dictionary(grouping: rows.filter { affected($0.owner.id) }, by: { $0.owner.id }) {
                staleDisplay[owner] = members
            }
        } else {
            staleDisplay = staleDisplay.filter { !affected($0.key) }
            let kept = rows.filter { !affected($0.owner.id) }
            if kept != rows { rows = kept; sources = Self.group(kept) }
        }
        cached = cached.filter { !affected($0.key) }
        detailCached = detailCached.filter { !affected($0.key.owner) }
        failures = failures.filter { !affected($0.key) }
    }

    public func retain(authorities: Set<SourceMediaAuthority>) {
        if !activeAuthorities.isSubset(of: authorities) { generation = UUID(); loading = false }
        activeAuthorities.formIntersection(authorities)
        cached = cached.filter { authorities.contains($0.value.authority) }
        let ownerIDs = Set(authorities.map(\.owner))
        presentation = presentation.filter { ownerIDs.contains($0.key) }
        staleDisplay = staleDisplay.filter { $0.value.allSatisfy { authorities.contains($0.authority) } }
        let kept = rows.filter { authorities.contains($0.authority) }
        if kept.count != rows.count { rows = kept; sources = Self.group(kept) }
    }

    public func load(owners: [LibraryEntry], backend: (UUID) -> (any MoldBackend)?,
                     route: (UUID) -> MoldHost? = { _ in nil },
                     instance: (UUID) -> String? = { _ in nil },
                     refresh: Bool = false, now: Date = Date()) async {
        let token = UUID(); generation = token
        let allowed = Dictionary(owners.map { ($0.id, SourceMediaAuthority(owner: $0, route: route($0.hostID), instance: instance($0.hostID))) }, uniquingKeysWith: { first, _ in first })
        activeAuthorities = Set(allowed.values)
        for owner in owners { presentation[owner.id] = owner }
        // A collection/machine filter changes presentation, not authorization.
        // retain(authorities:) prunes removed routes/owners independently.
        cached = cached.filter { allowed[$0.key] == nil || allowed[$0.key] == $0.value.authority }
        // Detail and browser visits share the same authority-bound answer.
        for owner in owners where cached[owner.id] == nil {
            if let authority = allowed[owner.id], let hit = detailCached[authority] { cached[owner.id] = hit }
        }
        let pending = owners.filter { refresh || cached[$0.id] == nil }
        failures = Dictionary(uniqueKeysWithValues: owners.compactMap { owner in
            guard let hit = cached[owner.id], let message = Self.disclosure(hit.inventory, owner: owner) else { return nil }
            return (owner.id, message)
        })
        total = owners.count
        completed = owners.count - pending.count
        publish(owners: owners)
        // A presentation change or an idle clock never becomes a loading
        // transition. Only an actual read can show scan progress.
        guard !pending.isEmpty else {
            if loading { loading = false }
            return
        }
        loading = true
        defer { if generation == token { loading = false } }
        // Resolve routes before dispatch. Tasks carry only immutable, Sendable request data.
        let batches = Self.batches(pending)
        let requests = batches.map { entries in
            let host = entries[0].hostID
            let capturedRoute = route(host), capturedInstance = instance(host)
            let valid = entries.allSatisfy { allowed[$0.id] == SourceMediaAuthority(owner: $0, route: capturedRoute, instance: capturedInstance) }
            return Request(entries: entries, client: valid ? backend(host) : nil,
                           route: capturedRoute, instance: capturedInstance)
        }
        var resolved = owners.filter { cached[$0.id] != nil }
        var publishedCount = resolved.count
        await withTaskGroup(of: (Int, Result<RetainedSourceMedia.Inventories, Error>).self) { group in
            var next = 0
            func enqueue() {
                guard next < requests.count else { return }
                let index = next, request = requests[next]; next += 1
                let host = request.entries[0].hostID
                let current = route(host) == request.route && instance(host) == request.instance
                group.addTask {
                    do {
                        try Task.checkCancellation()
                        guard current else { throw MoldClientError.unreachable("The machine changed while reading sources. Retry to refresh its inventory.") }
                        guard let client = request.client else { throw MoldClientError.unreachable("Connect this machine to read its source media.") }
                        return (index, .success(try await client.retainedSourceMediaInventories(filenames: request.entries.map { $0.print.filename })))
                    } catch { return (index, .failure(error)) }
                }
            }
            enqueue(); enqueue()
            while let (index, response) = await group.next() {
                guard !Task.isCancelled, generation == token else { group.cancelAll(); return }
                let request = requests[index], entries = request.entries, host = request.entries[0].hostID
                do {
                    guard entries.allSatisfy({ allowed[$0.id] == SourceMediaAuthority(owner: $0, route: request.route, instance: request.instance) }),
                          route(host) == request.route, instance(host) == request.instance else {
                        throw MoldClientError.unreachable("The machine changed while reading sources. Retry to refresh its inventory.")
                    }
                    let result = try response.get()
                    guard result.instanceId == nil || request.instance == nil || result.instanceId == request.instance else {
                        throw MoldClientError.unreachable("The machine changed while reading sources. Retry to refresh its inventory.")
                    }
                    for owner in entries {
                        guard let inventory = result.inventories[owner.print.filename] else {
                            revoke(owner.id)
                            failures[owner.id] = "The machine did not return this print's source inventory."; continue
                        }
                        if let previous = cached[owner.id], previous.inventory != inventory {
                            contentEpoch &+= 1; ownerContentRevisions[owner.id] = contentEpoch
                        }
                        cached[owner.id] = Cached(authority: .init(owner: owner, route: request.route, instance: request.instance), inventory: inventory)
                        staleDisplay[owner.id] = nil
                        failures[owner.id] = Self.disclosure(inventory, owner: owner)
                    }
                } catch {
                    for owner in entries {
                        revoke(owner.id)
                        failures[owner.id] = error.localizedDescription
                    }
                }
                completed += entries.count
                resolved.append(contentsOf: entries)
                // Geometric publication bounds repeated grouping work; progress stays live.
                if entries.contains(where: { failures[$0.id] != nil }) || resolved.count >= max(1, publishedCount * 2) || completed == total {
                    publish(owners: owners)
                    publishedCount = resolved.count
                }
                enqueue()
            }
        }
    }

    private struct Request: Sendable {
        let entries: [LibraryEntry]
        let client: (any MoldBackend)?
        let route: MoldHost?
        let instance: String?
    }

    /// Each host gets a small first-paint request before any bulk requests.
    static func batches(_ owners: [LibraryEntry]) -> [[LibraryEntry]] {
        var order: [UUID] = [], hosts: [UUID: [LibraryEntry]] = [:]
        for owner in owners {
            if hosts[owner.hostID] == nil { order.append(owner.hostID) }
            hosts[owner.hostID, default: []].append(owner)
        }
        var initial: [[LibraryEntry]] = [], bulk: [[LibraryEntry]] = []
        for host in order {
            let entries = hosts[host]!
            initial.append(Array(entries.prefix(4)))
            var start = min(4, entries.count)
            while start < entries.count {
                let end = min(start + 100, entries.count)
                bulk.append(Array(entries[start..<end])); start = end
            }
        }
        return initial + bulk
    }

    /// Details and browser share fresh answers; this does not change the browser scope.
    public func inventory(for owner: LibraryEntry, backend: any MoldBackend, route: MoldHost,
                          instance: String?, refresh: Bool = false) async throws -> RetainedSourceMedia.Inventory {
        let authority = SourceMediaAuthority(owner: owner, route: route, instance: instance)
        if !refresh, let hit = cached[owner.id] ?? detailCached[authority], hit.authority == authority {
            return hit.inventory
        }
        let revision = contentRevision(for: owner.id)
        let result = try await backend.retainedSourceMedia(for: owner.print.filename)
        try Task.checkCancellation()
        // Cache detail answers without publishing them into another browser's
        // scope. A relevant invalidation fences a late negative answer too.
        if revision == contentRevision(for: owner.id) {
            if detailCached.count >= 512 { detailCached.removeAll(keepingCapacity: true) }
            detailCached[authority] = Cached(authority: authority, inventory: result)
        }
        return result
    }

    private static func disclosure(_ inventory: RetainedSourceMedia.Inventory, owner: LibraryEntry) -> String? {
        guard inventory.availability != .available,
              inventory.availability != .unavailableLegacy || RetainedSourceMedia.disclosable(owner.print.metadata) else { return nil }
        return RetainedSourceMedia.disclosure(inventory.availability) ?? "Source media unavailable."
    }

    /// Display-only edits update captions/search/order without restarting reads
    /// or replacing a stable tile's content authority.
    public func updatePresentation(owners: [LibraryEntry]) {
        for owner in owners { presentation[owner.id] = owner }
        publish(owners: owners)
    }

    private func publish(owners: [LibraryEntry]) {
        let updated = owners.flatMap { captured -> [Row] in
            // A network read holds an older owner snapshot. Display edits do
            // not cancel that read, and must not be overwritten on completion.
            let owner = presentation[captured.id] ?? captured
            if let hit = cached[owner.id] {
                guard hit.inventory.availability == .available,
                      hit.authority == SourceMediaAuthority(owner: owner, route: hit.authority.route, instance: hit.authority.instance) else { return [] }
                return hit.inventory.members.map { Row(owner: owner, member: $0, route: hit.authority.route, instance: hit.authority.instance) }
            }
            return (staleDisplay[owner.id] ?? []).compactMap { row in
                guard row.authority == SourceMediaAuthority(owner: owner, route: row.route, instance: row.instance) else { return nil }
                return Row(owner: owner, member: row.member, route: row.route, instance: row.instance)
            }
        }
        guard updated != rows else { return }
        rows = updated
        sources = Self.group(updated)
    }

    public nonisolated static func group(_ rows: [Row]) -> [Source] {
        var order: [String] = [], groups: [String: [Row]] = [:]
        for row in rows {
            let digest = row.member.contentSha256
            let valid = digest?.count == 64 && digest!.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
            let key = valid ? "sha256/\(digest!)" : "member/\(row.id)"
            if groups[key] == nil { order.append(key) }
            groups[key, default: []].append(row)
        }
        return order.map { key in
            let usages = groups[key]!
            let search = usages.map { "\($0.member.sourceLabel) \($0.member.displayName) \($0.owner.print.displayName) \($0.owner.hostName)" }.joined(separator: " ")
            return Source(id: key, usages: usages, searchKey: search.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current))
        }
    }
}

/// Cache authority excludes display-only title, favorite, collection and host name changes.
public struct SourceMediaAuthority: Hashable, Sendable {
    public let owner: PrintID
    public let route: MoldHost?
    public let instance: String?
    private let version: String?
    private let timestamp: UInt64
    private let size: Int?
    private let metadata: OutputMetadata
    /// A repair can change retained members without changing output bytes.
    /// Therefore an identical row is conservative; only actual organization
    /// edits with unchanged content fields can bypass source invalidation.
    public static func isDisplayOnlyUpdate(previous: GalleryPrint, updated: GalleryPrint) -> Bool {
        let presentationChanged = previous.title != updated.title || previous.tags != updated.tags
            || previous.favorite != updated.favorite || previous.collections != updated.collections
        return presentationChanged && previous.filename == updated.filename
            && previous.mediaVersion == updated.mediaVersion && previous.timestamp == updated.timestamp
            && previous.sizeBytes == updated.sizeBytes && previous.metadata == updated.metadata
            && previous.format == updated.format && previous.assets == updated.assets
            && previous.metadataSynthetic == updated.metadataSynthetic
            && previous.trashedAt == updated.trashedAt && previous.purgeAt == updated.purgeAt
    }

    public init(owner: LibraryEntry, route: MoldHost?, instance: String?) {
        self.owner = owner.id
        var normalized = route; normalized?.name = ""
        self.route = normalized; self.instance = instance
        version = owner.print.mediaVersion; timestamp = owner.print.timestamp
        size = owner.print.sizeBytes; metadata = owner.print.metadata
    }
}
