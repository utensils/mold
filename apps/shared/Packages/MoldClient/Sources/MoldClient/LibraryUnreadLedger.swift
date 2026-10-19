import Foundation

/// Durable, device-local icon state. Read identity follows merged copy groups,
/// including collision-renamed copies and distinct outputs sharing a filename.
public struct LibraryUnreadLedger: Codable, Equatable, Sendable {
    private var known: [UUID: Set<PrintID>] = [:]
    private var unread: Set<PrintID> = []
    private var visibleGroups: [Set<PrintID>] = []
    private var pendingViewed: Set<PrintID>?

    // Derived state is never encoded: old ledgers restore the same wire shape.
    private var positions: [PrintID: Int]?
    private var unreadCount: Int?
    private(set) var observedEntryCount = 0
    private enum CodingKeys: String, CodingKey { case known, unread, visibleGroups, pendingViewed }

    public init() {}
    public var count: Int { unreadCount ?? visibleGroups.filter { !$0.isDisjoint(with: unread) }.count }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.known == rhs.known && lhs.unread == rhs.unread && lhs.pendingViewed == rhs.pendingViewed
            && (lhs.visibleGroups == rhs.visibleGroups
                || Set(lhs.visibleGroups.filter { !$0.isEmpty }) == Set(rhs.visibleGroups.filter { !$0.isEmpty }))
    }

    public mutating func observe(entries: [LibraryEntry], visible: [LibraryEntry], loadedHosts: Set<UUID>, presentHosts: Set<UUID>) {
        retainHosts(presentHosts)
        observeIdentities(entries, loadedHosts: loadedHosts)
        ensurePositions()
        let incoming = visible.map { entry -> Set<PrintID> in
            if let position = positions?[entry.id] {
                let group = visibleGroups[position]
                if group.count == entry.copies.count + 1,
                   entry.copies.allSatisfy({ group.contains($0.id) }) { return group }
            }
            return Set([entry.id] + entry.copies.map(\.id))
        }
        // Saved read state survives cache clearing/unavailable machines. Keep
        // their last visible groups unless a loaded copy already represents it.
        let incomingCopyIDs = incoming.reduce(into: Set<PrintID>()) { $0.formUnion($1) }
        let retained = visibleGroups.compactMap { group -> Set<PrintID>? in
            guard group.isDisjoint(with: incomingCopyIDs) else { return nil }
            let unavailable = group.filter { presentHosts.contains($0.host) && !loadedHosts.contains($0.host) }
            return unavailable.isEmpty ? nil : Set(unavailable)
        }
        visibleGroups = incoming + retained
        positions = nil
        unreadCount = visibleGroups.filter { !$0.isDisjoint(with: unread) }.count
    }

    /// Apply changed logical groups after the caller has preserved merge identity.
    /// Untouched groups retain their visibility; a privacy/host-scope reset must
    /// use full observe instead. No inventory-wide equality/union/scan is needed.
    @discardableResult
    public mutating func observeChanges(entries: [LibraryEntry], visible: [LibraryEntry], removed: [PrintID],
                                       loadedHosts: Set<UUID>, presentHosts: Set<UUID>) -> Bool {
        let pruned = known.keys.contains { !presentHosts.contains($0) }
            || pendingViewed?.contains(where: { !presentHosts.contains($0.host) }) == true
        retainHosts(presentHosts)
        ensurePositions()
        let changedIDs = Set(entries.flatMap { [$0.id] + $0.copies.map(\.id) }).union(removed)
        let affected = Set(changedIDs.compactMap { positions?[$0] })
        let previousGroups = Set(affected.map { visibleGroups[$0] })
        for position in affected {
            let group = visibleGroups[position]
            if !group.isDisjoint(with: unread) { unreadCount? -= 1 }
            for id in group { positions?[id] = nil }
            visibleGroups[position] = []
        }
        let changed = observeIdentities(entries, loadedHosts: loadedHosts)
        var nextGroups = Set<Set<PrintID>>()
        for entry in visible {
            let group = Set([entry.id] + entry.copies.map(\.id))
            let position = visibleGroups.count
            visibleGroups.append(group)
            for id in group { positions?[id] = position }
            if !group.isDisjoint(with: unread) { unreadCount? += 1 }
            nextGroups.insert(group)
        }
        // Keep tombstones bounded without making every removal shift indexes.
        if visibleGroups.count > max(256, (positions?.count ?? 0) * 2) {
            visibleGroups.removeAll { $0.isEmpty }
            positions = nil
        }
        return pruned || changed || previousGroups != nextGroups
    }

    @discardableResult
    private mutating func observeIdentities(_ entries: [LibraryEntry], loadedHosts: Set<UUID>) -> Bool {
        observedEntryCount += entries.count
        let initialized = Set(known.keys)
        var changed = false
        for entry in entries {
            let copies = [entry.id] + entry.copies.map(\.id)
            let familiar = copies.filter { known[$0.host]?.contains($0) == true }
            let pending = pendingViewed.map { pending in copies.contains { pending.contains($0) } } == true
            // Metadata changes do not change read identity. Most observations
            // contain thousands of these unchanged groups and one new copy.
            if !pending, familiar.count == copies.count {
                let firstUnread = unread.contains(copies[0])
                if copies.dropFirst().allSatisfy({ unread.contains($0) == firstUnread }) { continue }
            }
            let read = pendingViewed.map { pending in copies.contains { pending.contains($0) } } == true
                || familiar.contains { !unread.contains($0) }
            if read {
                for id in copies { if unread.remove(id) != nil { changed = true } }
            } else if copies.contains(where: { unread.contains($0) })
                || (familiar.isEmpty && copies.contains { initialized.contains($0.host) }) {
                for id in copies { if unread.insert(id).inserted { changed = true } }
            }
            for copy in copies {
                if known[copy.host, default: []].insert(copy).inserted { changed = true }
                if pendingViewed?.remove(copy) != nil { changed = true }
            }
        }
        for host in loadedHosts where known[host] == nil { known[host] = []; changed = true }
        return changed
    }

    private mutating func ensurePositions() {
        guard positions == nil else { return }
        positions = [:]
        var total = 0
        for (index, group) in visibleGroups.enumerated() {
            for id in group { positions?[id] = index }
            if !group.isDisjoint(with: unread) { total += 1 }
        }
        unreadCount = total
    }

    private mutating func read(_ ids: Set<PrintID>) {
        ensurePositions()
        let affected = Set(ids.compactMap { positions?[$0] })
        let previous = affected.filter { !visibleGroups[$0].isDisjoint(with: unread) }.count
        unread.subtract(ids)
        let current = affected.filter { !visibleGroups[$0].isDisjoint(with: unread) }.count
        unreadCount? -= previous - current
    }

    public func isUnread(_ entry: LibraryEntry) -> Bool { !Set(entry.everyCopy.map(\.id)).isDisjoint(with: unread) }

    /// A successfully displayed generation can precede its gallery listing.
    public mutating func view(_ id: PrintID) {
        if known[id.host]?.contains(id) == true {
            // The viewer can outlive a temporarily unavailable listing. Preserve
            // the last authoritative copy identity instead of leaving a mirror
            // unread when only its physical ID is available to the caller.
            ensurePositions()
            let copies = positions?[id].map { visibleGroups[$0] } ?? [id]
            read(copies)
        }
        else {
            if pendingViewed == nil { pendingViewed = [] }
            pendingViewed?.insert(id)
        }
    }

    public mutating func view(_ entry: LibraryEntry) { read(Set([entry.id] + entry.copies.map(\.id))) }
    public mutating func markSeen(_ entries: [LibraryEntry]) { read(Set(entries.flatMap { [$0.id] + $0.copies.map(\.id) })) }

    public mutating func markSeen(_ ids: Set<PrintID>) { read(ids) }

    public static let key = "library.unreadMedia.v1"
    public static func load(from defaults: UserDefaults) -> Self {
        defaults.data(forKey: key).flatMap { try? MoldJSON.decoder.decode(Self.self, from: $0) } ?? Self()
    }
    public func save(to defaults: UserDefaults) {
        if let data = try? MoldJSON.encoder.encode(self) { defaults.set(data, forKey: Self.key) }
    }
    public mutating func retainHosts(_ hosts: Set<UUID>) {
        pendingViewed = pendingViewed.map { $0.filter { hosts.contains($0.host) } }
        guard known.keys.contains(where: { !hosts.contains($0) }) else { return }
        positions = nil
        unreadCount = nil
        known = known.filter { hosts.contains($0.key) }
        unread = unread.filter { hosts.contains($0.host) }
        pendingViewed = pendingViewed.map { $0.filter { hosts.contains($0.host) } }
        visibleGroups = visibleGroups.compactMap {
            let present = $0.filter { hosts.contains($0.host) }
            return present.isEmpty ? nil : Set(present)
        }
    }
}

/// IDs captured from displayed Library projections, scoped to one window visit.
/// Opening a viewer does not end the visit; leaving the Library does.
public struct LibraryVisit: Sendable {
    private var copies: Set<PrintID> = []
    public init() {}
    public mutating func capture(_ entries: [LibraryEntry], scope: LibraryScope = .all) {
        guard !scope.isTrash else { return }
        for entry in entries { copies.formUnion(entry.everyCopy.map(\.id)) }
    }
    public mutating func finish() -> Set<PrintID> {
        let captured = copies
        copies.removeAll(keepingCapacity: true)
        return captured
    }
}
