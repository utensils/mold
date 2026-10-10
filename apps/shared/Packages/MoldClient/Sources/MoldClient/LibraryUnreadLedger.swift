import Foundation

/// Durable, device-local icon state. Read identity follows merged copy groups,
/// including collision-renamed copies and distinct outputs sharing a filename.
public struct LibraryUnreadLedger: Codable, Equatable, Sendable {
    private var known: [UUID: Set<PrintID>] = [:]
    private var unread: Set<PrintID> = []
    private var visibleGroups: [Set<PrintID>] = []
    private var pendingViewed: Set<PrintID>?

    public init() {}
    public var count: Int { visibleGroups.filter { !$0.isDisjoint(with: unread) }.count }

    public mutating func observe(entries: [LibraryEntry], visible: [LibraryEntry], loadedHosts: Set<UUID>, presentHosts: Set<UUID>) {
        retainHosts(presentHosts)
        let initialized = Set(known.keys)
        let previous = known.values.reduce(into: Set<PrintID>()) { $0.formUnion($1) }
        for entry in entries {
            let copies = Set(entry.everyCopy.map(\.id))
            let familiar = copies.intersection(previous)
            if pendingViewed?.isDisjoint(with: copies) == false || !familiar.subtracting(unread).isEmpty {
                // A new copy of a read print inherits its read state.
                unread.subtract(copies)
            } else if !copies.isDisjoint(with: unread) || (familiar.isEmpty && copies.contains { initialized.contains($0.host) }) {
                unread.formUnion(copies)
            }
            for copy in copies { known[copy.host, default: []].insert(copy) }
            pendingViewed?.subtract(copies)
        }
        for host in loadedHosts where known[host] == nil { known[host] = [] }
        let incoming = visible.map { Set($0.everyCopy.map(\.id)) }
        // Saved read state survives cache clearing/unavailable machines. Keep
        // their last visible groups unless a loaded copy already represents it.
        let incomingCopyIDs = incoming.reduce(into: Set<PrintID>()) { $0.formUnion($1) }
        let retained = visibleGroups.compactMap { group -> Set<PrintID>? in
            guard group.isDisjoint(with: incomingCopyIDs) else { return nil }
            let unavailable = group.filter { presentHosts.contains($0.host) && !loadedHosts.contains($0.host) }
            return unavailable.isEmpty ? nil : Set(unavailable)
        }
        visibleGroups = incoming + retained
    }

    public func isUnread(_ entry: LibraryEntry) -> Bool { !Set(entry.everyCopy.map(\.id)).isDisjoint(with: unread) }

    /// A successfully displayed generation can precede its gallery listing.
    public mutating func view(_ id: PrintID) {
        if known[id.host]?.contains(id) == true { unread.remove(id) }
        else {
            if pendingViewed == nil { pendingViewed = [] }
            pendingViewed?.insert(id)
        }
    }

    public mutating func view(_ entry: LibraryEntry) { unread.subtract(entry.everyCopy.map(\.id)) }
    public mutating func markSeen(_ entries: [LibraryEntry]) { unread.subtract(entries.flatMap(\.everyCopy).map(\.id)) }

    public static let key = "library.unreadMedia.v1"
    public static func load(from defaults: UserDefaults) -> Self {
        defaults.data(forKey: key).flatMap { try? MoldJSON.decoder.decode(Self.self, from: $0) } ?? Self()
    }
    public func save(to defaults: UserDefaults) {
        if let data = try? MoldJSON.encoder.encode(self) { defaults.set(data, forKey: Self.key) }
    }
    public mutating func retainHosts(_ hosts: Set<UUID>) {
        known = known.filter { hosts.contains($0.key) }
        unread = unread.filter { hosts.contains($0.host) }
        pendingViewed = pendingViewed.map { $0.filter { hosts.contains($0.host) } }
        visibleGroups = visibleGroups.compactMap {
            let present = $0.filter { hosts.contains($0.host) }
            return present.isEmpty ? nil : Set(present)
        }
    }
}
