import Foundation

/// Derive library privacy once per mutation, never once per source usage or row.
@MainActor
public final class SourceMediaAccessIndex {
    public struct Key: Equatable {
        let revision: Int
        let privacyRevision: Int?
        let visibility: UUID
        let hosts: [MoldHost]
        let instances: [UUID: String]
        public init(revision: Int, visibility: UUID, hosts: [MoldHost], instances: [UUID: String], privacyRevision: Int? = nil) {
            self.privacyRevision = privacyRevision
            self.revision = revision; self.visibility = visibility
            self.hosts = hosts; self.instances = instances
        }
    }
    public struct Snapshot {
        public let owners: [LibraryEntry]
        public let authorities: Set<SourceMediaAuthority>
        private let visible: [PrintID: SourceMediaAuthority]
        private let details: [PrintID: SourceMediaAuthority]
        public init(visible: [LibraryEntry], details: [LibraryEntry], hosts: [MoldHost], instances: [UUID: String]) {
            let routes = Dictionary(hosts.map { ($0.id, $0) }, uniquingKeysWith: { first, _ in first })
            owners = visible.filter { routes[$0.hostID] != nil }
            func index(_ entries: [LibraryEntry]) -> [PrintID: SourceMediaAuthority] {
                Dictionary(entries.compactMap { owner in
                    routes[owner.hostID].map { (owner.id, SourceMediaAuthority(owner: owner, route: $0, instance: instances[owner.hostID])) }
                }, uniquingKeysWith: { first, _ in first })
            }
            self.visible = index(owners); self.details = index(details)
            authorities = Set(self.visible.values)
        }
        public func allows(_ authority: SourceMediaAuthority, detail: Bool = false) -> Bool {
            (detail ? details : visible)[authority.owner] == authority
        }
    }
    private var key: Key?
    private var value: Snapshot?
    public private(set) var derivations = 0
    public init() {}
    public func snapshot(key: Key, build: () -> Snapshot) -> Snapshot {
        if self.key == key, let value { return value }
        let result = build()
        self.key = key; value = result; derivations += 1
        return result
    }
}
