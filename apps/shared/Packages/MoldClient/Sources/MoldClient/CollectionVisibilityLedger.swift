import Foundation
import Observation

/// Explicit collection visibility intent survives partial writes and app restarts.
/// Only captured routes may receive it; a replacement machine never inherits a
/// mutation intended for the former machine. Newer edits fence older replies.
@MainActor
@Observable
public final class CollectionVisibilityLedger {
    public struct Route: Codable, Equatable, Sendable {
        public let url: URL
        public let instance: String?
        public init(_ host: MoldHost) {
            url = host.connectionOriginalURL ?? host.baseURL
            instance = host.connectionInstanceID
        }
    }
    public struct Intent: Codable, Sendable {
        public let hidden: Bool
        public let revision: UUID
        public let routes: [UUID: Route]
    }
    public private(set) var generation = UUID()
    public private(set) var intents: [String: Intent]

    public init() { intents = [:] }
    private init(_ intents: [String: Intent]) { self.intents = intents }

    @discardableResult
    public func set(_ slug: String, hidden: Bool, hosts: [MoldHost]) -> Intent {
        let intent = Intent(hidden: hidden, revision: UUID(), routes: Dictionary(
            hosts.map { ($0.id, Route($0)) }, uniquingKeysWith: { first, _ in first }))
        intents[slug] = intent
        generation = UUID()
        return intent
    }

    public func desiredHidden(slug: String, fallback: Bool) -> Bool { intents[slug]?.hidden ?? fallback }
    public func isCurrent(slug: String, revision: UUID, host: MoldHost) -> Bool {
        guard let intent = intents[slug], intent.revision == revision else { return false }
        return intent.routes[host.id] == Route(host)
    }
    public func complete(slug: String, revision: UUID) {
        if intents[slug]?.revision == revision { intents[slug] = nil; generation = UUID() }
    }
    /// Repairs mixed hidden flags conservatively, or the newer deliberate
    /// intent. Reads and writes are fenced across every suspension.
    public func reconcile(
        hosts: () -> [MoldHost],
        collections: () -> [UUID: [Collection]],
        available: () -> Set<UUID>,
        send: (MoldHost, Collection, Bool) async -> Bool
    ) async {
        let shelves = CollectionShelf.merge(collections())
        for shelf in shelves {
            if intents[shelf.slug] == nil, shelf.hidden,
               shelf.hosts.contains(where: { host, id in
                   collections()[host]?.contains { $0.id == id && $0.hidden != true } == true
               }) {
                set(shelf.slug, hidden: true, hosts: hosts())
            }
            guard let intent = intents[shelf.slug] else { continue }
            var complete = true
            for (hostID, route) in intent.routes {
                guard let host = hosts().first(where: { $0.id == hostID }) else { continue }
                guard Route(host) == route else { continue } // replacement route is never a target
                guard available().contains(hostID) else { complete = false; continue }
                guard let collection = collections()[hostID]?.first(where: { $0.slug == shelf.slug }) else { continue }
                guard isCurrent(slug: shelf.slug, revision: intent.revision, host: host) else { complete = false; break }
                if collection.hidden == intent.hidden { continue }
                let succeeded = await send(host, collection, intent.hidden)
                guard isCurrent(slug: shelf.slug, revision: intent.revision, host: host),
                      hosts().contains(host) else { complete = false; break }
                // A write is not a fresh inventory. Keep intent until a later
                // listing confirms it, so a concurrent old read cannot revive
                // the flags that the person just changed.
                complete = false
                if !succeeded { continue }
            }
            if complete { self.complete(slug: shelf.slug, revision: intent.revision) }
        }
    }

    public func persist(to defaults: UserDefaults, key: String) {
        if let data = try? MoldJSON.localEncoder.encode(intents) { defaults.set(data, forKey: key) }
    }
    public static func load(from defaults: UserDefaults, key: String) -> CollectionVisibilityLedger {
        let intents = defaults.data(forKey: key).flatMap {
            try? MoldJSON.localDecoder.decode([String: Intent].self, from: $0)
        } ?? [:]
        return CollectionVisibilityLedger(intents)
    }
}
