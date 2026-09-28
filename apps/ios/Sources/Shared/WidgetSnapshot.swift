import Foundation

/// What the widgets draw, written by the app into the App Group (DESIGN.md
/// §5.8). Widgets never network and never read a key: this file and the
/// small JPEGs beside it are everything they know.
nonisolated struct WidgetSnapshot: Codable, Equatable {
    struct Print: Codable, Equatable, Identifiable {
        var host: UUID
        var machine: String
        var filename: String
        var title: String
        /// The JPEG's name in `AppGroup.widget`.
        var image: String
        var favourite: Bool
        var kind: Kind
        var made: Date

        enum Kind: String, Codable { case picture, clip, mesh }
        var id: String { "\(host.uuidString)/\(filename)" }
        var link: DeepLink { .print(host: host, filename: filename) }
    }

    struct Machine: Codable, Equatable, Identifiable {
        var id: UUID
        var name: String
    }

    var updated: Date
    /// Newest first.
    var prints: [Print]
    var machines: [Machine]
    var rendering: Int
    var held: Int
    var waiting: Int
    /// The render on screen in the app, 0...1, when there is one.
    var progress: Double?

    static let empty = WidgetSnapshot(updated: .distantPast, prints: [], machines: [], rendering: 0, held: 0,
                                      waiting: 0, progress: nil)

    static var url: URL { AppGroup.widget.appending(path: "snapshot.json") }

    static func load(from url: URL = url) -> WidgetSnapshot {
        guard let data = try? Data(contentsOf: url),
              let snapshot = try? JSONDecoder().decode(WidgetSnapshot.self, from: data) else { return .empty }
        return snapshot
    }

    func save(to url: URL = Self.url) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        try encoder.encode(self).write(to: url, options: .atomic)
    }

    /// The prints a configured widget shows: one machine's, or everyone's;
    /// every print, or favourites only.
    func prints(machine: UUID?, favouritesOnly: Bool) -> [Print] {
        prints.filter { (machine == nil || $0.host == machine) && (!favouritesOnly || $0.favourite) }
    }

    /// "2 rendering · 1 held", or "Nothing waiting".
    var queueSummary: String {
        var parts: [String] = []
        if rendering > 0 { parts.append(String(localized: "\(rendering) rendering")) }
        if held > 0 { parts.append(String(localized: "\(held) held")) }
        if waiting > 0 { parts.append(String(localized: "\(waiting) waiting")) }
        return parts.isEmpty ? String(localized: "Nothing waiting") : parts.joined(separator: " · ")
    }
}
