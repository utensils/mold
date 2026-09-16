import Foundation

/// A print's identity in a library merged across machines.
///
/// A filename alone is NOT an identity: two hosts generate names from the same
/// scheme and will collide, and mold's own rule is that a print belongs to the
/// machine that made it.
public struct PrintID: Hashable, Codable, Sendable {
    public let host: MoldHost.ID
    public let filename: String

    public init(host: MoldHost.ID, filename: String) {
        self.host = host
        self.filename = filename
    }
}

/// A print paired with the machine that owns it -- what the merged Library
/// actually holds.
public struct LibraryEntry: Identifiable, Hashable, Sendable {
    public let hostID: MoldHost.ID
    public let hostName: String
    public let print: GalleryPrint
    /// Everything searchable, folded once at construction.
    ///
    /// A library holds thousands of prints and the search field filters on
    /// every keystroke; folding each row's text again per keystroke is work
    /// proportional to the library, repeated for every character typed.
    public let searchKey: String

    /// A print as one machine reports it.
    ///
    /// Takes the whole `MoldHost` rather than an id and a name apart, because
    /// every construction site already has the machine in hand -- and the one
    /// that didn't (an event arriving for a host with no prints yet) is what
    /// used to silently drop the row.
    public init(host: MoldHost, print: GalleryPrint) {
        self.init(hostID: host.id, hostName: host.name, print: print)
    }

    /// The same row with a new print, re-folding the search key. What every
    /// optimistic local edit and every re-applied server echo actually wants:
    /// the machine identity is unchanged, only what it says about the print.
    public func replacingPrint(_ print: GalleryPrint) -> LibraryEntry {
        LibraryEntry(hostID: hostID, hostName: hostName, print: print)
    }

    private init(hostID: MoldHost.ID, hostName: String, print: GalleryPrint) {
        self.hostID = hostID
        self.hostName = hostName
        self.print = print
        self.searchKey = Self.fold([
            print.metadata.prompt, print.metadata.model, print.metadata.family,
            print.title, print.filename, hostName,
            print.metadata.seed.map(String.init),
        ].compactMap(\.self).joined(separator: " ") + " " + print.tagList.joined(separator: " "))
    }

    public var id: PrintID { PrintID(host: hostID, filename: print.filename) }
    public var createdAt: Date { print.createdAt }

    /// Case-, diacritic- and width-insensitive, so "cafe" finds "Café".
    ///
    /// Public because anything OFFERING a filter has to fold the same way the
    /// filter itself does, or a suggestion appears that then matches nothing.
    public static func fold(_ text: String) -> String {
        text.folding(options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive],
                     locale: .current)
    }

    /// Every whitespace-separated token must appear, so more words narrow.
    public func matches(_ query: String) -> Bool {
        let tokens = Self.fold(query).split(separator: " ")
        guard !tokens.isEmpty else { return true }
        return tokens.allSatisfy { searchKey.contains($0) }
    }
}
