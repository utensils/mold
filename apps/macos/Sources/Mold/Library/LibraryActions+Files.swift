import AppKit
import MoldClient

// Turning prints into files on this disk, and the four things that need one.
//
// Quick Look, ShareLink, Save to… and the Finder drag all want a real file,
// and the bytes live on another machine. They all go through the same
// materializer, so previewing a clip and then sharing it costs one download.
extension LibraryActions {

    /// Files for these prints, downloading whatever the cache has not got.
    func files(for entries: [LibraryEntry]) async -> [(url: URL, title: String)] {
        guard let materializer else { return [] }
        var made: [(url: URL, title: String)] = []
        for entry in entries {
            guard let url = await materializer.url(for: entry, fetch: { await data(for: entry) })
            else { continue }
            made.append((url, entry.print.displayName))
        }
        return made
    }

    /// Space. Downloads first, so the panel opens on the picture rather than
    /// on an empty frame that fills in later.
    func quickLook(_ entries: [LibraryEntry]) {
        Task { QuickLook.shared.show(await files(for: entries)) }
    }
}
