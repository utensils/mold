import AppKit
import MoldClient
import SwiftUI
import UniformTypeIdentifiers

/// The things you can do to a print, in one place so the context menu, the
/// keyboard and the viewer all do the same thing.
@MainActor
struct LibraryActions {
    let hosts: HostStore
    let library: LibraryStore
    /// Set by the pane so a print can seed a new render. Absent in contexts
    /// that have no Generate pane to send it to.
    var reuse: ((LibraryEntry) -> Void)?
    /// Set by the pane, which owns the dialog. Destroying somebody's pictures
    /// must ask first, and there are three doors into it -- the Delete key,
    /// the context menu and the inspector -- so the question belongs here
    /// rather than at each of them.
    var confirmDestruction: ((Destruction) -> Void)?
    /// Turns prints into files on this disk. Absent in contexts that only
    /// read -- nothing here fetches bytes without it.
    var materializer: PrintMaterializer?

    /// Something permanent, waiting on an answer.
    struct Destruction: Identifiable {
        let id = UUID()
        let title: String
        let message: String
        let verb: String
        let perform: () -> Void
    }

    // Internal, not private: `LibraryActions+Destructive` needs it, and the
    // 150-line lint is what put that half in its own file.
    func backend(_ id: MoldHost.ID) -> (any MoldBackend)? {
        hosts.hosts.first { $0.id == id }.map { hosts.backend(for: $0) }
    }

    private func host(_ entry: LibraryEntry) -> MoldHost? {
        hosts.hosts.first { $0.id == entry.hostID }
    }

    func toggleFavorite(_ entries: [LibraryEntry]) {
        // If any is not a favourite, the action makes them all favourites --
        // the same rule the Finder uses for mixed selections.
        let makeFavorite = entries.contains { !$0.print.isFavorite }
        library.setFavorite(makeFavorite, on: entries, backend: backend)
    }

    func setTag(_ tag: String, adding: Bool, on entries: [LibraryEntry]) {
        library.setTag(tag, adding: adding, on: entries, backend: backend)
    }

    func moveToTrash(_ entries: [LibraryEntry]) {
        Task { await library.moveToTrash(entries, backend: backend) }
    }

    func restore(_ entries: [LibraryEntry]) {
        Task {
            await library.restore(entries, backend: backend)
            await reload()
        }
    }

    func reload() async {
        await library.refresh(hosts: hosts.hosts) { hosts.backend(for: $0) }
        await library.refreshTrash(hosts: hosts.hosts) { hosts.backend(for: $0) }
        // Shelves and tags travel with the index: reloading one without the
        // other leaves a renamed collection still reading its old name.
        await library.refreshOrganization(hosts: hosts.hosts) { hosts.backend(for: $0) }
    }

    /// Puts the picture on the pasteboard, so ⌘V works anywhere.
    func copy(_ entries: [LibraryEntry]) {
        Task {
            var images: [NSImage] = []
            for entry in entries.prefix(10) {
                if let data = await data(for: entry), let image = NSImage(data: data) {
                    images.append(image)
                }
            }
            guard !images.isEmpty else { return }
            NSPasteboard.general.clearContents()
            NSPasteboard.general.writeObjects(images)
        }
    }

    /// Saves the original bytes. One print gets a save panel; several get a
    /// folder, because ten save panels in a row is not a feature.
    ///
    /// Goes through the materializer like everything else, so saving a clip
    /// you just previewed is a local copy rather than a second download.
    func save(_ entries: [LibraryEntry]) {
        Task {
            guard let first = entries.first else { return }
            if entries.count == 1 {
                let panel = NSSavePanel()
                panel.nameFieldStringValue = first.print.filename
                guard await panel.begin() == .OK, let url = panel.url,
                      let source = await files(for: [first]).first else { return }
                try? FileManager.default.removeItem(at: url)
                try? FileManager.default.copyItem(at: source.url, to: url)
            } else {
                let panel = NSOpenPanel()
                panel.canChooseDirectories = true
                panel.canChooseFiles = false
                panel.prompt = "Save Here"
                guard await panel.begin() == .OK, let folder = panel.url else { return }
                for entry in entries {
                    guard let source = await files(for: [entry]).first else { continue }
                    let destination = folder.appending(path: entry.print.filename)
                    try? FileManager.default.removeItem(at: destination)
                    try? FileManager.default.copyItem(at: source.url, to: destination)
                }
            }
        }
    }

    /// A URL a player can open directly, ticketed if the machine needs it.
    func playableURL(for entry: LibraryEntry) async -> URL? {
        guard let host = host(entry), let client = backend(host.id) else { return nil }
        return await client.playableURL(for: entry.print.filename)
    }

    /// The stored bytes for a print, fetched from the machine that holds it.
    func data(for entry: LibraryEntry) async -> Data? {
        guard let host = host(entry) else { return nil }
        return try? await backend(host.id)?.media(entry.print.filename,
                                                  trashed: entry.print.trashedAt != nil)
    }
}
