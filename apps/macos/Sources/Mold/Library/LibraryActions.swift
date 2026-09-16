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

    private func backend(_ id: MoldHost.ID) -> (any MoldBackend)? {
        hosts.hosts.first { $0.id == id }.map { hosts.backend(for: $0) }
    }

    private func host(_ entry: LibraryEntry) -> MoldHost? {
        hosts.hosts.first { $0.id == entry.hostID }
    }

    func toggleFavorite(_ entries: [LibraryEntry]) {
        // If any is not a favourite, the action makes them all favourites --
        // the same rule the Finder uses for mixed selections.
        let makeFavorite = entries.contains { !$0.print.isFavorite }
        Task { await library.setFavorite(makeFavorite, on: entries, backend: backend) }
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

    func deleteForever(_ entries: [LibraryEntry]) {
        Task {
            await library.deleteForever(entries, backend: backend)
            await library.refreshTrash(hosts: hosts.hosts) { hosts.backend(for: $0) }
        }
    }

    func reload() async {
        await library.refresh(hosts: hosts.hosts) { hosts.backend(for: $0) }
        await library.refreshTrash(hosts: hosts.hosts) { hosts.backend(for: $0) }
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
    func save(_ entries: [LibraryEntry]) {
        Task {
            guard let first = entries.first else { return }
            if entries.count == 1 {
                let panel = NSSavePanel()
                panel.nameFieldStringValue = first.print.filename
                guard await panel.begin() == .OK, let url = panel.url,
                      let data = await data(for: first) else { return }
                try? data.write(to: url)
            } else {
                let panel = NSOpenPanel()
                panel.canChooseDirectories = true
                panel.canChooseFiles = false
                panel.prompt = "Save Here"
                guard await panel.begin() == .OK, let folder = panel.url else { return }
                for entry in entries {
                    guard let data = await data(for: entry) else { continue }
                    try? data.write(to: folder.appending(path: entry.print.filename))
                }
            }
        }
    }

    /// The stored bytes for a print, fetched from the machine that holds it.
    func data(for entry: LibraryEntry) async -> Data? {
        guard let host = host(entry) else { return nil }
        var request = URLRequest(
            url: MediaURL(baseURL: host.baseURL).media(entry.print.filename,
                                                       trashed: entry.print.trashedAt != nil))
        if let key = host.apiKey, !key.isEmpty {
            request.setValue(key, forHTTPHeaderField: "X-Api-Key")
        }
        return try? await URLSession.shared.data(for: request).0
    }
}
