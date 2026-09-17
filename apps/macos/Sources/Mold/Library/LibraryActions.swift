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
    /// Making a print bigger, on the machine that holds it. Absent in
    /// contexts with no store to act through, which reads as "cannot".
    var upscales: UpscaleStore?
    /// What to do about the SHELF being shown -- rename it, hide it, delete
    /// it. Declared in the menu plan so both menus offer the three, and
    /// answered by whoever owns the sheet and the confirm.
    var collectionAction: ((LibraryAction) -> Void)?
    /// Set by whoever owns the export sheet. A mesh export carries controls
    /// the host advertised -- a print size, an up axis, a turntable's frames
    /// -- so the ones that have any ASK before converting. Absent in contexts
    /// with nowhere to put a sheet, where the host's own defaults are used.
    var meshExport: ((MeshExportPrompt) -> Void)?

    /// Moved to `Shell/Destruction.swift` (M5 S5, decision 12) so the Models
    /// pane can raise the same confirm without depending on a Library type.
    /// Kept as a typealias so every existing `LibraryActions.Destruction` and
    /// unqualified `Destruction(...)` inside this type's extensions still
    /// compiles unchanged.
    typealias Destruction = Mold.Destruction

    func toggleFavorite(_ entries: [LibraryEntry]) {
        // If any is not a favourite, the action makes them all favourites --
        // the same rule the Finder uses for mixed selections.
        let makeFavorite = entries.contains { !$0.print.isFavorite }
        library.setFavorite(makeFavorite, on: entries)
    }

    func setTag(_ tag: String, adding: Bool, on entries: [LibraryEntry]) {
        library.setTag(tag, adding: adding, on: entries)
    }

    func moveToTrash(_ entries: [LibraryEntry]) {
        Task { await library.moveToTrash(entries) }
    }

    func restore(_ entries: [LibraryEntry]) {
        Task {
            await library.restore(entries)
            await reload()
        }
    }

    func reload() async {
        await library.reload()
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
                // The panel already asked about replacing, so removing first
                // is what the person agreed to -- but a failure here is theirs
                // to hear about, not something to swallow.
                do {
                    try? FileManager.default.removeItem(at: url)
                    try FileManager.default.copyItem(at: source.url, to: url)
                } catch {
                    hosts.report(error, on: first.hostID, doing: "save that print")
                }
            } else {
                let panel = NSOpenPanel()
                panel.canChooseDirectories = true
                panel.canChooseFiles = false
                panel.prompt = "Save Here"
                guard await panel.begin() == .OK, let folder = panel.url else { return }
                await saveAll(entries, into: folder)
            }
        }
    }

    /// A URL a player can open directly, ticketed if the machine needs it.
    /// `nil` on a failed ticket -- reported rather than thrown, so the viewer
    /// stays a place that shows pictures, not one that also handles errors.
    func playableURL(for entry: LibraryEntry) async -> URL? {
        guard let backend = hosts.backend(for: entry.hostID) else { return nil }
        do {
            return try await backend.playableURL(for: entry.print.filename)
        } catch {
            hosts.report(error, on: entry.hostID, doing: "play that clip")
            return nil
        }
    }

    /// The stored bytes for a print, fetched from the machine that holds it.
    ///
    /// Bounded, because the whole body is buffered before anything here sees
    /// it: a host answering without limit would take the process down, and
    /// `copy` decodes up to ten of these into `NSImage` at once. The ceiling
    /// is the server's own for one member -- see `ResponseCeiling`, and the
    /// note there about where this belongs once the transport streams.
    func data(for entry: LibraryEntry) async -> Data? {
        guard let backend = hosts.backend(for: entry.hostID) else { return nil }
        do {
            let bytes = try await backend.media(entry.print.filename,
                                                trashed: entry.print.trashedAt != nil)
            return try ResponseCeiling.checked(bytes, ceiling: ResponseCeiling.media,
                                               what: "that print")
        } catch is CancellationError {
            return nil
        } catch {
            hosts.report(error, on: entry.hostID, doing: "read that print")
            return nil
        }
    }
}
