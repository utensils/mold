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
    ///
    /// A MESH is previewed by its poster (`meshPosterFile`): macOS ships no
    /// GLB preview generator, so the stored bytes produced a generic icon --
    /// after downloading all of them.
    func quickLook(_ entries: [LibraryEntry]) {
        Task {
            var files: [(url: URL, title: String)] = []
            for entry in entries {
                if entry.print.isMesh {
                    if let poster = await meshPosterFile(for: entry) { files.append(poster) }
                } else if let file = await self.files(for: [entry]).first {
                    files.append(file)
                }
            }
            QuickLook.shared.show(files)
        }
    }

    /// Saves several prints into one folder the person chose.
    ///
    /// Nothing there is ever destroyed: a name already in the folder, or
    /// already claimed by an earlier print in this same selection, gets the
    /// Finder's ` 2` suffix. This path used to `removeItem` at the
    /// destination first -- so somebody's own `robot.png` went, with no
    /// overwrite prompt (the single-print `NSSavePanel` asks; this never did)
    /// -- and `Robot.png` beside `robot.png` collapsed to one file on the
    /// case-insensitive volume APFS is by default: ten prints asked for, nine
    /// saved, no message.
    func saveAll(_ entries: [LibraryEntry], into folder: URL) async {
        let existing = (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
        var names = SaveNames(existing: existing)
        for entry in entries {
            guard let source = await files(for: [entry]).first else { continue }
            // The person chose THIS folder and nothing above it.
            guard let destination = SafeFilename.url(names.claim(entry.print.filename),
                                                     in: folder) else { continue }
            do {
                try FileManager.default.copyItem(at: source.url, to: destination)
            } catch {
                // A disk full, a read-only folder: silent before, through a
                // `try?` that swallowed it whole.
                hosts.report(error, on: entry.hostID, doing: "save that print")
                return
            }
        }
    }
}
