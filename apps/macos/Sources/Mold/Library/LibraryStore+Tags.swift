import Foundation
import MoldClient

// Tags as things in their own right, rather than as marks on one print.
//
// Renaming and deleting reach every print carrying the tag, on every machine.
// That is one request per machine and not a loop over prints, which is what
// stops a rename finishing halfway and leaving a library with two spellings.
@MainActor
extension LibraryStore {

    /// Renames a tag everywhere, on every machine.
    ///
    /// Undoable, because it is exactly reversible: the inverse of a rename is
    /// the rename back, and nothing is lost on the way.
    func renameTag(_ name: String, to newName: String,
                   backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) {
        let clean = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty, clean.caseInsensitiveCompare(name) != .orderedSame else { return }

        // Locally first, on every print that carries it, so the chips change
        // as you press Return rather than a round trip later.
        retag(name, to: clean)
        undo.register("Rename Tag") { [weak self] in
            self?.renameTag(clean, to: name, backend: backend)
        }

        Task {
            for hostID in tagsPerHost.keys {
                guard let client = backend(hostID) as? HTTPBackend else { continue }
                do { _ = try await client.renameTag(name, to: clean) } catch {
                    // A machine that has never seen the tag answers 404, which
                    // is not a failure of the rename -- it is a machine with
                    // nothing to rename.
                    if let mold = error as? MoldClientError,
                       case let .http(status, _, _) = mold, status == 404 { continue }
                    failures[hostID] = "Couldn't rename \(name)."
                }
            }
            await reloadTags(backend)
        }
    }

    /// Takes a tag off every print on every machine.
    ///
    /// NOT undoable, and the caller asks first. Putting it back would mean
    /// knowing which prints carried it, and by the time the answer came back
    /// the machines had already forgotten.
    func deleteTag(_ name: String, backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) {
        retag(name, to: nil)
        // A tag that no longer exists cannot be renamed back to, and an undo
        // stack that offers it is offering a lie.
        undo.forget()

        Task {
            for hostID in tagsPerHost.keys {
                guard let client = backend(hostID) as? HTTPBackend else { continue }
                try? await client.deleteTag(name)
            }
            await reloadTags(backend)
        }
    }

    /// Rewrites or removes a tag on every print on screen.
    private func retag(_ name: String, to replacement: String?) {
        for (hostID, list) in perHost {
            perHost[hostID] = list.map { entry in
                guard entry.print.tagList.contains(where: {
                    $0.caseInsensitiveCompare(name) == .orderedSame
                }) else { return entry }
                var mutable = GalleryPrint.Mutable(entry.print)
                var tags = mutable.tags ?? []
                tags.removeAll { $0.caseInsensitiveCompare(name) == .orderedSame }
                if let replacement,
                   !tags.contains(where: { $0.caseInsensitiveCompare(replacement) == .orderedSame }) {
                    tags.append(replacement)
                }
                mutable.tags = tags
                return LibraryEntry(hostID: entry.hostID, hostName: entry.hostName,
                                    print: mutable.build())
            }
        }
        rebuild()
    }

    private func reloadTags(_ backend: @escaping (MoldHost.ID) -> (any MoldBackend)?) async {
        etags.removeAll()
        for hostID in tagsPerHost.keys {
            guard let client = backend(hostID) as? HTTPBackend else { continue }
            if let tags = try? await client.tags() { tagsPerHost[hostID] = tags }
        }
    }
}
