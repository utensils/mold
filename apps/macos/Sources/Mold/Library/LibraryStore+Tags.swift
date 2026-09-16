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
    ///
    /// Reaches every machine `HostStore` knows about, not just the ones that
    /// have reported a tag already: a machine whose `tags()` call never landed
    /// (or simply hasn't been asked yet) still carries the tag on its prints,
    /// and skipping it left a rename half-done with no sign anything was wrong.
    func renameTag(_ name: String, to newName: String) {
        let clean = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty, clean.caseInsensitiveCompare(name) != .orderedSame else { return }

        // Locally first, on every print that carries it, so the chips change
        // as you press Return rather than a round trip later.
        retag(name, to: clean)
        undo.register("Rename Tag") { [weak self] in
            self?.renameTag(clean, to: name)
        }

        Task {
            for host in hosts.hosts {
                do { _ = try await hosts.backend(for: host).renameTag(name, to: clean) } catch {
                    // A machine that has never seen the tag answers 404, which
                    // is not a failure of the rename -- it is a machine with
                    // nothing to rename.
                    if let mold = error as? MoldClientError,
                       case let .http(status, _, _) = mold, status == 404 { continue }
                    failures[host.id] = "Couldn't rename \(name)."
                }
            }
            await reloadTags()
        }
    }

    /// Takes a tag off every print on every machine.
    ///
    /// NOT undoable, and the caller asks first. Putting it back would mean
    /// knowing which prints carried it, and by the time the answer came back
    /// the machines had already forgotten.
    func deleteTag(_ name: String) {
        retag(name, to: nil)
        // A tag that no longer exists cannot be renamed back to, and an undo
        // stack that offers it is offering a lie.
        undo.forget()

        Task {
            for host in hosts.hosts {
                try? await hosts.backend(for: host).deleteTag(name)
            }
            await reloadTags()
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
                return entry.replacingPrint(mutable.build())
            }
        }
        rebuild()
    }

    private func reloadTags() async {
        etags.removeAll()
        for host in hosts.hosts {
            if let tags = try? await hosts.backend(for: host).tags() { tagsPerHost[host.id] = tags }
        }
    }
}
