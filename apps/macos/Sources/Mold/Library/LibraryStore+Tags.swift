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
                do {
                    _ = try await hosts.backend(for: host).renameTag(name, to: clean)
                    hosts.succeeded(on: host.id)
                } catch {
                    // A machine that has never seen the tag answers 404, which
                    // is not a failure of the rename -- it is a machine with
                    // nothing to rename.
                    if let mold = error as? MoldClientError,
                       case let .http(status, _, _) = mold, status == 404 { continue }
                    hosts.report(error, on: host.id, doing: "rename the tag “\(name)”")
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
                do {
                    try await hosts.backend(for: host).deleteTag(name)
                    hosts.succeeded(on: host.id)
                } catch {
                    hosts.report(error, on: host.id, doing: "delete the tag “\(name)”")
                }
            }
            await reloadTags()
        }
    }

    /// Rewrites or removes a tag on every print on screen. What the rewrite
    /// IS belongs to `TagRewrite`; this is which rows it runs over.
    private func retag(_ name: String, to replacement: String?) {
        for (hostID, list) in perHost {
            perHost[hostID] = TagRewrite.applied(to: list, name: name,
                                                 replacement: replacement)
        }
        rebuild()
    }

    private func reloadTags() async {
        etags.removeAll()
        for host in hosts.hosts {
            do {
                tagsPerHost[host.id] = try await hosts.backend(for: host).tags()
                // Scoped to its own verb: this is a passive refresh that runs
                // after every rename and delete, and must not silently clear
                // a failure THAT action just reported.
                hosts.succeeded(on: host.id, doing: "read its tags")
            } catch {
                hosts.report(error, on: host.id, doing: "read its tags")
            }
        }
    }
}
