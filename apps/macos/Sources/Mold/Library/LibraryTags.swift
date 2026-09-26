import Foundation
import MoldClient

/// Tags as things in their own right, rather than as marks on one print.
///
/// Its own object rather than more of `LibraryStore`: the store holds the
/// merged timeline, and this holds the tag INDEX -- what each machine says it
/// carries and how often -- which is a different list with a different
/// lifetime. Renaming and deleting reach every print carrying the tag, on
/// every machine: one request per machine and not a loop over prints, which is
/// what stops a rename finishing halfway and leaving a library with two
/// spellings. It takes the store as a parameter for the rows it rewrites on
/// screen, the arrangement `LibraryMutations` already has with it.
@MainActor
@Observable
final class LibraryTags {
    /// Tags as each MACHINE holds them. Never merged in storage -- only when
    /// they are read, by `counts`.
    var perHost: [MoldHost.ID: [TagCount]] = [:]

    /// Tag names and how many prints carry them, summed across machines.
    /// Sorted by how much they are used, because a tag suggestion list is only
    /// useful if the tags you actually use are at the top.
    var counts: [TagCount] {
        var totals: [String: Int] = [:]
        for counts in perHost.values {
            for tag in counts { totals[tag.name, default: 0] += tag.count }
        }
        return totals
            .map { TagCount(name: $0.key, count: $0.value) }
            .sorted {
                $0.count == $1.count
                    ? $0.name.localizedStandardCompare($1.name) == .orderedAscending
                    : $0.count > $1.count
            }
    }

    /// A machine that was removed must not keep contributing tags.
    func prune(to live: Set<MoldHost.ID>) {
        perHost = perHost.filter { live.contains($0.key) }
    }

    /// Renames a tag everywhere, on every machine.
    ///
    /// Undoable, because it is exactly reversible: the inverse of a rename is
    /// the rename back, and nothing is lost on the way.
    ///
    /// Reaches every machine `HostStore` knows about, not just the ones that
    /// have reported a tag already: a machine whose `tags()` call never landed
    /// (or simply hasn't been asked yet) still carries the tag on its prints,
    /// and skipping it left a rename half-done with no sign anything was wrong.
    func rename(_ name: String, to newName: String, in store: LibraryStore) {
        let clean = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty, clean.caseInsensitiveCompare(name) != .orderedSame else { return }

        // Locally first, on every print that carries it, so the chips change
        // as you press Return rather than a round trip later.
        retag(name, to: clean, in: store)
        store.undo.register("Rename Tag") { [weak store] in
            guard let store else { return }
            store.tags.rename(clean, to: name, in: store)
        }

        let hosts = store.hosts
        // Freeze the destinations with the action. A machine added halfway
        // through belongs to the next refresh; changing the denominator while
        // this operation is on screen would make its progress misleading.
        let destinations: [(host: MoldHost, client: any MoldBackend)] = hosts.hosts.map {
            ($0, hosts.backend(for: $0))
        }
        let activity = store.beginBulkActivity(
            "Renaming tag on 0 of \(destinations.count.formatted()) machines…"
        )
        Task {
            defer { store.endBulkActivity(activity) }
            for (index, destination) in destinations.enumerated() {
                let (host, client) = destination
                guard hosts.host(host.id) == host else { continue }
                store.updateBulkActivity(
                    activity,
                    "Renaming tag on \((index + 1).formatted()) of \(destinations.count.formatted()) machines…"
                )
                do {
                    _ = try await client.renameTag(name, to: clean)
                    guard hosts.host(host.id) == host else { continue }
                    hosts.succeeded(on: host.id)
                } catch {
                    guard hosts.host(host.id) == host else { continue }
                    // A machine that has never seen the tag answers 404, which
                    // is not a failure of the rename -- it is a machine with
                    // nothing to rename.
                    if let mold = error as? MoldClientError,
                       case let .http(status, _, _) = mold, status == 404 { continue }
                    hosts.report(error, on: host.id, doing: "rename the tag “\(name)”")
                }
            }
            store.updateBulkActivity(activity, "Refreshing tags…")
            await reload(destinations, in: store)
        }
    }

    /// Takes a tag off every print on every machine.
    ///
    /// NOT undoable, and the caller asks first. Putting it back would mean
    /// knowing which prints carried it, and by the time the answer came back
    /// the machines had already forgotten.
    func delete(_ name: String, in store: LibraryStore) {
        retag(name, to: nil, in: store)
        // A tag that no longer exists cannot be renamed back to, and an undo
        // stack that offers it is offering a lie.
        store.undo.forget()

        let hosts = store.hosts
        let destinations: [(host: MoldHost, client: any MoldBackend)] = hosts.hosts.map {
            ($0, hosts.backend(for: $0))
        }
        let activity = store.beginBulkActivity(
            "Deleting tag on 0 of \(destinations.count.formatted()) machines…"
        )
        Task {
            defer { store.endBulkActivity(activity) }
            for (index, destination) in destinations.enumerated() {
                let (host, client) = destination
                guard hosts.host(host.id) == host else { continue }
                store.updateBulkActivity(
                    activity,
                    "Deleting tag on \((index + 1).formatted()) of \(destinations.count.formatted()) machines…"
                )
                do {
                    try await client.deleteTag(name)
                    guard hosts.host(host.id) == host else { continue }
                    hosts.succeeded(on: host.id)
                } catch {
                    guard hosts.host(host.id) == host else { continue }
                    hosts.report(error, on: host.id, doing: "delete the tag “\(name)”")
                }
            }
            store.updateBulkActivity(activity, "Refreshing tags…")
            await reload(destinations, in: store)
        }
    }

    /// Rewrites or removes a tag on every print on screen. What the rewrite
    /// IS belongs to `TagRewrite`; this is which rows it runs over.
    private func retag(_ name: String, to replacement: String?, in store: LibraryStore) {
        for (hostID, list) in store.perHost {
            store.perHost[hostID] = TagRewrite.applied(to: list, name: name,
                                                       replacement: replacement)
        }
        store.rebuild()
    }

    private func reload(
        _ destinations: [(host: MoldHost, client: any MoldBackend)], in store: LibraryStore
    ) async {
        store.etags.removeAll()
        for (host, client) in destinations {
            guard store.hosts.host(host.id) == host else { continue }
            do {
                let tags = try await client.tags()
                guard store.hosts.host(host.id) == host else { continue }
                perHost[host.id] = tags
                // Scoped to its own verb: this is a passive refresh that runs
                // after every rename and delete, and must not silently clear
                // a failure THAT action just reported.
                store.hosts.succeeded(on: host.id, doing: "read its tags")
            } catch {
                guard store.hosts.host(host.id) == host else { continue }
                store.hosts.report(error, on: host.id, doing: "read its tags")
            }
        }
    }
}
