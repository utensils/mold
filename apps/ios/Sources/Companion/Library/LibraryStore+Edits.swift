import Foundation
import MoldClient

// Changing prints: favourite, tag, title, file -- on EVERY copy of a print, on
// every machine that holds one, through the shared outbox (`MutationOutbox`):
// applied on screen at once, sent per machine in order, retried 1s/2s/4s with
// the same operation id so a retry cannot double-apply, and given up only with
// a sentence and a re-read of that machine so the grid never lies.
extension LibraryStore {
    func apply(_ change: PrintChange, to entries: [LibraryEntry], undoable: Bool = true) {
        let copies = entries.flatMap(\.everyCopy)
        let edit = PrintEdit.plan(change, over: copies, collectionIDs: collectionIDs(for: change))
        guard !edit.isEmpty else { return }
        if undoable { lastEdit = edit }
        show(edit)
        let queued = outbox.enqueue(edit)
        for host in Set(queued.map(\.host)) { drain(host) }
    }

    /// Puts back the most recent change (shake, or ⌘Z on iPad).
    func undo() {
        guard let edit = lastEdit else { return }
        lastEdit = nil
        show(edit.inverse)
        let queued = outbox.enqueue(edit.inverse)
        for host in Set(queued.map(\.host)) { drain(host) }
    }

    var undoName: String? { lastEdit?.actionName }

    private func collectionIDs(for change: PrintChange) -> [MoldHost.ID: String] {
        guard case let .collection(_, slug, _) = change else { return [:] }
        return shelves.first { $0.slug == slug }?.hosts ?? [:]
    }

    private func show(_ edit: PrintEdit) {
        let ids = collectionIDs(for: edit.change)
        for (host, filenames) in edit.targets {
            let wanted = Set(filenames)
            updateLive(host) { prints in
                prints.map { print in
                    guard wanted.contains(print.filename) else { return print }
                    var mutable = GalleryPrint.Mutable(print)
                    edit.change.applied(to: &mutable, collectionID: ids[host])
                    return mutable.build()
                }
            }
        }
        rebuildNow()
    }

    private func drain(_ id: MoldHost.ID) {
        guard draining.insert(id).inserted else { return }
        Task {
            defer { draining.remove(id) }
            while true {
                switch outbox.next(for: id) {
                case .idle:
                    return
                case let .send(entry):
                    await send(entry, to: id)
                case let .wait(delay, entry):
                    try? await Task.sleep(for: delay)
                    await send(entry, to: id)
                case let .giveUp(entry, _):
                    if let host = hosts.host(id) {
                        hosts.report(host, doing: entry.change.verb, lastFailure[id] ?? CancellationError())
                    }
                    await reload(id)
                }
            }
        }
    }

    private func send(_ entry: MutationOutbox.Entry, to id: MoldHost.ID) async {
        guard let client = hosts.backend(for: id) else { _ = outbox.failed(entry.id); return }
        do {
            switch entry.wire {
            case let .patch(patch, filenames):
                for filename in filenames { try await client.patch(filename, with: patch) }
            case let .mutate(mutation):
                try await client.mutate(mutation)
            }
            outbox.succeeded(entry.id)
        } catch {
            lastFailure[id] = error
            outbox.retry(entry.id)
        }
    }

    // MARK: - Recently Deleted

    /// Moves prints to Recently Deleted on every machine holding a copy; a
    /// machine without a trash deletes for good (the menu said so first).
    func trash(_ entries: [LibraryEntry]) async {
        await perHost(entries.flatMap(\.everyCopy), doing: String(localized: "move those prints to Recently Deleted")) {
            client, host, files in
            if self.hosts.capabilities[host]?.trashEnabled == true {
                try await client.trash(files)
            } else {
                try await client.deleteForever(files)
            }
        }
    }

    func putBack(_ entries: [LibraryEntry]) async {
        await perHost(entries.flatMap(\.everyCopy), doing: String(localized: "put those prints back")) {
            client, _, files in try await client.restoreFromTrash(files)
        }
    }

    func deleteImmediately(_ entries: [LibraryEntry]) async {
        await perHost(entries.flatMap(\.everyCopy), doing: String(localized: "delete those prints")) {
            client, _, files in try await client.deleteForever(files)
        }
    }

    func emptyTrash() async {
        await perHost(trashPool.flatMap(\.everyCopy), doing: String(localized: "empty Recently Deleted")) {
            client, _, _ in try await client.emptyTrash()
        }
    }

    private func perHost(_ copies: [LibraryEntry], doing verb: String,
                         _ act: @escaping (any MoldBackend, MoldHost.ID, [String]) async throws -> Void) async {
        let byHost = Dictionary(grouping: copies, by: \.hostID)
        for (id, entries) in byHost {
            guard let host = hosts.host(id) else { continue }
            do {
                try await act(hosts.backend(for: host), id, entries.map(\.print.filename))
            } catch {
                hosts.report(host, doing: verb, error)
            }
            await reload(id)
        }
    }
}
