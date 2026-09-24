import Foundation
import MoldClient

private struct MirrorTarget: Sendable {
    let print: GalleryPrint
    let source: any MoldBackend
    let collections: [(slug: String, name: String)]
}

private func mirror(_ target: MirrorTarget, to destination: any MoldBackend) async -> (String?, String?) {
    do {
        let bytes = try await target.source.media(target.print.filename, trashed: false)
        let bounded = try ResponseCeiling.checked(bytes, ceiling: ResponseCeiling.media,
                                                  what: "that print")
        let item = GalleryImport(mirroring: target.print, file: bounded)
        return (try await destination.importPrint(item, as: target.print.filename), nil)
    } catch {
        return (nil, error.localizedDescription)
    }
}

@MainActor
extension LibraryStore {
    static func canSaveLocally(_ entry: LibraryEntry) -> Bool {
        guard entry.hostID != MoldEngine.localHostID, entry.print.kind == .picture else {
            return false
        }
        return ["png", "jpg", "jpeg", "webp"].contains(
            URL(fileURLWithPath: entry.print.filename).pathExtension.lowercased())
    }

    func saveLocally(_ selection: [LibraryEntry]) async {
        guard localSaveProgress == nil else { return }
        defer { localSaveTask = nil }
        let candidates = selection.filter(Self.canSaveLocally)
        guard !candidates.isEmpty else { return }
        guard let local = hosts.host(MoldEngine.localHostID), hosts.isUp(local) else {
            localSaveReport = "Start This Mac’s engine to save remote pictures in its Library."
            localSaveFailures = []
            localSaveAlertPresented = true
            return
        }
        localSaveProgress = "Preparing \(candidates.count.formatted()) pictures…"
        defer {
            localSaveProgress = nil
            localSaveStopRequested = false
        }

        var failures: [String] = []
        var sourcePrints: [MoldHost.ID: [String: GalleryPrint]] = [:]
        var sourceCollections: [MoldHost.ID: [String: Collection]] = [:]
        var hostErrors: [MoldHost.ID: String] = [:]
        for hostID in Set(candidates.map(\.hostID)) {
            guard let source = hosts.backend(for: hostID) else {
                hostErrors[hostID] = "its machine is no longer available"
                continue
            }
            do {
                guard case let .fresh(prints, _) = try await source.gallery(etag: nil) else {
                    throw MoldClientError.malformedResponse
                }
                sourcePrints[hostID] = Dictionary(prints.map { ($0.filename, $0) },
                                                  uniquingKeysWith: { first, _ in first })
                let selectedNames = Set(candidates.filter { $0.hostID == hostID }
                    .map { $0.print.filename })
                if prints.contains(where: { selectedNames.contains($0.filename)
                    && !$0.collectionList.isEmpty }) {
                    do {
                        sourceCollections[hostID] = Dictionary(
                            try await source.collections().map { ($0.id, $0) },
                            uniquingKeysWith: { first, _ in first })
                    } catch {
                        failures.append("Collections on \(hostID): \(error.localizedDescription)")
                    }
                }
            } catch {
                hostErrors[hostID] = error.localizedDescription
            }
        }

        var targets: [MirrorTarget] = []
        for entry in candidates {
            if let error = hostErrors[entry.hostID] {
                failures.append("\(entry.print.filename): \(error)")
                continue
            }
            guard let print = sourcePrints[entry.hostID]?[entry.print.filename],
                  let source = hosts.backend(for: entry.hostID) else {
                failures.append("\(entry.print.filename): its recipe is no longer available")
                continue
            }
            let collectionIDs = sourceCollections[entry.hostID] ?? [:]
            let names = print.collectionList.compactMap { id -> (slug: String, name: String)? in
                guard let collection = collectionIDs[id] else { return nil }
                return (collection.slug, collection.name)
            }
            if names.count != print.collectionList.count,
               sourceCollections[entry.hostID] != nil {
                failures.append("\(print.filename): a source collection is no longer available")
            }
            targets.append(MirrorTarget(print: print, source: source, collections: names))
        }

        let destination = hosts.backend(for: local)
        let localByName = Dictionary((perHost[local.id] ?? []).map {
            ($0.print.filename, $0.print)
        }, uniquingKeysWith: { first, _ in first })
        let localNames = Dictionary((collectionsPerHost[local.id] ?? []).map {
            ($0.slug, $0.name)
        }, uniquingKeysWith: { first, _ in first })
        var collectionNames: [String: String] = [:]
        var collectionFiles: [String: [String]] = [:]
        var alreadyLocal = 0
        var transferred = 0
        var completed = 0
        func record(_ target: MirrorTarget, as filename: String) {
            for collection in target.collections {
                let name = localNames[collection.slug] ?? collection.name
                if let previous = collectionNames[collection.slug], previous != name {
                    failures.append("Collection “\(collection.slug)” has different names on source machines; using “\(previous)”.")
                } else {
                    collectionNames[collection.slug] = name
                }
                collectionFiles[collection.slug, default: []].append(filename)
            }
        }

        var work: [(Int, MirrorTarget)] = []
        for (index, target) in targets.enumerated() {
            if Task.isCancelled || localSaveStopRequested { break }
            // An import preserves its source timestamp. Name, timestamp,
            // length and recipe identity let a retry avoid transferring it.
            if let existing = localByName[target.print.filename],
               existing.timestamp == target.print.timestamp,
               existing.sizeBytes != nil, existing.sizeBytes == target.print.sizeBytes,
               existing.metadata.model == target.print.metadata.model,
               existing.metadata.seed == target.print.metadata.seed {
                alreadyLocal += 1
                completed += 1
                record(target, as: existing.filename)
            } else {
                work.append((index, target))
            }
        }

        // Bound concurrency by image count and memory. Each HTTP upload is
        // file-backed, so it does not duplicate the downloaded Data in RAM.
        var next = 0
        await withTaskGroup(of: (Int, String?, String?).self) { group in
            for _ in 0..<min(3, work.count) {
                let (index, target) = work[next]
                next += 1
                group.addTask {
                    let (name, error) = await mirror(target, to: destination)
                    return (index, name, error)
                }
            }
            for await (index, name, error) in group {
                completed += 1
                if let name {
                    transferred += 1
                    record(targets[index], as: name)
                } else {
                    failures.append("\(targets[index].print.filename): \(error ?? "save failed")")
                }
                if completed % 10 == 0 || completed == targets.count {
                    localSaveProgress = "Saving \(completed.formatted()) of \(targets.count.formatted())…"
                }
                if next < work.count && !Task.isCancelled && !localSaveStopRequested {
                    let (index, target) = work[next]
                    next += 1
                    group.addTask {
                        let (name, error) = await mirror(target, to: destination)
                        return (index, name, error)
                    }
                }
            }
        }
        if Task.isCancelled || localSaveStopRequested {
            failures.append("Save stopped. Remaining pictures were not transferred.")
        }

        // A host resolves a collection name to its own slug and creates the
        // local copy once. Never send the source machine's collection ID.
        for slug in collectionFiles.keys.sorted() {
            guard let name = collectionNames[slug],
                  let filenames = collectionFiles[slug], !filenames.isEmpty else { continue }
            do {
                try await destination.mutate(GalleryBulkMutation(
                    filenames: Array(Set(filenames)).sorted(), addToCollection: .named(name)))
            } catch {
                failures.append("Collection “\(name)”: \(error.localizedDescription)")
            }
        }

        if transferred > 0 || !collectionFiles.isEmpty {
            if let collections = try? await destination.collections() {
                collectionsPerHost[local.id] = collections
            }
        }
        // End suppression before the final read. An unrelated client may
        // import while collections load; the read then catches it, and later
        // events apply normally.
        localSaveProgress = nil
        await refresh(on: local.id)
        let skipped = selection.count - candidates.count
        var summary = "Copied \(transferred) pictures to This Mac’s Library."
        if alreadyLocal > 0 { summary += " \(alreadyLocal) were already here." }
        let unsaved = candidates.count - transferred - alreadyLocal
        if unsaved > 0 { summary += " \(unsaved) were not copied." }
        if skipped > 0 { summary += " Skipped \(skipped) local or unsupported prints." }
        if !failures.isEmpty { summary += " \(failures.count) issues need attention." }
        localSaveReport = summary
        localSaveFailures = failures
        localSaveAlertPresented = true
    }
}
