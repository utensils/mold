import Foundation
import MoldClient

// Copy a remote picture into This Mac's real gallery. Each original remains
// owned by its remote host, and a subsequent remote trash action names only
// that remote row.
@MainActor
extension LibraryStore {
    /// Picture formats the server's gallery import route accepts. A GLB or a
    /// clip in a mixed selection is left alone, rather than becoming a failed
    /// picture with a misleading count.
    static func canSaveLocally(_ entry: LibraryEntry) -> Bool {
        guard entry.hostID != MoldEngine.localHostID, entry.print.kind == .picture else {
            return false
        }
        let suffix = URL(fileURLWithPath: entry.print.filename).pathExtension.lowercased()
        return ["png", "jpg", "jpeg", "webp"].contains(suffix)
    }

    func saveLocally(_ selection: [LibraryEntry]) async {
        guard localSaveProgress == nil else { return }
        let targets = selection.filter(Self.canSaveLocally)
        guard !targets.isEmpty else { return }
        guard let local = hosts.host(MoldEngine.localHostID), hosts.isUp(local) else {
            localSaveReport = "Start This Mac’s engine to save remote pictures in its Library."
            localSaveAlertPresented = true
            return
        }

        var saved = 0
        var failures: [String] = []
        var freshPrints: [MoldHost.ID: [String: GalleryPrint]] = [:]
        localSaveProgress = "Saving 0 of \(targets.count)…"
        defer { localSaveProgress = nil }
        for (index, entry) in targets.enumerated() {
            localSaveProgress = "Saving \(index + 1) of \(targets.count)…"
            guard let source = hosts.backend(for: entry.hostID),
                  hosts.host(MoldEngine.localHostID) == local else {
                failures.append("\(entry.print.filename): its machine is no longer available")
                continue
            }
            do {
                if freshPrints[entry.hostID] == nil {
                    guard case let .fresh(prints, _) = try await source.gallery(etag: nil) else {
                        throw MoldClientError.malformedResponse
                    }
                    freshPrints[entry.hostID] = Dictionary(
                        prints.map { ($0.filename, $0) }, uniquingKeysWith: { first, _ in first })
                }
                guard let print = freshPrints[entry.hostID]?[entry.print.filename],
                      !(source is HTTPBackend) || print.rawMetadataAvailable else {
                    failures.append("\(entry.print.filename): its recipe is no longer available")
                    continue
                }
                let bytes = try await source.media(entry.print.filename, trashed: false)
                let bounded = try ResponseCeiling.checked(bytes, ceiling: ResponseCeiling.media,
                                                          what: "that print")
                guard hosts.host(MoldEngine.localHostID) == local else {
                    failures.append("\(entry.print.filename): This Mac’s engine changed")
                    continue
                }
                let destination = hosts.backend(for: local)
                let item = GalleryImport(mirroring: print, file: bounded)
                _ = try await destination.importPrint(item, as: entry.print.filename)
                saved += 1
            } catch {
                failures.append("\(entry.print.filename): \(error.localizedDescription)")
            }
        }
        if saved > 0 { await refresh(on: MoldEngine.localHostID) }
        let noun = saved == 1 ? "picture" : "pictures"
        localSaveReport = "Saved \(saved) of \(targets.count) \(noun) to This Mac’s Library."
        if !failures.isEmpty { localSaveReport += "\n\n" + failures.joined(separator: "\n") }
        let skipped = selection.count - targets.count
        if skipped > 0 {
            localSaveReport += "\n\nSkipped \(skipped) local or unsupported prints."
        }
        localSaveAlertPresented = true
    }
}
