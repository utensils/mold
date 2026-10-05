import Foundation
import MoldClient

extension PrintActions {
    /// Give each downloaded print its real filename: UIKit uses the extension
    /// to recognize a video, image or mesh instead of presenting a generic file.
    func files(for entries: [LibraryEntry]) async -> [URL]? {
        guard !busy else { return nil }
        busy = true
        defer { busy = false }
        return await downloadedFiles(for: entries)
    }

    /// The calling operation owns busy state and cancellation cleanup.
    func downloadedFiles(for entries: [LibraryEntry]) async -> [URL]? {
        status = nil
        var urls: [URL] = []
        for entry in entries {
            guard !Task.isCancelled else { Self.removeFiles(urls); return nil }
            guard let host = hosts.host(entry.hostID) else {
                Self.removeFiles(urls)
                hosts.report(entry.hostID, name: entry.hostName,
                             doing: String(localized: "send \(entry.print.displayName)"),
                             MoldClientError.unreachable(String(localized: "This machine was removed. Add it again under Machines.")))
                return nil
            }
            do {
                let downloaded = try await hosts.backend(for: host).mediaFile(entry.print.filename,
                                                                            trashed: entry.print.trashedAt != nil)
                defer { try? FileManager.default.removeItem(at: downloaded) }
                try Task.checkCancellation()
                let directory = FileManager.default.temporaryDirectory.appending(path: "mold-print-export-\(UUID())")
                guard let named = SafeFilename.url(entry.print.filename, in: directory) else {
                    throw MoldClientError.malformedResponse
                }
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                do {
                    try FileManager.default.moveItem(at: downloaded, to: named)
                } catch {
                    try? FileManager.default.removeItem(at: directory)
                    throw error
                }
                urls.append(named)
            } catch {
                Self.removeFiles(urls)
                guard !Task.isCancelled else { return nil }
                hosts.report(host, doing: String(localized: "send \(entry.print.displayName)"), error)
                return nil
            }
        }
        return urls
    }

    static func removeFiles(_ urls: [URL]) {
        for url in urls {
            let directory = url.deletingLastPathComponent()
            guard directory.lastPathComponent.hasPrefix("mold-print-export-") else { continue }
            try? FileManager.default.removeItem(at: directory)
        }
    }
}
