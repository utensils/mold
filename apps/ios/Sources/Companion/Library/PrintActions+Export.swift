import Foundation
import MoldClient

extension PrintActions {
    func openExport(_ entry: LibraryEntry) {
        guard !busy, MediaExportKind(filename: entry.print.filename, trashed: entry.print.trashedAt != nil) != nil else { return }
        status = nil
        sheet = .export(MediaExportSession(entry: entry, actions: self))
    }
    /// Only the dismissed presentation releases its files. Options own none.
    func presentationDismissed() {
        let dismissed = presentedSheet
        presentedSheet = nil
        switch dismissed {
        case let .share(urls), let .files(urls):
            Self.removeFiles(urls)
            sharedFiles.removeAll { urls.contains($0) }
        case let .export(session): session.cancel()
        default: break
        }
        if let pending = pendingDelivery {
            pendingDelivery = nil
            sheet = pending
        }
    }
    func cancelExports() {
        if let operation = fileExportTask {
            operation.cancel(); fileExportTask = nil; activeExportID = nil; busy = false
        }
        if case let .export(session) = sheet { session.cancel() }
        if case let .share(urls) = pendingDelivery { Self.removeFiles(urls) }
        if case let .files(urls) = pendingDelivery { Self.removeFiles(urls) }
        pendingDelivery = nil
    }
    func deliverOriginal(_ entry: LibraryEntry, destination: ExportDestination) {
        guard !busy else { return }
        busy = true; status = nil
        let operation = UUID(); activeExportID = operation
        fileExportTask = Task {
            defer { finishFileExport(operation) }
            guard !Task.isCancelled else { return }
            guard let urls = await downloadedFiles(for: [entry]), let url = urls.first else { return }
            guard !Task.isCancelled else { Self.removeFiles(urls); return }
            deliver(url, destination: destination)
        }
    }
    func deliverAsset(_ asset: GenerationAsset, entry: LibraryEntry, destination: ExportDestination) {
        guard !busy else { return }
        busy = true; status = nil
        let operation = UUID(); activeExportID = operation
        fileExportTask = Task {
            defer { finishFileExport(operation) }
            var staged: URL?
            defer { if let staged { Self.removeFiles([staged]) } }
            do {
                try Task.checkCancellation()
                guard let backend = hosts.backend(for: entry.hostID) else { throw MoldClientError.malformedResponse }
                let bytes = try await backend.generationAsset(entry.print.filename, assetID: asset.assetId)
                try Task.checkCancellation()
                let url = try ExportFiles.stage(bytes, filename: asset.displayName, asset: asset)
                staged = url
                try Task.checkCancellation()
                deliver(url, destination: destination)
                staged = nil
            } catch { if !Task.isCancelled { status = "Couldn't export that file: \(error.localizedDescription)" } }
        }
    }
    private func finishFileExport(_ operation: UUID) {
        guard activeExportID == operation else { return }
        activeExportID = nil; busy = false; fileExportTask = nil
    }
    private func deliver(_ url: URL, destination: ExportDestination) {
        if destination == .folder {
            defer { Self.removeFiles([url]) }
            do {
                let saved = try ExportFiles.saveToFolder(url)
                status = "Saved to Files ▸ Mold ▸ \(saved.lastPathComponent)."
            } catch { status = "Couldn't save that file: \(error.localizedDescription)" }
        } else { sheet = destination == .files ? .files([url]) : .share([url]) }
    }
}
