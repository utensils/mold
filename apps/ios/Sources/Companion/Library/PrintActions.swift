import MoldClient
import Photos
import SwiftUI
import UIKit

/// What a print can do that needs more than one tap: share and save need the
/// file itself, tags and a new collection need a sheet. One per window, so a
/// menu in the grid, the selection bar and the viewer all go through the same
/// door and say the same words.
@Observable
final class PrintActions {
    enum Sheet: Identifiable {
        case share([URL])
        case files([URL])
        case export(MediaExportSession)
        case tags([LibraryEntry])
        case newCollection([LibraryEntry])
        case rename(LibraryEntry)

        var id: String {
            switch self {
            case let .share(urls): "share-\(urls.first?.path ?? "")"
            case let .files(urls): "files-\(urls.first?.path ?? "")"
            case let .export(session): "export-\(session.id)"
            case let .tags(entries): "tags-\(entries.count)"
            case let .newCollection(entries): "collection-\(entries.count)"
            case let .rename(entry): "rename-\(entry.id.filename)"
            }
        }
    }

    var sheet: Sheet?
    @ObservationIgnored var presentedSheet: Sheet?
    @ObservationIgnored var pendingDelivery: Sheet?
    /// A one-line result a person should see ("Saved to Photos"), shown in
    /// place and cleared on the next action -- never a toast.
    var status: String?
    var busy = false
    @ObservationIgnored var activeExportID: UUID?
    @ObservationIgnored var fileExportTask: Task<Void, Never>?
    var permissionRecovery: PermissionRecovery?

    @ObservationIgnored var sharedFiles: [URL] = []
    @ObservationIgnored let hosts: HostStore
    init(hosts: HostStore) { self.hosts = hosts }

    /// The prints' own files, fetched once and handed to the share sheet.
    func share(_ entries: [LibraryEntry]) {
        Task {
            guard let urls = await files(for: entries), !urls.isEmpty else { return }
            shareFinished()
            sharedFiles = urls
            sheet = .share(urls)
        }
    }

    /// Stills and clips into Photos (add-only permission). A 3-D object has
    /// no place in Photos, so it is skipped and said so.
    func saveToPhotos(_ entries: [LibraryEntry], interactive: Bool = true) {
        Task {
            status = nil
            let saveable = entries.filter { $0.print.kind != .mesh }
            guard !saveable.isEmpty else { return }
            let allowed = interactive ? await PhotosAccess.request() : PHPhotoLibrary.authorizationStatus(for: .addOnly)
            guard PhotosAccess.canSave(allowed) else {
                if interactive { permissionRecovery = PermissionRecovery.photos(allowed) }
                return
            }
            guard let urls = await files(for: saveable), !urls.isEmpty else { return }
            defer { Self.removeFiles(urls) }
            // Resolve app-isolated metadata before entering PhotoKit's queue.
            let resources = Self.photoResources(urls: urls, entries: saveable)
            do {
                try await PhotosWriter.save(resources)
                status = saveable.count == 1 ? String(localized: "Saved to Photos.")
                    : String(localized: "Saved \(saveable.count) prints to Photos.")
            } catch {
                if interactive, let recovery = PermissionRecovery.photos(PHPhotoLibrary.authorizationStatus(for: .addOnly)) {
                    permissionRecovery = recovery
                } else { status = String(localized: "Photos couldn't save that: \(error.localizedDescription)") }
            }
        }
    }

    static func photoResources(urls: [URL], entries: [LibraryEntry]) -> [PhotosWriter.Resource] {
        zip(urls, entries).map { url, entry in
            // Playback also calls animated GIF/WebP a clip; PhotoKit needs
            // their image resource, and only video containers use .video.
            let format = (entry.print.format ?? url.pathExtension).lowercased()
            return PhotosWriter.Resource(url: url, video: ["mp4", "mov", "m4v"].contains(format))
        }
    }

    /// The still itself onto the pasteboard.
    func copy(_ entry: LibraryEntry) {
        Task {
            guard let urls = await files(for: [entry]), let url = urls.first else { return }
            defer { Self.removeFiles(urls) }
            guard let image = UIImage(contentsOfFile: url.path(percentEncoded: false)) else { return }
            UIPasteboard.general.image = image
            status = String(localized: "Copied.")
        }
    }

    /// The share sheet retains its files until it is dismissed or completed.
    func shareFinished() {
        Self.removeFiles(sharedFiles)
        sharedFiles = []
    }

}

/// UIKit's share sheet, for files already on this device.
struct ShareSheet: UIViewControllerRepresentable {
    let items: [URL]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
