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
        case tags([LibraryEntry])
        case newCollection([LibraryEntry])
        case rename(LibraryEntry)

        var id: String {
            switch self {
            case let .share(urls): "share-\(urls.count)"
            case let .tags(entries): "tags-\(entries.count)"
            case let .newCollection(entries): "collection-\(entries.count)"
            case let .rename(entry): "rename-\(entry.id.filename)"
            }
        }
    }

    var sheet: Sheet?
    /// A one-line result a person should see ("Saved to Photos"), shown in
    /// place and cleared on the next action -- never a toast.
    var status: String?
    var busy = false

    @ObservationIgnored let hosts: HostStore
    init(hosts: HostStore) { self.hosts = hosts }

    /// The prints' own files, fetched once and handed to the share sheet.
    func share(_ entries: [LibraryEntry]) {
        Task {
            guard let urls = await files(for: entries), !urls.isEmpty else { return }
            sheet = .share(urls)
        }
    }

    /// Stills and clips into Photos (add-only permission). A 3-D object has
    /// no place in Photos, so it is skipped and said so.
    func saveToPhotos(_ entries: [LibraryEntry]) {
        Task {
            let saveable = entries.filter { $0.print.kind != .mesh }
            guard let urls = await files(for: saveable), !urls.isEmpty else { return }
            let allowed = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
            guard allowed == .authorized || allowed == .limited else {
                status = String(localized: "Mold Studio needs permission to add to Photos. Allow it in Settings.")
                return
            }
            do {
                try await PHPhotoLibrary.shared().performChanges {
                    for (url, entry) in zip(urls, saveable) {
                        let request = PHAssetCreationRequest.forAsset()
                        request.addResource(with: entry.print.isVideo ? .video : .photo, fileURL: url, options: nil)
                    }
                }
                status = saveable.count == 1 ? String(localized: "Saved to Photos.")
                    : String(localized: "Saved \(saveable.count) prints to Photos.")
            } catch {
                status = String(localized: "Photos couldn't save that: \(error.localizedDescription)")
            }
        }
    }

    /// The still itself onto the pasteboard.
    func copy(_ entry: LibraryEntry) {
        Task {
            guard let url = await files(for: [entry])?.first,
                  let image = UIImage(contentsOfFile: url.path(percentEncoded: false)) else { return }
            UIPasteboard.general.image = image
            status = String(localized: "Copied.")
        }
    }

    private func files(for entries: [LibraryEntry]) async -> [URL]? {
        busy = true
        defer { busy = false }
        status = nil
        var urls: [URL] = []
        for entry in entries {
            guard let host = hosts.host(entry.hostID) else { continue }
            do {
                urls.append(try await hosts.backend(for: host).mediaFile(entry.print.filename,
                                                                          trashed: entry.print.trashedAt != nil))
            } catch {
                hosts.report(host, doing: String(localized: "send \(entry.print.displayName)"), error)
                return nil
            }
        }
        return urls
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
