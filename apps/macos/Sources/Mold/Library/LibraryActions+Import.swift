import AppKit
import ImageIO
import MoldClient
import UniformTypeIdentifiers

// Putting a file from this Mac into a machine's library.
extension LibraryActions {

    /// Asks for files, then asks which machine, then sends them.
    ///
    /// A machine, not "the" machine: a print belongs to the one that holds it,
    /// and with a fleet there is no obvious default. With exactly one reachable
    /// machine the question answers itself and is not asked.
    func importFiles(into host: MoldHost) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.image, .movie, .threeDContent]
        panel.prompt = "Import"
        panel.message = "Add to \(host.name)"

        Task {
            guard await panel.begin() == .OK else { return }
            await send(panel.urls, to: host)
            await reload()
        }
    }

    private func send(_ urls: [URL], to host: MoldHost) async {
        guard let client = hosts.backend(for: host.id) else { return }
        for url in urls {
            guard let data = try? Data(contentsOf: url) else { continue }
            // The file's own date, so an old picture lands where it belongs in
            // a day-sectioned timeline instead of at the top of today.
            let made = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
                .contentModificationDate
            let size = Self.pixelSize(of: url)
            let item = GalleryImport(importing: data, named: url.lastPathComponent,
                                     version: version(of: host),
                                     width: size.width, height: size.height, madeAt: made)
            do {
                _ = try await client.importPrint(item, as: url.lastPathComponent)
                hosts.succeeded(on: host.id)
            } catch {
                hosts.report(error, on: host.id, doing: "import “\(url.lastPathComponent)”")
                return
            }
        }
    }

    /// The picture's real shape, read from the file's own header rather than
    /// by decoding it -- an import may be a 60 MB PNG and nothing here needs
    /// the pixels. A clip or a mesh has no such header, and zero is the honest
    /// answer.
    private static func pixelSize(of url: URL) -> (width: Int, height: Int) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
                  as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int
        else { return (0, 0) }
        return (width, height)
    }

    /// The receiving machine's own version: it is what is creating the row.
    private func version(of host: MoldHost) -> String {
        if case let .up(status) = hosts.reachability(of: host) { return status.version }
        return "0"
    }
}
