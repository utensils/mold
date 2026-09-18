import AppKit
import ImageIO
import MoldClient
import UniformTypeIdentifiers

/// Files from this Mac on their way into a machine's library.
///
/// Its own object rather than more of `LibraryActions`: everything else there
/// is an action on prints the library already holds, and this is a batch with
/// a policy of its own -- what one unreadable file in the middle of ten means,
/// what a machine refusing means, and the single line said about the lot of it
/// once every import that could happen has. It takes the machines and the
/// library it is filling, the arrangement `LibraryMutations` has with
/// `LibraryStore`.
@MainActor
struct PrintImport {
    let hosts: HostStore
    let library: LibraryStore

    /// Asks for files, then asks which machine, then sends them.
    ///
    /// A machine, not "the" machine: a print belongs to the one that holds it,
    /// and with a fleet there is no obvious default. With exactly one reachable
    /// machine the question answers itself and is not asked.
    func chooseFiles(for host: MoldHost) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.image, .movie, .threeDContent]
        panel.prompt = "Import"
        panel.message = "Add to \(host.name)"

        Task {
            guard await panel.begin() == .OK else { return }
            await send(panel.urls, to: host)
            await library.reload()
        }
    }

    /// The batch itself. `LibraryImportTests` sends one without an open panel,
    /// which is the only way to pin what a batch does with one bad file in the
    /// middle of it.
    func send(_ urls: [URL], to host: MoldHost) async {
        guard let client = hosts.backend(for: host.id) else { return }
        /// The files this Mac could not read, reported ONCE when the batch is
        /// done. Not per file: every successful import calls
        /// `hosts.succeeded(on:)`, which clears that machine's failures, so a
        /// report made mid-loop is wiped by the next file that works -- ten
        /// chosen, one unreadable, nothing said.
        var unreadable: [(name: String, error: Error)] = []
        for url in urls {
            let data: Data
            do {
                // Off the main actor, like every other file this app reads:
                // an import is routinely a 60 MB PNG or a clip, and this
                // method is MainActor-isolated by default.
                data = try await MediaImport.bytes(of: url)
            } catch {
                // CONTINUE, not return: a file this Mac cannot read is about
                // THAT FILE, and abandoning the other nine is worse than the
                // silence it replaced. A refused upload below is about the
                // MACHINE, which is why that one stops.
                unreadable.append((url.lastPathComponent, error))
                continue
            }
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
        report(unreadable, to: host)
    }

    /// One line for the whole batch, once every import that could happen has.
    private func report(_ unreadable: [(name: String, error: Error)], to host: MoldHost) {
        guard let first = unreadable.first else { return }
        let verb = unreadable.count == 1
            ? "import “\(first.name)”"
            : "import \(unreadable.count) of those files"
        hosts.report(first.error, on: host.id, doing: verb)
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
