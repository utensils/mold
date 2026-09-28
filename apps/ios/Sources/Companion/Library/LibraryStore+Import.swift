import Foundation
import ImageIO
import MoldClient
import UniformTypeIdentifiers

extension LibraryStore {
    /// A picture from outside -- dropped on the grid, or from Share -- into
    /// one machine's Library. `false` (with the reason in the banner) when
    /// the machine is not answering or refuses it.
    @discardableResult
    func importPicture(_ data: Data, stem: String, taken: Date? = nil, to host: MoldHost) async -> Bool {
        guard case let .up(status) = hosts.reachability(of: host) else {
            hosts.report(host, doing: String(localized: "add that picture to its Library"), NotAnswering())
            return false
        }
        var width = 0, height = 0, suffix = "png"
        if let source = CGImageSourceCreateWithData(data as CFData, nil) {
            if let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] {
                width = properties[kCGImagePropertyPixelWidth] as? Int ?? 0
                height = properties[kCGImagePropertyPixelHeight] as? Int ?? 0
            }
            // The name says what the bytes are.
            if let type = CGImageSourceGetType(source).flatMap({ UTType($0 as String) }),
               let ext = type.preferredFilenameExtension { suffix = ext }
        }
        let name = "\(stem).\(suffix)"
        let upload = GalleryImport(prompt: "", model: "", width: width, height: height,
                                   version: status.version, file: data, timestamp: taken)
        do {
            _ = try await hosts.backend(for: host).importPrint(upload, as: name)
            await reload(host.id)
            return true
        } catch {
            hosts.report(host, doing: String(localized: "add that picture to its Library"), error)
            return false
        }
    }
}

private struct NotAnswering: LocalizedError {
    var errorDescription: String? { String(localized: "It isn't answering.") }
}
