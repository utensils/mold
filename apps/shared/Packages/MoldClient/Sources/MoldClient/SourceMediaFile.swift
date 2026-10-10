import Foundation

/// A private, short-lived original for one open viewer, never an offline cache.
public final class SourceMediaFile: @unchecked Sendable {
    public static let previewLimit = 128 * 1024 * 1024
    public let url: URL
    private let directory: URL
    init(extension suffix: String) throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("mold-source-view-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false,
                                               attributes: [.posixPermissions: 0o700])
        url = directory.appendingPathComponent("source").appendingPathExtension(suffix)
        guard FileManager.default.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600]) else {
            try? FileManager.default.removeItem(at: directory)
            throw CocoaError(.fileWriteUnknown)
        }
    }
    deinit { try? FileManager.default.removeItem(at: directory) }
    public static func checkPreviewSize(_ size: Int) throws {
        guard size <= previewLimit else {
            throw MoldClientError.http(status: 413, code: nil,
                message: "This original is larger than the 128 MB viewer limit. You can still reuse it with a compatible model.")
        }
    }
}
