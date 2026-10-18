import Foundation
import UniformTypeIdentifiers

public extension GenerationReferenceImporter {
    /// Only negotiated fresh typed videos bypass local canonical probing. Their original bytes
    /// remain immutable until the selected authority supplies exact facts and finalizes them.
    static func loadAuthoring(url: URL, capabilities: ReferenceUploadCapabilities?, apiKey: String?,
                              role: GenerationImageReferenceRole? = nil) async throws -> AuthoringReference {
        if role == nil, videoMime(for: url.pathExtension.lowercased()) != nil {
            try VideoNormalizationPolicy.requireValidAdvertisement(capabilities)
        }
        guard role == nil, let cap = VideoNormalizationPolicy.capability(capabilities, apiKey: apiKey),
              cap.acceptedExtensions?.contains(url.pathExtension.lowercased()) == true,
              let mime = videoMime(for: url.pathExtension.lowercased()),
              cap.acceptedMimeTypes?.contains(mime) == true else {
            return .canonical(try await load(url: url, role: role))
        }
        let access = url.startAccessingSecurityScopedResource()
        defer { if access { url.stopAccessingSecurityScopedResource() } }
        let maximum = min(cap.maxOriginalFileBytes!, 512 * 1024 * 1024)
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let declared = try handle.seekToEnd()
        guard declared > 0, declared <= UInt64(maximum) else {
            throw ReferenceImportError("This video exceeds the selected host's original-file limit or is empty.")
        }
        try handle.seek(toOffset: 0)
        var data = Data()
        while true {
            try Task.checkCancellation()
            let chunk = try handle.read(upToCount: min(1024 * 1024, maximum - data.count + 1)) ?? Data()
            if chunk.isEmpty { break }
            guard chunk.count <= maximum - data.count else {
                throw ReferenceImportError("This video exceeds the selected host's original-file limit.")
            }
            data.append(chunk)
        }
        try Task.checkCancellation()
        guard !data.isEmpty else { throw ReferenceImportError("This video is empty.") }
        return .freshVideo(.init(data: data, mimeType: mime, name: url.lastPathComponent))
    }

    static func authoringContentTypes(capabilities: ReferenceUploadCapabilities?, apiKey: String?) -> [UTType] {
        var types: [UTType] = [.image, .movie, .wav]
        if let cap = VideoNormalizationPolicy.capability(capabilities, apiKey: apiKey) {
            types += (cap.acceptedExtensions ?? []).compactMap { UTType(filenameExtension: $0) }
        }
        return Array(Set(types))
    }

    private static func videoMime(for ext: String) -> String? {
        switch ext {
        case "mp4": "video/mp4"
        case "mov": "video/quicktime"
        case "m4v": "video/x-m4v"
        case "webm": "video/webm"
        case "mkv": "video/x-matroska"
        default: nil
        }
    }
}
