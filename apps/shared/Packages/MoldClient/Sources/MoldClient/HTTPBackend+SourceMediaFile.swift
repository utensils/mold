import Foundation

public extension HTTPBackend {
    func retainedSourceMediaFile(for filename: String, member: RetainedSourceMedia.Member) async throws -> SourceMediaFile {
        try SourceMediaFile.checkPreviewSize(member.sizeBytes)
        let path = retainedSourceMediaPath(filename) + "/\(escaped(member.memberId))"
        do { return try await sourceMediaFile(at: path + "/content", member: member) }
        catch MoldClientError.http(let status, _, _) where (status == 404 || status == 405) && member.role != "keyframes" {
            return try await sourceMediaFile(at: path, member: member)
        }
    }

    private func sourceMediaFile(at path: String, member: RetainedSourceMedia.Member) async throws -> SourceMediaFile {
        let (stream, response) = try await relayBytes(request(path))
        guard (200..<300).contains(response.statusCode) else {
            if response.statusCode == 401 { throw MoldClientError.unauthorized }
            throw MoldClientError.http(status: response.statusCode, code: nil, message: nil)
        }
        try SourceMediaFile.checkPreviewSize(Int(clamping: response.expectedContentLength))
        let proposed = (member.displayName as NSString).pathExtension.lowercased()
        let known = ["png", "jpg", "jpeg", "webp", "gif", "heic", "mp4", "mov", "m4v", "wav", "mp3", "m4a", "aac", "flac", "ogg"]
        let suffix = known.contains(proposed) ? proposed : (member.contentKind == "video" ? "mp4" : member.contentKind == "audio" ? "wav" : "png")
        let file = try SourceMediaFile(extension: suffix)
        let handle = try FileHandle(forWritingTo: file.url)
        defer { try? handle.close() }
        var chunk = Data(), count = 0
        chunk.reserveCapacity(65_536)
        for try await byte in stream {
            try Task.checkCancellation()
            count += 1
            try SourceMediaFile.checkPreviewSize(count)
            chunk.append(byte)
            if chunk.count == 65_536 { try handle.write(contentsOf: chunk); chunk.removeAll(keepingCapacity: true) }
        }
        try Task.checkCancellation()
        if !chunk.isEmpty { try handle.write(contentsOf: chunk) }
        return file
    }
}
