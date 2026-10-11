import AVFoundation
import Foundation
import ImageIO

/// On-demand inspection of authenticated originals, never a catalog-wide probe.
/// Retain the caller-owned file for playback; this result creates no copy or cache.
public enum SourceMediaOriginal {
    public enum Kind: String, Sendable { case image, video, audio }
    public struct Inspection: Sendable {
        public let kind: Kind
        public let url: URL
        public let image: SourceMediaOriginalImage?
    }
    public enum InspectionError: LocalizedError, Equatable, Sendable {
        case unsupported, imageTooLarge, timedOut
        public var errorDescription: String? {
            switch self {
            case .unsupported: "This original uses an unsupported or damaged media format. Try another retained copy."
            case .imageTooLarge: "This image exceeds the viewer's safe dimension limit. You can still reuse the original."
            case .timedOut: "Inspecting this original took too long. Try again or choose another retained copy."
            }
        }
    }

    public static func inspect(_ file: SourceMediaFile) async throws -> Inspection {
        try Task.checkCancellation()
        let worker = Task.detached(priority: .utility) {
            try Task.checkCancellation()
            let size = try file.url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            try SourceMediaFile.checkPreviewSize(size)
            if try isImage(file.url) {
                do {
                    let image = try await SourceMediaOriginalImage.decode(file)
                    return Inspection(kind: .image, url: file.url, image: image)
                } catch is CancellationError { throw CancellationError() }
                catch { throw InspectionError.unsupported }
            }
            let playbackURL = try playableURL(file)
            let asset = AVURLAsset(url: playbackURL)
            do {
                let kind = try await withTaskCancellationHandler {
                    try await withTimeout(.seconds(5), cancel: { asset.cancelLoading() }) {
                        guard try await asset.load(.isPlayable) else { throw InspectionError.unsupported }
                        if !(try await asset.loadTracks(withMediaType: .video)).isEmpty { return Kind.video }
                        if !(try await asset.loadTracks(withMediaType: .audio)).isEmpty { return Kind.audio }
                        throw InspectionError.unsupported
                    }
                } onCancel: { asset.cancelLoading() }
                try Task.checkCancellation()
                return Inspection(kind: kind, url: playbackURL, image: nil)
            } catch {
                try Task.checkCancellation()
                if let error = error as? InspectionError { throw error }
                throw InspectionError.unsupported
            }
        }
        return try await withTaskCancellationHandler {
            let result = try await worker.value
            try Task.checkCancellation()
            return result
        } onCancel: { worker.cancel() }
    }

    /// AVFoundation requires a suitable container suffix even when its bytes are
    /// valid. Inspect a tiny prefix, then verify tracks with AVFoundation; magic
    /// alone never declares the media valid. A hard link avoids another 128 MB copy.
    private static func playableURL(_ file: SourceMediaFile) throws -> URL {
        let handle = try FileHandle(forReadingFrom: file.url)
        defer { try? handle.close() }
        let bytes = [UInt8](try handle.read(upToCount: 64) ?? Data())
        func marker(_ text: String, at offset: Int = 0) -> Bool {
            let expected = Array(text.utf8)
            guard bytes.count >= offset + expected.count else { return false }
            return Array(bytes[offset..<(offset + expected.count)]) == expected
        }
        let suffix: String
        if marker("RIFF"), marker("WAVE", at: 8) { suffix = "wav" }
        else if marker("ftyp", at: 4) { suffix = "mp4" }
        else if marker("moov", at: 4) || marker("mdat", at: 4) || marker("wide", at: 4) { suffix = "mov" }
        else if marker("fLaC") { suffix = "flac" }
        else if marker("OggS") { suffix = "ogg" }
        else if marker("FORM"), marker("AIFF", at: 8) || marker("AIFC", at: 8) { suffix = "aiff" }
        else if marker("ID3") { suffix = "mp3" }
        else if bytes.count >= 2, bytes[0] == 0xff, bytes[1] & 0xf6 == 0xf0 { suffix = "aac" }
        else if bytes.count >= 2, bytes[0] == 0xff, bytes[1] & 0xe0 == 0xe0 { suffix = "mp3" }
        else { throw InspectionError.unsupported }
        let url = file.url.deletingLastPathComponent().appendingPathComponent("playback.\(suffix)")
        // Re-inspection of the same immutable original can reuse its private link.
        if !FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.linkItem(at: file.url, to: url)
        }
        return url
    }

    private static func isImage(_ url: URL) throws -> Bool {
        guard let source = CGImageSourceCreateWithURL(url as CFURL,
            [kCGImageSourceShouldCache: false] as CFDictionary), CGImageSourceGetCount(source) > 0 else { return false }
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.doubleValue,
              let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.doubleValue,
              width > 0, height > 0 else { throw InspectionError.unsupported }
        // Check encoded geometry before asking ImageIO for a 4096-edge thumbnail.
        guard width <= 65_536, height <= 65_536, width * height <= 100_000_000 else {
            throw InspectionError.imageTooLarge
        }
        return true
    }

    /// The AVFoundation cancellation hook stops its underlying metadata load as
    /// well as the task. Structured children finish before the file can be removed.
    static func withTimeout<T: Sendable>(_ timeout: Duration, cancel: @escaping @Sendable () -> Void,
                                         operation: @escaping @Sendable () async throws -> T) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(for: timeout)
                throw InspectionError.timedOut
            }
            defer { group.cancelAll(); cancel() }
            return try await group.next()!
        }
    }
}

public extension RetainedSourceMedia.Member {
    /// An ephemeral presentation of this exact inspected original. Callers must
    /// bind it to the same owner/member and current authority, never promote an
    /// entire catalog group or use a previous selection's inspection result.
    func inspected(as kind: SourceMediaOriginal.Kind) -> Self {
        Self(memberId: memberId, role: role, displayName: displayName, sizeBytes: sizeBytes,
             label: label, mediaKind: kind.rawValue, position: position, frameIndex: frameIndex,
             imageRole: imageRole, contentSha256: contentSha256)
    }
}
