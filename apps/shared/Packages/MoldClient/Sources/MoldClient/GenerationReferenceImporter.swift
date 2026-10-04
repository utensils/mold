import Foundation
import AVFoundation
import CryptoKit
import ImageIO
import UniformTypeIdentifiers

/// Probes decoded content; duration arithmetic never supplies exact frame/sample counts.
public nonisolated enum GenerationReferenceImporter {
    public static func image(_ picture: ImportedPicture, role: GenerationImageReferenceRole? = nil) throws -> GenerationReference {
        let picture = try PictureImport.conform(picture.data, name: picture.name,
            accepting: [UTType.png.identifier, UTType.jpeg.identifier])
        guard let source = CGImageSourceCreateWithData(picture.data as CFData, nil),
              let type = CGImageSourceGetType(source), let ut = UTType(type as String),
              let mime = ut.preferredMIMEType,
              let dimensions = ReferenceCanvas.uprightPixels(ofBase64: picture.encoded) else {
            throw ReferenceImportError("The selected file is not a readable image.")
        }
        return GenerationReference(kind: role == nil ? "image" : "named_image",
            media: .init(authority: "inline", data: picture.encoded), mimeType: mime,
            provenance: .init(name: picture.name, sha256: digest(picture.data)),
            width: dimensions.width, height: dimensions.height, role: role)
    }
    public static func load(url: URL, role: GenerationImageReferenceRole? = nil) async throws -> GenerationReference {
        let access = url.startAccessingSecurityScopedResource()
        defer { if access { url.stopAccessingSecurityScopedResource() } }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count <= 512 * 1024 * 1024 else { throw ReferenceImportError("This reference exceeds the 512 MiB import limit.") }
        if let source = CGImageSourceCreateWithData(data as CFData, nil),
           let identifier = CGImageSourceGetType(source),
           let type = UTType(identifier as String), type.conforms(to: .image),
           CGImageSourceGetCount(source) > 0 {
            let picture = try PictureImport.conform(data, name: url.lastPathComponent, accepting: [UTType.png.identifier, UTType.jpeg.identifier])
            return try image(picture, role: role)
        }
        guard role == nil else { throw ReferenceImportError("Camera views must be image files.") }
        let prefix = Array(data.prefix(12))
        let isWav = prefix.count >= 12 && String(bytes: prefix[0..<4], encoding: .ascii) == "RIFF" &&
            String(bytes: prefix[8..<12], encoding: .ascii) == "WAVE"
        let isMp4 = prefix.count >= 12 && String(bytes: prefix[4..<8], encoding: .ascii) == "ftyp"
        guard isWav || isMp4 else { throw ReferenceImportError("Use PCM WAV audio or an MP4 video reference.") }
        let asset = AVURLAsset(url: url)
        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        guard !videoTracks.isEmpty || !audioTracks.isEmpty else { throw ReferenceImportError("Select an image, video or audio file.") }
        guard (videoTracks.isEmpty && isWav) || (!videoTracks.isEmpty && isMp4) else {
            throw ReferenceImportError("Use PCM WAV audio or an MP4 video reference.")
        }
        let mime = videoTracks.isEmpty ? "audio/wav" : "video/mp4"
        var reference = GenerationReference(kind: videoTracks.isEmpty ? "audio" : "video",
            media: .init(authority: "inline", data: data.base64EncodedString()), mimeType: mime,
            provenance: .init(name: url.lastPathComponent, sha256: digest(data)))
        if let video = videoTracks.first {
            let formats = try await video.load(.formatDescriptions)
            guard !formats.isEmpty, formats.allSatisfy({
                CMFormatDescriptionGetMediaSubType($0) == kCMVideoCodecType_H264
            }) else { throw ReferenceImportError("Video references require H.264/AVC in an MP4 file. Convert HEVC video before attaching it.") }
            let size = try await video.load(.naturalSize)
            // Server MP4 probing prices the encoded track geometry, before display rotation.
            reference.width = Int(abs(size.width).rounded())
            reference.height = Int(abs(size.height).rounded())
            let probe = try decode(asset: asset, track: video, audio: false)
            reference.frameCount = probe.count
            // Some decoders omit per-sample duration on the final video buffer.
            // The track duration includes that frame; frame_count remains decoded.
            let trackRange = try await video.load(.timeRange)
            let durationMs = Int((CMTimeGetSeconds(trackRange.duration) * 1000).rounded())
            guard durationMs > 0 else { throw ReferenceImportError("This video has no decoded duration.") }
            reference.durationMs = durationMs
            reference.fps = Double(probe.count) * 1000 / Double(durationMs)
        }
        if let audio = audioTracks.first {
            if videoTracks.isEmpty {
                let formats = try await audio.load(.formatDescriptions)
                guard formats.allSatisfy({ description in
                    CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee.mFormatID == kAudioFormatLinearPCM
                }) else { throw ReferenceImportError("Audio references require an uncompressed PCM WAV file.") }
            }
            let probe = try decode(asset: asset, track: audio, audio: true)
            if videoTracks.isEmpty {
                guard (1...2).contains(probe.channels) else { throw ReferenceImportError("Audio references must be mono or stereo.") }
                reference.durationMs = probe.durationMs; reference.sampleRate = probe.rate
                reference.channels = probe.channels; reference.sampleCount = probe.count
            } else {
                reference.hasAudio = true; reference.audioDurationMs = probe.durationMs
                reference.audioSampleRate = probe.rate; reference.audioChannels = probe.channels
                reference.audioSampleCount = probe.count
            }
        }
        return reference
    }
    private static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    private struct Probe { var count = 0; var durationMs = 0; var rate = 0; var channels = 0 }
    private static func decode(asset: AVAsset, track: AVAssetTrack, audio: Bool) throws -> Probe {
        let reader = try AVAssetReader(asset: asset)
        let settings: [String: Any] = audio ? [AVFormatIDKey: kAudioFormatLinearPCM,
            AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false] :
            [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw ReferenceImportError("This media cannot be decoded.") }
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? ReferenceImportError("Media decoding failed.") }
        var probe = Probe()
        var start: Double?; var end = 0.0
        while let buffer = output.copyNextSampleBuffer() {
            if Task.isCancelled { reader.cancelReading(); throw CancellationError() }
            let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(buffer))
            let duration = CMTimeGetSeconds(CMSampleBufferGetDuration(buffer))
            probe.count += CMSampleBufferGetNumSamples(buffer)
            if timestamp.isFinite { start = min(start ?? timestamp, timestamp); end = max(end, timestamp + (duration.isFinite ? duration : 0)) }
            if audio, let format = CMSampleBufferGetFormatDescription(buffer),
               let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format) {
                let rate = Int(asbd.pointee.mSampleRate.rounded())
                let channels = Int(asbd.pointee.mChannelsPerFrame)
                if probe.rate != 0 && (probe.rate != rate || probe.channels != channels) { throw ReferenceImportError("Changing audio formats are unsupported.") }
                probe.rate = rate; probe.channels = channels
            }
        }
        guard reader.status == .completed, probe.count > 0 else { throw reader.error ?? ReferenceImportError("Media decoding failed.") }
        probe.durationMs = audio && probe.rate > 0
            ? Int((Double(probe.count) * 1000 / Double(probe.rate)).rounded())
            : Int(((end - (start ?? 0)) * 1000).rounded())
        guard probe.durationMs > 0 else { throw ReferenceImportError("This media has no decoded duration.") }
        return probe
    }
}
public struct ReferenceImportError: LocalizedError, Sendable {
    public var errorDescription: String? { message }
    private let message: String
    public init(_ message: String) { self.message = message }
}
