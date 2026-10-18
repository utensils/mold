import AVFoundation
import Foundation

public enum VideoPlaybackEndpoint {
    /// Presentation time of the final visible video sample, independent of audio
    /// duration, frame-rate guesses and the media timeline's MP4 edit offsets.
    @MainActor public static func resolve(in asset: AVAsset) async throws -> CMTime {
        try Task.checkCancellation()
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let range = try await track.load(.timeRange)
        guard range.start.isNumeric, range.duration.isNumeric, range.duration > .zero else {
            throw CocoaError(.fileReadCorruptFile)
        }
        // AVFoundation maps source/decode timestamps through edit lists and
        // returns the actual presentation timestamp, including for VFR media.
        // Asking at the VIDEO endpoint also preserves legitimate trailing audio.
        let request = EndpointFrameRequest(asset: asset)
        let time = try await withTaskCancellationHandler {
            try await request.time(at: range.end)
        } onCancel: {
            Task { @MainActor in request.cancel() }
        }
        try Task.checkCancellation()
        guard time.isNumeric, time >= range.start,
              time < range.end else { throw CocoaError(.fileReadCorruptFile) }
        return time
    }
}

/// Keep the non-Sendable AVFoundation generator inside one actor while its
/// asynchronous decode runs; cancellation never shares it between executors.
@MainActor private final class EndpointFrameRequest {
    private let generator: AVAssetImageGenerator
    init(asset: AVAsset) {
        generator = AVAssetImageGenerator(asset: asset)
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
    }
    func time(at endpoint: CMTime) async throws -> CMTime {
        try Task.checkCancellation()
        return try await withCheckedThrowingContinuation { continuation in
            generator.generateCGImagesAsynchronously(forTimes: [NSValue(time: endpoint)]) { _, image, time, result, error in
                if result == .succeeded, image != nil { continuation.resume(returning: time) }
                else if result == .cancelled { continuation.resume(throwing: CancellationError()) }
                else { continuation.resume(throwing: error ?? CocoaError(.fileReadCorruptFile)) }
            }
        }
    }
    func cancel() { generator.cancelAllCGImageGeneration() }
}
