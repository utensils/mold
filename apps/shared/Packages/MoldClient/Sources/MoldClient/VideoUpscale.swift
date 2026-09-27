import Foundation

/// One durable framewise clip upscale, as the host reports it
/// (`crates/mold-core/src/video_upscale.rs:61-78`).
///
/// A clip upscale is not a queue row -- `/api/queue` never lists one. Its
/// frames ARE scheduler work (`video_upscale.rs:1271-1281`), so `/api/activity`
/// reports a `standalone_upscale` under a new uuid per frame; that row knows
/// neither which print it belongs to nor how far through the clip it is. This
/// job is the only thing that does, which is why the app polls it.
///
/// Everything beyond identity and progress is optional here. The wire carries
/// more (`scale_factor`, both media-fact blocks, the timestamps); nothing in
/// this app reads them, and decoding a field nobody uses is one more way for
/// a newer host to lose the whole listing.
public struct VideoUpscaleJob: Codable, Hashable, Sendable, Identifiable {
    public let id: String
    public let state: VideoUpscaleJobState
    /// What is being upscaled. Only a `library` source belongs to a print in
    /// this app's Library -- an `upload` handle is somebody else's session.
    public let source: VideoUpscaleSource?
    public let model: String
    public let completedFrames: Int
    public let totalFrames: Int
    /// The Library filename of the finished clip, once there is one.
    public let outputFilename: String?
    /// The host's own sentence about why it stopped.
    public let error: String?

    /// The library print this job is upscaling, or nil for an upload.
    public var libraryFilename: String? {
        guard case let .library(filename) = source else { return nil }
        return filename
    }
}

public enum VideoUpscaleJobState: String, OpenWireEnum {
    case queued, running, finalizing, paused, completed, failed, cancelled
    /// A state added after this build. Not polled and not recovered onto a
    /// print -- see `UpscalePlan`.
    case unknown

    /// Settled for good (`video_upscale.rs:39-42`). Everything else is a job
    /// that can still move, including a state this build does not know.
    public var isTerminal: Bool {
        switch self {
        case .completed, .failed, .cancelled: true
        case .queued, .running, .finalizing, .paused, .unknown: false
        }
    }
}

/// The tagged source union (`video_upscale.rs:9-16`).
public enum VideoUpscaleSource: Codable, Hashable, Sendable {
    case library(filename: String)
    case upload(handle: String)
    /// A source kind added after this build. Named rather than dropped, so a
    /// row carrying one still decodes and still reads as "not this print".
    case other(kind: String)

    private enum CodingKeys: String, CodingKey { case kind, filename, handle }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(String.self, forKey: .kind) {
        case "library": self = .library(filename: try container.decode(String.self, forKey: .filename))
        case "upload": self = .upload(handle: try container.decode(String.self, forKey: .handle))
        case let kind: self = .other(kind: kind)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case let .library(filename):
            try container.encode("library", forKey: .kind)
            try container.encode(filename, forKey: .filename)
        case let .upload(handle):
            try container.encode("upload", forKey: .kind)
            try container.encode(handle, forKey: .handle)
        case let .other(kind):
            try container.encode(kind, forKey: .kind)
        }
    }
}

/// `POST /api/gallery/upscale`'s answer (`video_upscale.rs:678-683`). The
/// still is upscaled and published SYNCHRONOUSLY -- there is no job to follow.
public struct GalleryImageUpscale: Codable, Hashable, Sendable {
    public let filename: String
    public let model: String
    public let scaleFactor: Int
}

/// What a clip job may be told to do next.
public enum FramewiseTransition: String, Sendable {
    case pause, resume, cancel
}
