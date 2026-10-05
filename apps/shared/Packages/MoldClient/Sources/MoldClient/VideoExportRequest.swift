import Foundation

public enum GifPlayback: String, Codable, CaseIterable, Sendable { case loop, bounce }
public enum GifRepeat: String, Codable, CaseIterable, Sendable { case forever, once }

/// Extra boundary dwell, distinct from each frame's FPS-derived duration.
public struct GifPauseControl: Codable, Hashable, Sendable {
    public let min: Int
    public let max: Int
    public let step: Int
    public let defaultValue: Int
    public init(min: Int, max: Int, step: Int, defaultValue: Int) {
        self.min = min; self.max = max; self.step = step; self.defaultValue = defaultValue
    }
    private enum CodingKeys: String, CodingKey { case min, max, step; case defaultValue = "default" }
    public var valid: Bool {
        min >= 0 && max >= min && max <= 5000 && step > 0 && step % 10 == 0 && min % 10 == 0
            && accepts(defaultValue)
    }
    public func accepts(_ value: Int) -> Bool {
        step > 0 && value >= min && value <= max && (value - min) % step == 0
    }
}

public enum MediaExportKind: Sendable {
    case video, mesh
    public init?(filename: String, trashed: Bool) {
        guard !trashed else { return nil }
        switch URL(fileURLWithPath: filename).pathExtension.lowercased() {
        case "mp4": self = .video
        case "glb": self = .mesh
        default: return nil
        }
    }
}

public struct VideoExportRequest: Encodable, Equatable, Sendable {
    public var format: String
    public var playback: GifPlayback
    public var repeatMode: GifRepeat
    public var maxDimension: Int?
    public var fps: Int?
    public var pauseMs: Int?
    public init(format: String = "gif", playback: GifPlayback = .loop, repeatMode: GifRepeat = .forever,
                maxDimension: Int? = 720, fps: Int? = 12, pauseMs: Int? = nil) {
        self.format = format; self.playback = playback; self.repeatMode = repeatMode
        self.maxDimension = maxDimension; self.fps = fps; self.pauseMs = pauseMs
    }
    public var effectivePauseMs: Int? {
        format == "gif" && (playback == .bounce || repeatMode == .forever) ? pauseMs : nil
    }
    private enum CodingKeys: String, CodingKey { case format, playback, repeatMode = "repeat", maxDimension, fps, pauseMs }
    public func encode(to encoder: any Encoder) throws {
        var row = encoder.container(keyedBy: CodingKeys.self)
        try row.encode(format, forKey: .format)
        try row.encode(format == "gif" ? playback : .loop, forKey: .playback)
        try row.encode(format == "gif" ? repeatMode : .forever, forKey: .repeatMode)
        try row.encodeIfPresent(maxDimension, forKey: .maxDimension)
        try row.encodeIfPresent(fps, forKey: .fps)
        try row.encodeIfPresent(effectivePauseMs, forKey: .pauseMs)
    }
    public static func filename(_ source: String, format: String) -> String {
        MeshExport.filename(source, format: format == "apng" ? "png" : format)
    }
}
