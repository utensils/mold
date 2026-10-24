import Foundation

/// The lengths a native Length slider may offer, all from the selected server.
///
/// An ORDINARY render's slider snaps to the model's frame grid (`17n+5` for H3
/// at 24 fps). A Duration video is trimmed to its exact length server-side, so
/// its sliders move in tenths of a second: the single prompt from 1.0 s up to
/// one generation, a scene from 0.1 s up to the longest scene that is always
/// one clip wherever it lands, `(single_clip_max_frames - overlap_frames) / fps`
/// floored to 0.1 s -- the same figure the server's "will repeat" warning names.
public struct VideoLengthGrid: Hashable, Sendable {
    public let fps: Int
    public let minFrames: Int
    public let maxFrames: Int
    public let step: Int
    public let offset: Int
    /// Frames a later clip spends on incoming context.
    public let overlapFrames: Int

    public init(fps: Int, minFrames: Int, maxFrames: Int, step: Int, offset: Int, overlapFrames: Int) {
        self.fps = max(fps, 1)
        self.step = max(step, 1)
        self.offset = offset
        self.overlapFrames = max(overlapFrames, 0)
        let lowest = Self.onGrid(up: max(minFrames, 1), step: self.step, offset: offset)
        self.minFrames = lowest
        self.maxFrames = max(Self.onGrid(down: maxFrames, step: self.step, offset: offset), lowest)
    }

    /// A Duration video's bounds: its PLAN's clip window and overlap, which is
    /// what `/api/video-jobs` splits on (the recommended window for every
    /// family but H3). Chain limits stand in only until a plan exists.
    public static func duration(temporal: TemporalProfile, fps requested: Int?, limits: ChainLimits?,
                                plan: VideoJobPlan?) -> VideoLengthGrid {
        let base = resolve(temporal: temporal, fps: requested, limits: limits)
        guard let plan else { return base }
        // Seconds, not frames, drive a Duration slider, so the plan's window is
        // taken exactly rather than re-snapped onto a chain grid.
        return .init(fps: plan.fps, minFrames: min(base.minFrames, plan.singleClipMaxFrames),
                     maxFrames: plan.singleClipMaxFrames, step: 1, offset: 0, overlapFrames: plan.overlapFrames)
    }

    /// An ordinary render's grid and bounds: chain limits outrank a plan (a
    /// fallback for a host too old to publish limits), which outranks the
    /// recipe. `ceiling` replaces the
    /// single-generation maximum for an ordinary render the router may chain.
    public static func resolve(temporal: TemporalProfile, fps requested: Int?, limits: ChainLimits?,
                               plan: VideoJobPlan? = nil, ceiling: Int? = nil) -> VideoLengthGrid {
        let rate = limits?.fps ?? plan?.fps ?? temporal.fps.resolvedRate(requested)
        let authoring = limits?.authoring
        let step = authoring?.frameStep ?? temporal.frames.step
        let offset = authoring?.frameOffset ?? temporal.frameOffset ?? 0
        let minimum = authoring?.minFrames ?? temporal.frames.min
        let single = limits?.framesPerClipCap ?? plan?.singleClipMaxFrames ?? temporal.durationCappedMaxFrames(fps: rate)
        let maximum = ceiling ?? single
        let tail = authoring?.motionTailFrames ?? 0
        let overlap = authoring == nil ? plan?.overlapFrames ?? min(8, max(single - 1, 0))
            : (tail == 0 ? min(8, max(single - 1, 0)) : tail)
        return .init(fps: rate, minFrames: minimum, maxFrames: maximum, step: step, offset: offset, overlapFrames: overlap)
    }

    /// Every accepted frame count, shortest first.
    public var frames: [Int] { Array(stride(from: minFrames, through: maxFrames, by: step)) }

    /// The nearest accepted frame count.
    public func snap(_ requested: Int) -> Int {
        let clamped = min(max(requested, minFrames), maxFrames)
        let steps = ((clamped - minFrames) + step / 2) / step
        return min(minFrames + steps * step, maxFrames)
    }

    public func nearestFrames(milliseconds: UInt64) -> Int {
        snap(Int(clamping: (milliseconds * UInt64(fps) + 500) / 1_000))
    }

    /// The exact decimal duration a Duration request carries for `frames`.
    public func seconds(frames: Int) -> String {
        (try? VideoDurationDraft.initialSeconds(frames: frames, fps: fps)) ?? "1"
    }

    /// "14.4 s".
    public func label(frames: Int) -> String {
        VideoJobRow.seconds(UInt64(max(frames, 0)) * 1_000 / UInt64(fps))
    }

    /// A Duration readout, always one decimal: "14.0 s", "14.4 s".
    public static func label(milliseconds: UInt64) -> String {
        let tenths = milliseconds / 100 + (milliseconds % 100 >= 50 ? 1 : 0)
        return "\(tenths / 10).\(tenths % 10) s"
    }

    /// The single-prompt Duration slider: 1.0 s up to one generation.
    public var singleMinTenths: Int { min(10, singleMaxTenths) }
    public var singleMaxTenths: Int { max(maxFrames * 10 / fps, 1) }

    /// A scene is trimmed to its exact length, so it may be as short as 0.1 s.
    public var sceneMinTenths: Int { 1 }

    /// The longest scene that always stays one clip.
    public var sceneMaxTenths: Int { max((maxFrames - overlapFrames) * 10 / fps, 1) }

    public func clampSceneTenths(_ tenths: Int) -> Int { min(max(tenths, sceneMinTenths), sceneMaxTenths) }

    /// "Total 38.6 s · 3 scenes".
    public static func total(milliseconds: [UInt64]) -> String {
        let sum = milliseconds.reduce(UInt64(0)) { $0.addingReportingOverflow($1).partialValue }
        return "Total \(VideoJobRow.seconds(sum)) · \(Plural.count(milliseconds.count, of: "scene"))"
    }

    /// A scene's saved seconds as tenths, rounded to the nearest.
    public static func tenths(seconds: String) -> Int? {
        guard let milliseconds = try? VideoDurationDraft.milliseconds(seconds) else { return nil }
        return Int(clamping: (milliseconds + 50) / 100)
    }

    public static func seconds(tenths: Int) -> String {
        VideoDurationDraft.seconds(milliseconds: UInt64(max(tenths, 1)) * 100)
    }

    private static func onGrid(up value: Int, step: Int, offset: Int) -> Int {
        guard step > 1 else { return value }
        let steps = max(value - offset + step - 1, 0) / step
        return steps * step + offset
    }

    private static func onGrid(down value: Int, step: Int, offset: Int) -> Int {
        guard step > 1 else { return value }
        return max(value - offset, 0) / step * step + offset
    }
}
