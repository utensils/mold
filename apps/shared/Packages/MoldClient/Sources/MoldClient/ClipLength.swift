import Foundation

/// How long a clip this app can actually ask for.
///
/// `temporal.frames.max` is not that number, twice over (findings 01#3, 02#3):
///
/// - LTX-2 advertises the grid maximum AT 120 FPS on purpose -- "advertise the
///   largest requestable value; admission applies the lower duration-derived
///   cap for the selected FPS" (`generation_profile.rs:2558-2566`). At the
///   default 24 fps the real ceiling is `20 x 24 + 1 = 481`, not 601, and
///   every value above it was a hard 422 at submit.
/// - Wan's `frames.max` is the family's flat 257-frame RESOURCE guard, not the
///   checkpoint's trained clip. A text-to-video tier hands nothing across a
///   clip boundary, so the only honest single-pass ceiling is its clip size.
public struct ClipLengthBounds: Hashable, Sendable {
    public let min: Int
    public let max: Int
    /// Why the ceiling is lower than the recipe's own maximum, when it is.
    /// `nil` where the advertised maximum stands.
    public let note: String?

    /// Spelled out so the app can widen a ceiling the router would chain
    /// past: `lengthBounds` answers for ONE denoise, and a chainable model's
    /// slider may reach further (`ClipRouting`).
    public init(min: Int, max: Int, note: String?) {
        self.min = min
        self.max = max
        self.note = note
    }
}

public extension TemporalProfile {
    /// The frame count admission would really accept at this rate.
    ///
    /// Mirrors `generation_profile.rs:1241-1251` exactly -- the narrowing runs
    /// BEFORE `validate_integer`, on the grid, from `max_duration_seconds`.
    /// Studio computes `max_runtime_seconds * fps + 4` and snaps down
    /// (`videoDuration.ts:138-167`), which agrees wherever `seconds x fps`
    /// lands on the grid; this follows the door that actually refuses.
    func durationCappedMaxFrames(fps: Int) -> Int {
        guard let seconds = maxDurationSeconds, seconds > 0 else { return frames.max }
        let offset = frameOffset ?? 0
        let step = Swift.max(frames.step, 1)
        let raw = Int((seconds * Double(Swift.max(fps, 1))).rounded(.down)) + offset
        let gridCap = (Swift.max(raw - offset, 0) / step) * step + offset
        return Swift.min(frames.max, gridCap)
    }

    /// The nearest frame count on the grid AT OR BELOW `requested`. A ceiling
    /// rounded to the nearest would round back up past itself.
    func snapDown(_ requested: Int) -> Int {
        let step = Swift.max(frames.step, 1)
        let offset = frameOffset ?? 0
        let clamped = Swift.min(Swift.max(requested, frames.min), frames.max)
        guard step > 1 else { return clamped }
        let steps = Swift.max(clamped - offset, 0) / step
        return Swift.min(Swift.max(steps * step + offset, frames.min), frames.max)
    }

    /// What the Length control may offer.
    ///
    /// `family`, `model` and `sourceImage` answer the clip-size question only;
    /// everything else comes off the recipe. This app renders one denoise per
    /// press -- automatic chaining is not built yet -- so where studio would
    /// split a longer render into stages, this stops at the clip and says so.
    func lengthBounds(
        fps: Int, family: String?, model: String?, sourceImage: SourceImageCapability?
    ) -> ClipLengthBounds {
        var ceiling = durationCappedMaxFrames(fps: fps)
        var note: String?
        if let clip = ClipLengthBounds.singleClipCeiling(
            family: family, model: model, sourceImage: sourceImage,
            tierDefault: frames.default), clip < ceiling {
            ceiling = clip
            note = ClipLengthBounds.singleClipNote(model: model ?? "This model", frames: clip)
        }
        return ClipLengthBounds(
            min: frames.min, max: Swift.max(snapDown(ceiling), frames.min), note: note)
    }
}

public extension ClipLengthBounds {
    /// Wan's per-checkpoint routing clip size -- what ONE generation renders.
    /// Port of `chainRouting.ts:275-301` (`wanRoutingClipFrames`), whose floor
    /// mirrors `mold_core::chain::wan_default_clip_frames`.
    static func wanRoutingClipFrames(model: String, tierDefault: Int?) -> Int {
        let floor = model.lowercased().contains("a14b")
            ? wanDefaultClipFrames : wanSingleExpertClipFrames
        guard let tierDefault, tierDefault > floor else { return floor }
        return tierDefault
    }

    static let wanDefaultClipFrames = 53
    static let wanSingleExpertClipFrames = 121

    /// The single-request ceiling of a model that cannot be auto-chained.
    ///
    /// Port of `chainRouting.ts:211-222`
    /// (`textOnlyAutoChainSingleClipCeiling`), pinned against
    /// `tests/fixtures/wan/surface-parity-v1.json`. A wan tier whose advertised
    /// `source_image` contract is `unsupported` has no conditioning channel, so
    /// nothing crosses a clip boundary and its clip size IS its ceiling. `nil`
    /// for every model this does not apply to, which is the only thing callers
    /// should key on.
    static func singleClipCeiling(
        family: String?, model: String?, sourceImage: SourceImageCapability?, tierDefault: Int?
    ) -> Int? {
        let normalized = (family ?? "").trimmingCharacters(in: .whitespaces).lowercased()
        guard normalized == "wan", sourceImage == .unsupported else { return nil }
        return wanRoutingClipFrames(model: model ?? "", tierDefault: tierDefault)
    }

    /// Why the slider stops there. The server's own refusal
    /// (`chainRouting.ts:224-258`) explains a CHAIN this app does not build,
    /// so this says the part that is true here and names the same number.
    static func singleClipNote(model: String, frames: Int) -> String {
        "'\(model)' is text-to-video and cannot carry motion across a clip "
            + "boundary, so one continuous clip is at most \(frames) frames. "
            + "An image-to-video tier can be continued."
    }
}
