import Foundation

/// LTX-2's per-request guidance overrides.
///
/// Port of `crates/mold-core/src/types.rs:2601-2625`. Every field is optional
/// and replaces exactly ONE of the constants the chosen pipeline pins for its
/// stage, so "a request without overrides is byte-identical to one made before
/// this contract existed" (`types.rs:2593-2596`) -- which is why an EMPTY set
/// is never sent rather than sent as `{}`.
///
/// Overrides tune a guider the pipeline already runs; they never switch one
/// on. A pipeline that disables a guider outright -- LTX-2's `a2-vid` audio
/// guider -- stays disabled whatever is set here (`types.rs:2598-2600`), so
/// the app offers a field only where the pipeline runs the guider it belongs
/// to and a value for a disabled one would be a lie about what will happen.
public struct Ltx2GuidanceOverrides: Codable, Hashable, Sendable {
    /// Spatiotemporal guidance scale. `0.0` disables the perturbed pass.
    public var stgScale: Double?
    /// Transformer block indices perturbed for STG.
    public var stgBlocks: [Int]?
    /// CFG-rescale factor (`0.0` no rescale, `1.0` full std matching).
    public var rescaleScale: Double?
    /// Cross-modality (audio <-> video) guidance scale. `1.0` disables the
    /// isolated-modality pass.
    public var modalityScale: Double?
    /// Guidance skip stride. `0` guides every step.
    public var skipStep: Int?

    public init(stgScale: Double? = nil, stgBlocks: [Int]? = nil,
                rescaleScale: Double? = nil, modalityScale: Double? = nil,
                skipStep: Int? = nil) {
        self.stgScale = stgScale
        self.stgBlocks = stgBlocks
        self.rescaleScale = rescaleScale
        self.modalityScale = modalityScale
        self.skipStep = skipStep
    }

    /// `Ltx2GuidanceOverrides::MAX_SCALE` (`types.rs:2629`) -- "well past any
    /// useful setting; the point is to reject nonsense (and NaN) rather than
    /// to tune".
    public static let maxScale: Double = 10
    /// `Ltx2GuidanceOverrides::MAX_SKIP_STEP` (`types.rs:2631`).
    public static let maxSkipStep = 8

    /// True when no field is set -- `Ltx2GuidanceOverrides::is_empty`
    /// (`types.rs:2635-2637`).
    public var isEmpty: Bool { self == Ltx2GuidanceOverrides() }

    /// What the request carries: the block when something is set, and NOTHING
    /// at all otherwise. Port of `guidanceOverridesToWire` /
    /// `guidanceOverridesAreEmpty` and of `Ltx2GuidanceOverrides::non_empty`
    /// (`types.rs:2639-2641`).
    public var wire: Ltx2GuidanceOverrides? { isEmpty ? nil : self }

    /// Both scales and the skip stride, clamped to what admission accepts.
    /// The blocks are NOT clamped: their ceiling is the selected checkpoint's
    /// transformer depth, which no client knows, so an out-of-range index is
    /// the server's refusal to make rather than this app's guess to silence.
    public func clamped() -> Ltx2GuidanceOverrides {
        Ltx2GuidanceOverrides(
            stgScale: stgScale.map { Swift.min(Swift.max($0, 0), Self.maxScale) },
            stgBlocks: stgBlocks,
            rescaleScale: rescaleScale.map { Swift.min(Swift.max($0, 0), 1) },
            modalityScale: modalityScale.map { Swift.min(Swift.max($0, 0), Self.maxScale) },
            skipStep: skipStep.map { Swift.min(Swift.max($0, 0), Self.maxSkipStep) })
    }
}
