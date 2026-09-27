import Foundation

/// The sampler controls a recipe advertises: the solver, CFG++, wan's own
/// flow shift and distill strengths, and LTX-2's guidance overrides.
///
/// One struct rather than eight fields on `RenderDraft` because they share a
/// rule: EVERY one is absent from the request until somebody moves it, since
/// the engine keeps the recipe's own constant only while the field is absent
/// -- "a null serialized as `0` would silently change every render"
/// (`studio/lib/guidanceOverrides.ts:9-12`, `wanRecipe.ts:11-13`). So `nil`
/// here means "the recipe's", never "zero".
///
/// Each is PARKED rather than dropped when the recipe stops advertising it
/// (`AdvancedControls+Park.swift`), the same rescue every conditioning input
/// gets.
public struct AdvancedControls: Hashable, Sendable {
    /// The sampler, echoed straight from `capabilities.schedulers`. A `String`
    /// rather than a Swift enum for the same reason `outputFormat` is one:
    /// this app's whole job is to hand back a spelling the recipe itself
    /// advertised, never to invent one that could drift from the server's
    /// strict `Scheduler` enum (`types.rs:138-154`). `nil` is "the recipe's
    /// own", which reaches the wire as absence.
    public var scheduler: String?
    /// CFG++ (`cfg_plus`). Sent only as `true` -- absence IS false to the
    /// server, and an explicit `false` would be a value nobody chose.
    public var cfgPlus = false

    // Wan's sampler recipe (`studio/lib/wanRecipe.ts`).
    /// Flow shift. Any finite positive value; the useful range moves with
    /// resolution and frame count, so there is no upper bound (`wanRecipe.ts:61-63`).
    public var sampleShift: Double?
    /// Lightning distill strengths, per expert. Band `(0, 4]`.
    public var distillStrengthHigh: Double?
    public var distillStrengthLow: Double?

    // LTX-2's guidance overrides (`studio/lib/guidanceOverrides.ts`).
    public var stgScale: Double?
    /// Free text, exactly as studio's control is: a comma-separated block
    /// list, parsed at request time. Holding the TEXT rather than the parsed
    /// list is what lets a half-typed "3, " stay on screen instead of being
    /// silently rewritten under the cursor.
    public var stgBlocks: String = ""
    public var rescaleScale: Double?
    public var modalityScale: Double?
    public var skipStep: Int?

    /// What the current recipe does not advertise, held so it comes back.
    public var parked = ParkedAdvancedControls()

    public init() {}
}

/// The parked twin of `AdvancedControls`. `cfgPlus` is `Bool?` rather than
/// `Bool` so "nothing parked" and "parked off" stay different answers, which
/// is what the shared `DraftMedia.reconcile` needs to work on it.
public struct ParkedAdvancedControls: Hashable, Sendable {
    public var scheduler: String?
    public var cfgPlus: Bool?
    public var sampleShift: Double?
    public var distillStrengthHigh: Double?
    public var distillStrengthLow: Double?
    public var stgScale: Double?
    public var stgBlocks: String?
    public var rescaleScale: Double?
    public var modalityScale: Double?
    public var skipStep: Int?

    public init() {}
}
