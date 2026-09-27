import Foundation

/// Which sampler controls a recipe actually offers.
///
/// Resolved ONCE, so the pane that draws the controls and the reconciliation
/// that parks their values can never disagree about what this recipe has.
/// Everything here comes off the advertised profile; the one exception is
/// CFG++, which has no capability at all and is documented below.
public struct AdvancedControlsOffered: Hashable, Sendable {
    /// The advertised solvers, in the recipe's own spelling. EMPTY means no
    /// picker: the server omits `schedulers` when the list is empty
    /// (`skip_serializing_if`), so a DMD tier that pins its sampler arrives
    /// with the key missing and absence there is a definitive "none", not an
    /// older host (`generationCapabilities.ts:404-410`).
    public let schedulers: [String]
    public let cfgPlus: Bool
    public let sampleShift: Bool
    public let distillStrength: Bool
    /// LTX-2's `guidance_overrides`.
    public let guidance: Bool
    /// `modality_scale` alone. An audio-only pipeline has no video modality to
    /// guide against, and admission refuses any value but 1.0 there
    /// (`validation.rs:3302-3312`), so the control is absent rather than
    /// offered and refused.
    public let modalityScale: Bool

    public var offersAnything: Bool {
        !schedulers.isEmpty || cfgPlus || sampleShift || distillStrength || guidance
    }

    /// The ONE sanctioned client family set in this app.
    ///
    /// CFG++ has no capability block anywhere -- `generationCapabilities.ts:282`
    /// is a client-side set and nothing else, which is why a contract test
    /// reads that TypeScript and fails when the two drift. Nowhere else in
    /// this app may a family name decide a control.
    public static let cfgPlusFamilies: Set<String> = ["sd3", "sd3.5"]

    /// `Ltx2PipelineMode::is_audio_only`'s one member -- `t2a` renders a WAV
    /// and nothing else (`studio/lib/ltx2Pipeline.ts:13`).
    public static let audioOnlyPipeline = "t2a"

    /// What `recipe` offers, given the model's whole profile and its family.
    ///
    /// The legacy-host family heuristic studio falls back to when it holds no
    /// recipe at all is DELIBERATELY NOT PORTED: this app draws every control
    /// from the model's generation profile, and a host that advertises no
    /// profile is not a target it supports. A recipe in hand answers for
    /// itself; no recipe means no controls.
    public static func resolve(
        recipe: GenerationRecipe?, in profile: GenerationProfileSet?, family: String?
    ) -> AdvancedControlsOffered {
        guard let recipe else {
            return AdvancedControlsOffered(
                schedulers: [], cfgPlus: false, sampleShift: false,
                distillStrength: false, guidance: false, modalityScale: false)
        }
        let wan = recipe.capabilities.wanRecipe
        let wanVisible = wan?.mode.isVisible == true
        // LTX-2 is the one family whose recipes are CHOSEN by pipeline
        // (`generation_profile.rs:2222`, `RecipeSelector.pipeline` is an
        // `Option<Ltx2PipelineMode>`), which is exactly the family
        // `require_ltx2_family` gates `guidance_overrides` on
        // (`validation.rs:3295-3296`) -- so this asks the profile's own shape
        // rather than matching a family name.
        let guidance = profile?.recipes.contains { $0.requestSelector?.pipeline != nil } ?? false
        let normalized = (family ?? "").trimmingCharacters(in: .whitespaces).lowercased()
        return AdvancedControlsOffered(
            schedulers: recipe.capabilities.schedulers ?? [],
            cfgPlus: cfgPlusFamilies.contains(normalized),
            sampleShift: wanVisible,
            distillStrength: wanVisible && wan?.supportsDistillStrength == true,
            guidance: guidance,
            modalityScale: guidance && recipe.requestSelector?.pipeline != audioOnlyPipeline)
    }
}
