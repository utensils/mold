import Foundation

/// Reading a recipe's capabilities, one documented absence rule at a time.
///
/// The app asks these questions and never the raw optionals, because the
/// answer to "is this missing" is different for every field -- the same
/// convention as `Capabilities+Reading`.
public extension RecipeCapabilities {

    /// Whether this recipe reads a still to condition on.
    ///
    /// ABSENT MEANS YES. The manifest omits the field for every image family
    /// (`crates/mold-core/src/manifest.rs:265-270`: "`None` omits the wire
    /// field (image families)"), and `validation::source_image_contract_violation`
    /// (`validation.rs:2061-2091`) refuses nothing for a `None` capability.
    /// Reading absence as "no source path" hid the well on every still model
    /// this fleet has installed, and took inpainting with it.
    var readsSourceImage: Bool { sourceImage?.isSupported ?? true }

    /// True only where the recipe REQUIRES one.
    var requiresSourceImage: Bool { sourceImage == .required }

    /// The adapter stack this recipe takes, or nil for none.
    ///
    /// Absent block falls back to `supportsLora`, and a `true` there with no
    /// block means an older host that never advertised a limit -- so the cap
    /// is `Lora.defaultMaxStack`. Absent block AND absent/false bool is an
    /// older host with no LoRA support at all: no stack.
    var loraStack: AdapterControl? {
        if let lora { return lora.mode.isVisible ? lora : nil }
        guard supportsLora == true else { return nil }
        return AdapterControl(mode: .adjustable, maxCount: Lora.defaultMaxStack, reason: nil)
    }

    /// ControlNet. Same fallback shape as `loraStack`; absent-and-absent is
    /// hidden, because `control_model` on a host that advertises neither
    /// would be refused by family validation anyway and there is nothing
    /// honest to draw.
    var controlNet: AdapterControl? {
        if let controlnet { return controlnet.mode.isVisible ? controlnet : nil }
        guard supportsControlnet == true else { return nil }
        return AdapterControl(mode: .adjustable, maxCount: 1, reason: nil)
    }

    /// An inpainting mask. Absent means an older host; treated as available,
    /// because the mask row is additionally gated on `readsSourceImage` and
    /// `validation.rs:3101-3107` is the authority either way.
    var acceptsMask: Bool { mask?.isAvailable ?? true }

    var acceptsKeyframes: Bool { keyframes?.isAvailable ?? false }
    var acceptsSourceAudio: Bool { audio?.isAvailable ?? false }
    var acceptsSourceVideo: Bool { sourceVideo?.isAvailable ?? false }

    /// What a blocked control SAYS. The server's own sentence, verbatim.
    var controlNetReason: String? { controlnet?.reason }
}
