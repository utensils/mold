import Foundation

// Which families a person may pick, and for what. The sets mirror
// `crates/mold-core/src/manifest.rs`; `ModelFamilyContractTests` reads that
// file and fails if they drift.
public extension Model {
    /// Families that are not standalone generators at all. A prompt-expansion
    /// LLM or a ControlNet in a model picker is a bug, not a listing.
    static let utilityFamilies: Set<String> = ["qwen3-expand", "companion"]
    static let upscalerFamilies: Set<String> = ["upscaler"]
    static let auxiliaryFamilies: Set<String> = [
        "controlnet", "ltx2-control", "ltx2-camera-control",
        "pulid", "ip-adapter", "hunyuan3d-paint",
    ]

    /// `"controlnet"` is already inside `auxiliaryFamilies`, mirrored from
    /// `manifest.rs`'s `AUXILIARY_FAMILIES` -- this names the SUBSET of it the
    /// Refine group's adapter picker may offer, not a second capability rule.
    static let controlNetFamilies: Set<String> = ["controlnet"]

    /// Families whose output is a MESH, not a picture.
    ///
    /// Deliberately NOT mirrored from a Rust constant, because there is none:
    /// `hunyuan3d` is a generating family on the server side and putting it in
    /// `auxiliaryFamilies` would both lie and break the contract test that
    /// reads `manifest.rs`. `hunyuan3d-paint` is the auxiliary half and is
    /// already there.
    static let meshFamilies: Set<String> = ["hunyuan3d"]

    var isUtility: Bool { Self.utilityFamilies.contains(family) }
    var isUpscaler: Bool { Self.upscalerFamilies.contains(family) }
    var isAuxiliary: Bool { Self.auxiliaryFamilies.contains(family) }
    var isControlNet: Bool { Self.controlNetFamilies.contains(family) }
    var isMeshMaker: Bool { Self.meshFamilies.contains(family) }

    /// True when this is a standalone model rather than an adapter, an
    /// upscaler or a prompt-expansion LLM.
    var isGenerator: Bool { !isUtility && !isUpscaler && !isAuxiliary }

    /// True when a person picking "what should make this picture" should see
    /// it -- a generator whose output is an image or a clip.
    ///
    /// A mesh family is a generator and is excluded here for one reason: this
    /// app cannot yet DRAW a GLB. Offering it produced a structurally valid
    /// request carrying no `mesh` block, so the server fell back to its
    /// `MESH_DEFAULT_*`, and the finished print rendered as a blank canvas --
    /// several GPU minutes for nothing. When the mesh viewer lands, this is
    /// the one line that reinstates it.
    var isPictureMaker: Bool { isGenerator && !isMeshMaker }
}
