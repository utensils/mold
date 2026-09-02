//! Static MiniMax H3 contracts.
//!
//! H3 acquisition and execution are separate authorities. The compact
//! upstream checkpoints are downloadable, while runtime admission remains
//! limited to independently qualified backend routes. This module records the
//! immutable model/layout/request facts shared by both boundaries.

use crate::manifest::{ManifestDefaults, ModelComponent, ModelFile, ModelManifest};
use crate::{
    GenerateRequest, GenerationReference, GenerationReferenceAuthority, OutputFormat,
    MINIMAX_H3_LICENSE_SHA256, MINIMAX_H3_LICENSE_URL,
};

pub const FAMILY: &str = "minimax-h3";
pub const FAMILY_ALIASES: &[&str] = &["minimax-h3", "minimax_h3", "minimaxh3"];

pub const OFFICIAL_REPO: &str = "MiniMaxAI/MiniMax-H3";
pub const OFFICIAL_REVISION: &str = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86";
pub const COMFY_REPO: &str = "Comfy-Org/MiniMax-H3";
pub const COMFY_REVISION: &str = "eb8a16107c595128b3a578f82d2ce2f75920c355";
pub const COMFY_IMPLEMENTATION_REPO: &str = "Comfy-Org/ComfyUI";
pub const COMFY_IMPLEMENTATION_REVISION: &str = "a464ac33588ae182f81a090d910cfbf21e255b73";
pub const OFFICIAL_IMPLEMENTATION_REPO: &str = "MiniMax-AI/MiniMax-H3";
pub const OFFICIAL_IMPLEMENTATION_REVISION: &str = "8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea";
pub const DIFFUSERS_REFERENCE_REPO: &str = "huggingface/diffusers";
pub const DIFFUSERS_REFERENCE_REVISION: &str = "9c6a68c32b3b2a64db91800b624d33cec6e25ab8";
/// Third-party repository publishing the pruned NVFP4 compact transformers.
///
/// This is the first source mold pins that is neither MiniMaxAI nor
/// Comfy-Org, and only the transformer comes from it: the conditioner, both
/// VAEs, and every config still resolve to [`COMFY_REPO`] and
/// [`OFFICIAL_REPO`]. Comfy-Org publishes no NVFP4 diffusion model at all,
/// so there is no first-party artifact to prefer.
///
/// Every object in this repository carries an appended
/// `\nL2P_bypass_<filename>_<unix_ts>\n` marker past the safetensors
/// payload. It is content-dedup defeat, not tampering — the same repository's
/// INT8 copy has its payload end at exactly the byte count
/// `H3ComfyPublishedArtifact::file_bytes()` pins, and its header hashes to
/// exactly `H3_COMFY_PUBLISHED_INT8_HEADER_SHA256`. The marker is *inside*
/// the pinned digest below, so it is part of the reviewed content identity:
/// a future re-upload without it is a different artifact and must be
/// re-pinned rather than silently accepted.
pub const NVFP4_REPO: &str = "Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot";
pub const NVFP4_REVISION: &str = "908eccad7e68751190d04c171956f163bfeed741";
/// Third-party repository publishing the ModelTC/lightx2v Turbo LoRA adapters.
///
/// The second non-MiniMaxAI/non-Comfy-Org source mold pins, and only ADAPTERS
/// come from it: the base stack every tag executes on still resolves to
/// [`COMFY_REPO`] and [`OFFICIAL_REPO`]. Its files sit at the repository ROOT
/// with no `loras/` directory, which is why a tier carries its own
/// repository-relative path and its own revision rather than inheriting a
/// repository-wide one.
///
/// The repository's v1.0 adapters are byte-identical (same SHA-256) to the
/// copies Comfy-Org re-hosts at [`COMFY_TURBO_LORA_REVISION`], which is the
/// provenance corroboration; the v1.1 4-step 768p and the 8-step 768p
/// adapters exist only here. It declares `apache-2.0` for the adapters
/// themselves, and the MiniMax H3 Community License still governs the base
/// checkpoint a tag executes on.
pub const LIGHTX2V_REPO: &str = "lightx2v/Minimax-h3-Turbo";
pub const LIGHTX2V_REVISION: &str = "05ef678438e84933c406131b59abbf86919b3aac";
/// Third-party repository publishing the SVD-resized Turbo LoRA adapters.
///
/// The third non-MiniMaxAI/non-Comfy-Org source mold pins, and like
/// [`LIGHTX2V_REPO`] only ADAPTERS come from it: the base stack every tag
/// executes on still resolves to [`COMFY_REPO`] and [`OFFICIAL_REPO`]. Its
/// files sit at the repository ROOT with no `loras/` directory, so a tier
/// carries its own repository-relative path and its own revision.
///
/// Each file here is a DERIVATIVE of an adapter mold already ships: a compact
/// SVD per module, `sqrt(S)`-balanced back into `lora_A`/`lora_B`, the `alpha`
/// tensors dropped and the source `alpha / rank` multiplied into `lora_B`.
/// That makes them a lossy low-rank approximation (average rank 21 against the
/// source's 128) of an artifact whose full-rank original is pinned in this
/// same table, which is the provenance corroboration for pinning them at all.
/// The publisher declares `apache-2.0` on the derivatives of the
/// Comfy-Org-re-hosted lightx2v adapters; the MiniMax H3 Community License
/// still governs the base checkpoint a tag executes on.
///
/// Only the three `resized_avg_rank_21` files are reviewed. The repository's
/// other rank-20/28/64 resizes at the same revision are deliberately excluded
/// (their `baked_scale` metadata is free text rather than a number and they
/// carry no `resized_from`, so nothing pins the scale that was folded in or
/// the adapter it approximates), and so are the eight `*_pruned_comfyui`
/// full checkpoints published beside them (whole transformers, not adapters —
/// mold's compact transformer identity is pinned from [`COMFY_REPO`]) and the
/// Kijai rank-21 files republished elsewhere (no declared license, and an
/// internal-checkpoint source that cannot be corroborated against anything
/// mold ships).
pub const DRBAPH_TURBO_LORA_REPO: &str = "drbaph/MiniMax-H3-Turbo-Lora-ComfyUI";
pub const DRBAPH_TURBO_LORA_REVISION: &str = "be8eb3ea3466cbb7def202ffec0d2fdc054256ac";
pub const LICENSE_SHA256: &str = MINIMAX_H3_LICENSE_SHA256;

pub const FL2VA_OFFICIAL: &str = "minimax-h3-fl2va:official-bf16";
pub const REF2VA_OFFICIAL: &str = "minimax-h3-ref2va:official-bf16";
pub const FL2VA_COMFY: &str = "minimax-h3-fl2va:comfy-pruned-int8";
pub const REF2VA_COMFY: &str = "minimax-h3-ref2va:comfy-pruned-int8";
pub const FL2VA_COMFY_TURBO_8STEP: &str = "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step";
pub const FL2VA_COMFY_TURBO_4STEP_768P: &str =
    "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p";
pub const REF2VA_COMFY_TURBO_4STEP: &str = "minimax-h3-ref2va:comfy-pruned-int8-turbo-4step";
/// lightx2v's v1.1 4-step 768p FL2VA distillation. The tag keeps the
/// `comfy-pruned-int8` prefix because that names the BASE layout it executes
/// on; the adapter's own origin is provenance carried by the manifest file
/// row. `-v1.1` appears because a v1.0 4-step 768p tier already ships.
pub const FL2VA_COMFY_TURBO_4STEP_768P_V11: &str =
    "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-v1.1";
/// lightx2v's 8-step 768p FL2VA distillation. No version suffix: there is no
/// other 8-step 768p tier to disambiguate it from.
pub const FL2VA_COMFY_TURBO_8STEP_768P: &str =
    "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-768p";
/// SVD-resized derivative of [`FL2VA_COMFY_TURBO_4STEP_768P`]. Same
/// distillation, same 5-point grid at the 768p shift; roughly 1.66 GB less to
/// download and about 1.6 GB less resident, at the cost of a lossy low-rank
/// approximation of the adapter it is derived from.
pub const FL2VA_COMFY_TURBO_4STEP_768P_R21: &str =
    "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-r21";
/// SVD-resized derivative of [`FL2VA_COMFY_TURBO_8STEP`]. The tag carries no
/// `768p` because its source is the 544p-trained 8-step tier.
pub const FL2VA_COMFY_TURBO_8STEP_R21: &str = "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-r21";
/// SVD-resized derivative of [`REF2VA_COMFY_TURBO_4STEP`], the first resized
/// tier of the Ref2VA partition.
pub const REF2VA_COMFY_TURBO_4STEP_R21: &str =
    "minimax-h3-ref2va:comfy-pruned-int8-turbo-4step-r21";
/// Pruned NVFP4 compact transformers. Deliberately absent from
/// [`REVIEWED_COMPACT_MODELS`]: they download, verify, inventory, and remove
/// like any other pinned model, but mold has no engine arm for the weight
/// layout, so execution is refused at the route.
pub const FL2VA_COMFY_NVFP4: &str = "minimax-h3-fl2va:comfy-pruned-nvfp4";
pub const REF2VA_COMFY_NVFP4: &str = "minimax-h3-ref2va:comfy-pruned-nvfp4";

/// Every reviewed compact identity admissible on the H3 runtime route: the
/// two base task partitions plus the reviewed Turbo LoRA tags of both. Aliases
/// are deliberately excluded — policy and validation match exact identities.
pub const REVIEWED_COMPACT_MODELS: &[&str] = &[
    FL2VA_COMFY,
    REF2VA_COMFY,
    FL2VA_COMFY_TURBO_8STEP,
    FL2VA_COMFY_TURBO_4STEP_768P,
    REF2VA_COMFY_TURBO_4STEP,
    FL2VA_COMFY_TURBO_4STEP_768P_V11,
    FL2VA_COMFY_TURBO_8STEP_768P,
    FL2VA_COMFY_TURBO_4STEP_768P_R21,
    FL2VA_COMFY_TURBO_8STEP_R21,
    REF2VA_COMFY_TURBO_4STEP_R21,
];

/// Exact-identity membership test for [`REVIEWED_COMPACT_MODELS`]. This is
/// deliberately not alias-resolving: `minimax-h3` stays outside the reviewed
/// set exactly as it does in `model_policy`.
pub fn is_reviewed_compact_model(value: &str) -> bool {
    let value = value.trim();
    REVIEWED_COMPACT_MODELS
        .iter()
        .any(|model| value.eq_ignore_ascii_case(model))
}

/// Pinned Comfy-Org repository revision that first published the reviewed
/// Turbo LoRA adapters under `loras/`. The compact base stack stays pinned at
/// [`COMFY_REVISION`]; the adapters do not exist at that older revision, so
/// their manifest files resolve through [`file_revision`] instead. Tiers whose
/// adapter comes from [`LIGHTX2V_REPO`] resolve through their own tier row
/// rather than through this constant.
pub const COMFY_TURBO_LORA_REVISION: &str = "dc559027db79c174125df4d827db55cd11178860";

/// Shape sentence for an adapter distilled at one uniform rank with an `alpha`
/// tensor per module — every adapter Comfy-Org and lightx2v publish.
const UNIFORM_TURBO_ADAPTER_SHAPE: &str = "Diffusers PEFT LoRA; rank 128; 208 modules";
/// Shape sentence for an SVD-resized derivative: the per-module rank varies
/// (21 on average against the source's 128), there is no `alpha` tensor, and
/// the source `alpha / rank` is already multiplied into `lora_B`. The module
/// set is the same 208 either way — a resize changes ranks, never targets.
const RESIZED_TURBO_ADAPTER_SHAPE: &str =
    "SVD-resized PEFT LoRA; dynamic per-module rank (avg 21); baked scale; 208 modules";

/// The manifest-facing contract of one reviewed Turbo LoRA tier.
///
/// This mirrors the runtime tier table owned by `mold-candle`
/// (`H3TurboLoraTier`) — stable id, published file identity, and the
/// terminal-inclusive reviewed step count — so acquisition manifests can pin
/// the adapter without mold-core depending on candle. A contract test in
/// `mold-inference` pins both tables together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurboManifestTier {
    /// The manifest identity carrying this tier (a tag on the compact task).
    pub model: &'static str,
    /// The runtime tier's durable stable id.
    pub tier_stable_id: &'static str,
    /// Human-facing tier label ("Turbo 8-step").
    pub display_label: &'static str,
    /// Repository publishing this tier's adapter: [`COMFY_REPO`],
    /// [`LIGHTX2V_REPO`], or [`DRBAPH_TURBO_LORA_REPO`]. The base stack is
    /// unaffected — a Turbo tag is the compact stack of its own task plus
    /// this one file.
    pub adapter_hf_repo: &'static str,
    /// Pinned revision of [`Self::adapter_hf_repo`] this adapter is published
    /// at. `file_revision` resolves the pair, so an adapter can never be
    /// fetched from an unpinned `main`.
    pub adapter_hf_revision: &'static str,
    /// Repository-relative adapter path at [`Self::adapter_hf_revision`].
    /// Comfy-Org publishes under `loras/`; lightx2v and drbaph publish at
    /// their repository roots.
    pub adapter_hf_filename: &'static str,
    pub adapter_size_bytes: u64,
    pub adapter_sha256: &'static str,
    /// What `artifact_contract` reports as this adapter's shape. It is a
    /// per-tier fact rather than one sentence for the component, because an
    /// SVD-resized derivative and the rank-128 adapter it approximates are
    /// both `DistilledLora` files and a user reading a contract before a pull
    /// must be able to tell them apart.
    pub adapter_shape_label: &'static str,
    /// Terminal-inclusive mold steps: published transformer evaluations + 1.
    pub steps: u32,
}

/// Reviewed Turbo tiers shipped as first-class manifest tags, for both task
/// partitions. Ref2VA's 4-step tier joined the list once Ref2VA execution
/// landed; its adapter is the one `H3TurboLoraTier::Ref2v4StepV10` already
/// pins, and selection stays by model identity.
///
/// Adapters come from three reviewed repositories: Comfy-Org's `loras/`
/// re-hosts; for the two adapters Comfy-Org never re-hosted,
/// [`LIGHTX2V_REPO`] at its repository root; and for the SVD-resized
/// derivatives, [`DRBAPH_TURBO_LORA_REPO`], also at its root. (768p training
/// is not the distinction — `FL2VA_COMFY_TURBO_4STEP_768P` is 768p-trained
/// too and comes from Comfy-Org.) Each row names its own source, revision and
/// adapter shape, so acquisition never has to infer provenance from the tag.
pub const REVIEWED_TURBO_MANIFEST_TIERS: &[TurboManifestTier] = &[
    TurboManifestTier {
        model: FL2VA_COMFY_TURBO_8STEP,
        tier_stable_id: "minimax-h3.turbo-lora.fl2v-8step-v1.0.comfyui-bf16.v1",
        display_label: "Turbo 8-step",
        adapter_hf_repo: COMFY_REPO,
        adapter_hf_revision: COMFY_TURBO_LORA_REVISION,
        adapter_hf_filename: "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        adapter_size_bytes: 1_956_193_000,
        adapter_sha256: "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e",
        adapter_shape_label: UNIFORM_TURBO_ADAPTER_SHAPE,
        steps: 9,
    },
    TurboManifestTier {
        model: FL2VA_COMFY_TURBO_4STEP_768P,
        tier_stable_id: "minimax-h3.turbo-lora.fl2v-4step-768p-v1.0.comfyui-bf16.v1",
        display_label: "Turbo 4-step 768p",
        adapter_hf_repo: COMFY_REPO,
        adapter_hf_revision: COMFY_TURBO_LORA_REVISION,
        adapter_hf_filename: "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        adapter_size_bytes: 1_956_192_992,
        adapter_sha256: "c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc",
        adapter_shape_label: UNIFORM_TURBO_ADAPTER_SHAPE,
        steps: 5,
    },
    TurboManifestTier {
        model: REF2VA_COMFY_TURBO_4STEP,
        tier_stable_id: "minimax-h3.turbo-lora.ref2v-4step-v0.1.comfyui-bf16.v1",
        display_label: "Turbo 4-step",
        adapter_hf_repo: COMFY_REPO,
        adapter_hf_revision: COMFY_TURBO_LORA_REVISION,
        adapter_hf_filename: "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        adapter_size_bytes: 1_956_193_000,
        adapter_sha256: "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c",
        adapter_shape_label: UNIFORM_TURBO_ADAPTER_SHAPE,
        steps: 5,
    },
    TurboManifestTier {
        model: FL2VA_COMFY_TURBO_4STEP_768P_V11,
        tier_stable_id: "minimax-h3.turbo-lora.fl2v-4step-768p-v1.1.comfyui-bf16.v1",
        display_label: "Turbo 4-step 768p v1.1",
        adapter_hf_repo: LIGHTX2V_REPO,
        adapter_hf_revision: LIGHTX2V_REVISION,
        adapter_hf_filename: "minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors",
        adapter_size_bytes: 1_956_192_992,
        adapter_sha256: "449d80f301ac571622c72e28b8fd72a4b3681b7a8df8a92f17c8f6ec43f56558",
        adapter_shape_label: UNIFORM_TURBO_ADAPTER_SHAPE,
        steps: 5,
    },
    TurboManifestTier {
        model: FL2VA_COMFY_TURBO_8STEP_768P,
        tier_stable_id: "minimax-h3.turbo-lora.fl2v-8step-768p-v1.0.comfyui-bf16.v1",
        display_label: "Turbo 8-step 768p",
        adapter_hf_repo: LIGHTX2V_REPO,
        adapter_hf_revision: LIGHTX2V_REVISION,
        adapter_hf_filename: "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
        adapter_size_bytes: 1_956_193_000,
        adapter_sha256: "08cfe946033af7d27719b964b6e0a0e50c32138daabbd6ce4137e23df6bf9980",
        adapter_shape_label: UNIFORM_TURBO_ADAPTER_SHAPE,
        steps: 9,
    },
    TurboManifestTier {
        model: FL2VA_COMFY_TURBO_4STEP_768P_R21,
        tier_stable_id:
            "minimax-h3.turbo-lora.fl2v-4step-768p-v1.0.comfyui-bf16.resized-avg-rank-21.v1",
        display_label: "Turbo 4-step 768p (rank 21)",
        adapter_hf_repo: DRBAPH_TURBO_LORA_REPO,
        adapter_hf_revision: DRBAPH_TURBO_LORA_REVISION,
        adapter_hf_filename:
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_resized_avg_rank_21_bf16.safetensors",
        adapter_size_bytes: 298_177_224,
        adapter_sha256: "1b85da614014024a0c9507f12558917dcc69b6adb564e716324594f401723115",
        adapter_shape_label: RESIZED_TURBO_ADAPTER_SHAPE,
        steps: 5,
    },
    TurboManifestTier {
        model: FL2VA_COMFY_TURBO_8STEP_R21,
        tier_stable_id: "minimax-h3.turbo-lora.fl2v-8step-v1.0.comfyui-bf16.resized-avg-rank-21.v1",
        display_label: "Turbo 8-step (rank 21)",
        adapter_hf_repo: DRBAPH_TURBO_LORA_REPO,
        adapter_hf_revision: DRBAPH_TURBO_LORA_REVISION,
        adapter_hf_filename:
            "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_resized_avg_rank_21_bf16.safetensors",
        adapter_size_bytes: 327_035_608,
        adapter_sha256: "a3208be61329c27a6754c53db9a21a3c86e2a285381700adf2d97e279c062840",
        adapter_shape_label: RESIZED_TURBO_ADAPTER_SHAPE,
        steps: 9,
    },
    TurboManifestTier {
        model: REF2VA_COMFY_TURBO_4STEP_R21,
        tier_stable_id:
            "minimax-h3.turbo-lora.ref2v-4step-v0.1.comfyui-bf16.resized-avg-rank-21.v1",
        display_label: "Turbo 4-step (rank 21)",
        adapter_hf_repo: DRBAPH_TURBO_LORA_REPO,
        adapter_hf_revision: DRBAPH_TURBO_LORA_REVISION,
        adapter_hf_filename:
            "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_resized_avg_rank_21_bf16.safetensors",
        adapter_size_bytes: 326_935_264,
        adapter_sha256: "2c6abb194cff3e26c2295c87892913adf0c92d8f784f305238246759f9b333d0",
        adapter_shape_label: RESIZED_TURBO_ADAPTER_SHAPE,
        steps: 5,
    },
];

/// The compact base identity one task executes as. Kept as an exhaustive
/// `Task` match — never a constant — so a reviewed Turbo tier of either task
/// partitions to its own base.
pub fn base_compact_model_for_task(task: Task) -> &'static str {
    match task {
        Task::Fl2va => FL2VA_COMFY,
        Task::Ref2va => REF2VA_COMFY,
    }
}

/// The base compact identity a reviewed model executes as: a Turbo tag's
/// underlying task partition, or the compact identity itself. The internal
/// engine partition is keyed by this value while the request keeps the full
/// tag for provenance. The partition is derived from the tag's own TASK, not
/// hard-wired to FL2VA, so a future reviewed Ref2VA tier cannot resolve to
/// the FL2VA base. Identities outside the compact layout — the hidden
/// official BF16 references included — have no compact engine partition and
/// return `None`, so admission's route refusal fires at the route instead of
/// deep inside artifact qualification.
pub fn base_compact_model(model: &str) -> Option<&'static str> {
    let canonical = resolve_model_name(model)?;
    if layout_for_model(canonical) != Some(Layout::ComfyPrunedInt8ConvrotNvfp4Awq) {
        return None;
    }
    if REVIEWED_TURBO_MANIFEST_TIERS
        .iter()
        .any(|tier| tier.model == canonical)
    {
        return Some(base_compact_model_for_task(task_for_model(canonical)?));
    }
    Some(canonical)
}

/// The model identity whose directory holds a manifest's model-specific
/// files. A reviewed Turbo tag is the base compact stack plus one shared
/// adapter, so its base-stack files install into — and verify against — the
/// base checkpoint's directory instead of duplicating ~41 GB per tag. The
/// tag's only novel artifact, the adapter, is routed to the shared family
/// bucket by `manifest::storage_path` before this identity is consulted.
///
/// Takes the exact manifest name (already canonical inside `storage_path`),
/// deliberately not `resolve_model_name`, so the mapping stays a pure lookup.
///
/// Deliberate consequence: after a Turbo-only pull, the base tag also reads
/// as installed everywhere presence is the authority (models list, removal
/// ownership, `AlreadyAvailable`), because its complete sha-verified file set
/// genuinely is on disk. A Turbo-only install and an explicit base+Turbo
/// install are byte-identical, so no rule can treat them differently without
/// inventing a new installation-receipt mechanism; the honest reading is
/// that the base IS installed and runnable. Full cleanup is therefore two
/// removals — the Turbo tag, then the base — and each removal reports the
/// kept files with the tags that still use them.
pub fn storage_identity(name: &str) -> &str {
    if REVIEWED_TURBO_MANIFEST_TIERS
        .iter()
        .any(|tier| tier.model == name)
    {
        // Each task partition owns its own ~41 GB base stack, so a Turbo tag
        // collapses onto the base of the task it renders, never onto FL2VA's.
        match task_for_model(name) {
            Some(task) => base_compact_model_for_task(task),
            None => name,
        }
    } else {
        name
    }
}

/// Resolve the reviewed Turbo tier a model identity selects, if any.
pub fn turbo_tier_for_model(model: &str) -> Option<&'static TurboManifestTier> {
    let canonical = resolve_model_name(model)?;
    REVIEWED_TURBO_MANIFEST_TIERS
        .iter()
        .find(|tier| tier.model == canonical)
}

pub const DEFAULT_WIDTH: u32 = 1344;
pub const DEFAULT_HEIGHT: u32 = 768;
/// The spatial stride one packed video row covers.
///
/// The visual VAE compresses 16x spatially (`spatial_downsample_factors`
/// `[2, 2, 2, 2, 1, 1]`) and the DiT then packs `patch_size [1, 2, 2]` latent
/// cells into one token, so one packed row is a 32x32 pixel cell. Admission,
/// the runtime envelope, the reference prepared-shape resolver, and the
/// canvas rule all derive from this one constant instead of restating `32`.
pub const VIDEO_ROW_STRIDE: u32 = 32;
const _: () = assert!(VIDEO_ROW_STRIDE == DIMENSION_ALIGNMENT);

/// The Qwen3-VL conditioner's vision patch, in pixels, and the spatial merge
/// that folds a 2x2 block of those patches into ONE token of its language
/// sequence.
///
/// Two row counts follow from this pair, and they are named apart on purpose:
///
/// * [`qwen_vision_patch_rows_per_block`] — pre-merge ViT patches on the
///   16-px grid. This is what the processor's `pixel_values.dim(0)` counts,
///   what the runtime compares its frozen admission against, and what
///   `qwen_vision_rows` means on every H3 authority: FL2VA's reviewed 4,032 at
///   1344x768 is this count, and the observed Qwen activation workspace was
///   measured over it.
/// * [`qwen_vision_pad_rows_per_block`] — merged pads on the 32-px grid, the
///   part of the conditioner's TEXT sequence a visual occupies beside the
///   prompt and its labels. One pad is one packed video row, which is why
///   both read [`VIDEO_ROW_STRIDE`].
///
/// Counting the first on the second's grid charged a 2048-square image
/// reference 4,096 vision rows at admission while the runtime prepared
/// 16,384, and held every Ref2VA image reference at execution (#1418).
pub const QWEN_VISION_PATCH_PIXELS: u32 = 16;
pub const QWEN_VISION_SPATIAL_MERGE: u32 = 2;
const _: () = assert!(QWEN_VISION_PATCH_PIXELS * QWEN_VISION_SPATIAL_MERGE == VIDEO_ROW_STRIDE);

/// Merged Qwen vision pads one normalized visual canvas occupies in the
/// conditioner's text sequence per temporal block.
pub const fn qwen_vision_pad_rows_per_block(width: u32, height: u32) -> u64 {
    (width / VIDEO_ROW_STRIDE) as u64 * (height / VIDEO_ROW_STRIDE) as u64
}

/// Pre-merge Qwen ViT patches one normalized visual canvas packs per temporal
/// block — the `qwen_vision_rows` charge. Exactly
/// `QWEN_VISION_SPATIAL_MERGE^2` times the pads for every 32-aligned canvas.
pub const fn qwen_vision_patch_rows_per_block(width: u32, height: u32) -> u64 {
    (width / QWEN_VISION_PATCH_PIXELS) as u64 * (height / QWEN_VISION_PATCH_PIXELS) as u64
}

/// The pixel ceiling a compact canvas must stay inside.
///
/// This is the qualifying campaign's own canvas area (1344x768). It is a real
/// measurement boundary rather than a preset property: `public_runtime_bounds`
/// was captured at exactly this shape, so a canvas above it would be priced by
/// an extrapolation nothing measured. Keeping it as the AREA ceiling — rather
/// than as an exact size — is what lets any 32-aligned canvas of the same or
/// smaller area be scored by the same linear packed-row model.
pub const COMPACT_MAX_PIXELS: u64 = DEFAULT_WIDTH as u64 * DEFAULT_HEIGHT as u64;

/// The shortest axis an admitted compact canvas may have.
///
/// 256 px is 8 packed rows on [`VIDEO_ROW_STRIDE`], which keeps every stage
/// that consumes the canvas well clear of a degenerate shape: the visual VAE
/// downsamples 16x (16 latent cells on the short axis), its temporal decode
/// chunks are full-canvas rather than spatially tiled so no tile size is
/// involved, and the conditioner normalizes the boundary endpoint onto the
/// same canvas. Nothing in the pipeline asks for more; the floor exists so a
/// 32x32 request cannot reach the runtime at all.
pub const MIN_COMPACT_AXIS_PIXELS: u32 = 256;

/// Canvas PRESETS the compact stack recommends, DEFAULT FIRST.
///
/// This is a recommendation list, never a gate: [`is_admitted_compact_canvas`]
/// is the rule, and every entry here must satisfy it (pinned by a test). Two
/// entries are real hardware campaigns on the 24 GB RTX 4090-class tier at 124
/// frames / 24 fps, recorded in `docs/qualification/minimax-h3.md`:
///
/// * `1344x768` — #827, 2026-08-19 (21 steps, 1216 s; `-turbo-8step`
///   759.5 s), the original qualification and the shape every memory bound in
///   `private_server.rs` was measured at.
/// * `768x768` — #1033, 2026-08-23 (21 steps, 937 s, 7.37 GiB VRAM high
///   water; `-turbo-8step` 664 s, 9.15 GiB).
///
/// The rest are ordinary aligned canvases inside the ceiling, offered so a
/// client has a ladder to pick from; they are admitted by the rule like any
/// other request and priced by the request-scaled envelope, not by a campaign.
///
/// The order is load-bearing in one place only: the default must be first, so
/// a client reading `recommended_dimensions` offers it first.
pub const REVIEWED_COMPACT_CANVASES: &[(u32, u32)] = &[
    (DEFAULT_WIDTH, DEFAULT_HEIGHT),
    (1280, 704),
    (1024, 576),
    (960, 960),
    (768, 768),
    (768, 1344),
    (704, 1280),
];

/// Whether the compact stack admits an exact canvas.
///
/// This replaced set membership in `REVIEWED_COMPACT_CANVASES`. The four
/// clauses are each a property something downstream needs:
///
/// * both axes on [`VIDEO_ROW_STRIDE`], because the packed-row arithmetic is
///   `(width / 32) * (height / 32)` and a remainder would silently truncate;
/// * both axes at least [`MIN_COMPACT_AXIS_PIXELS`];
/// * area within [`COMPACT_MAX_PIXELS`], the campaign's own canvas area, which
///   is what keeps the linear workspace scaling an interpolation;
/// * aspect inside the family's [`MIN_ASPECT_RATIO`]/[`MAX_ASPECT_RATIO`].
///
/// The area ceiling has a second consequence worth stating: `(w/32)*(h/32)` is
/// `w*h/1024`, so no admitted canvas can pack more than
/// `COMPACT_MAX_PIXELS / 1024` = 1,008 rows per latent frame — which is
/// exactly the default canvas's own figure, so the conditioning row ceilings
/// measured there remain ceilings for every admitted canvas.
pub fn is_admitted_compact_canvas(width: u32, height: u32) -> bool {
    if width == 0 || height == 0 {
        return false;
    }
    if !width.is_multiple_of(VIDEO_ROW_STRIDE) || !height.is_multiple_of(VIDEO_ROW_STRIDE) {
        return false;
    }
    if width < MIN_COMPACT_AXIS_PIXELS || height < MIN_COMPACT_AXIS_PIXELS {
        return false;
    }
    if u64::from(width) * u64::from(height) > COMPACT_MAX_PIXELS {
        return false;
    }
    let aspect = f64::from(width) / f64::from(height);
    (MIN_ASPECT_RATIO..=MAX_ASPECT_RATIO).contains(&aspect)
}

/// The compact canvas rule's area ceiling, as a client-facing number.
pub const fn reviewed_compact_max_pixels() -> u64 {
    COMPACT_MAX_PIXELS
}

/// The longest single axis the compact canvas rule can admit.
///
/// Derived from the rule rather than from the preset list: the widest legal
/// canvas is the one whose aspect sits exactly on [`MAX_ASPECT_RATIO`] and
/// whose area sits exactly on [`COMPACT_MAX_PIXELS`] — 2016x512. Advertising
/// the presets' own maximum instead would hand a client a ceiling smaller than
/// a canvas admission accepts.
pub fn reviewed_compact_max_axis_pixels() -> u32 {
    let mut best = MIN_COMPACT_AXIS_PIXELS;
    let mut axis = MIN_COMPACT_AXIS_PIXELS;
    while u64::from(axis) * u64::from(MIN_COMPACT_AXIS_PIXELS) <= COMPACT_MAX_PIXELS {
        if is_admitted_compact_canvas(axis, shortest_admitted_partner(axis)) {
            best = axis;
        }
        axis += VIDEO_ROW_STRIDE;
    }
    best
}

/// The shortest 32-aligned partner an axis could legally take under the aspect
/// bound alone (the area bound is checked by the caller).
fn shortest_admitted_partner(axis: u32) -> u32 {
    let by_aspect = (f64::from(axis) / MAX_ASPECT_RATIO).ceil() as u32;
    let floor = by_aspect.max(MIN_COMPACT_AXIS_PIXELS);
    floor.div_ceil(VIDEO_ROW_STRIDE) * VIDEO_ROW_STRIDE
}

/// The shortest single axis the compact canvas rule admits.
pub const fn reviewed_compact_min_axis_pixels() -> u32 {
    MIN_COMPACT_AXIS_PIXELS
}

/// The aspect bounds the compact canvas rule admits — the family's own.
pub fn reviewed_compact_aspect_bounds() -> (f64, f64) {
    (MIN_ASPECT_RATIO, MAX_ASPECT_RATIO)
}

/// The largest admitted compact canvas whose aspect matches a source's.
///
/// The compact rule admits any aligned canvas inside the area ceiling, so
/// fitting a source no longer means picking the nearest of a couple of fixed
/// sizes — it means rendering the source's own shape as large as the ceiling
/// allows. A 16:9 source is no longer letterboxed into 7:4 and a square one is
/// no longer letterboxed into 16:9.
///
/// The search walks the SHORT axis down from the ideal and derives the long
/// axis from the aspect each time, rather than rounding both axes
/// independently: independent rounding drifts the aspect (and turns a square
/// source into a rectangle as soon as the first candidate overshoots the
/// ceiling). The aspect is clamped into the family bounds first, so a 10:1
/// source renders 4:1 rather than being refused.
pub fn largest_admitted_compact_canvas(width: u32, height: u32) -> (u32, u32) {
    if width == 0 || height == 0 {
        return (DEFAULT_WIDTH, DEFAULT_HEIGHT);
    }
    let aspect = (f64::from(width) / f64::from(height)).clamp(MIN_ASPECT_RATIO, MAX_ASPECT_RATIO);
    let width_is_long = aspect >= 1.0;
    let ratio = if width_is_long { aspect } else { 1.0 / aspect };
    let stride = f64::from(VIDEO_ROW_STRIDE);
    let snap = |axis: f64| {
        ((axis / stride).round_ties_even().max(1.0) as u32).saturating_mul(VIDEO_ROW_STRIDE)
    };
    let mut short = snap((COMPACT_MAX_PIXELS as f64 / ratio).sqrt());
    while short >= MIN_COMPACT_AXIS_PIXELS {
        let long = snap(f64::from(short) * ratio);
        let candidate = if width_is_long {
            (long, short)
        } else {
            (short, long)
        };
        if is_admitted_compact_canvas(candidate.0, candidate.1) {
            return candidate;
        }
        short -= VIDEO_ROW_STRIDE;
    }
    (DEFAULT_WIDTH, DEFAULT_HEIGHT)
}

/// The canvas PRESETS one concrete model identity recommends, or `None` when
/// the identity keeps the family's flexible resolver (the hidden official BF16
/// references).
///
/// These are recommendations. The gate is [`valid_dimensions_for_model`].
pub fn qualified_canvases_for_model(family: &str, model: &str) -> Option<&'static [(u32, u32)]> {
    uses_reviewed_compact_envelope(family, model).then_some(REVIEWED_COMPACT_CANVASES)
}

/// [`valid_frame_count_for_model`]'s spatial twin: a compact tag renders any
/// canvas [`is_admitted_compact_canvas`] accepts; every other H3 identity takes
/// the family's alignment/area/aspect envelope.
pub fn valid_dimensions_for_model(family: &str, model: &str, width: u32, height: u32) -> bool {
    if uses_reviewed_compact_envelope(family, model) {
        is_admitted_compact_canvas(width, height)
    } else {
        true
    }
}

/// [`recommended_dimensions`], narrowed by the concrete model. Repairing a
/// stale or off-envelope canvas for a compact tag must land on a canvas that
/// runs, not on the family resolver's free-form answer.
pub fn recommended_dimensions_for_model(
    family: &str,
    model: &str,
    width: u32,
    height: u32,
) -> (u32, u32) {
    if uses_reviewed_compact_envelope(family, model) {
        largest_admitted_compact_canvas(width, height)
    } else {
        recommended_dimensions(width, height)
    }
}
pub const DEFAULT_STEPS: u32 = 50;
pub const COMFY_DEFAULT_STEPS: u32 = 21;
pub const FIXED_FPS: u32 = 24;
pub const MIN_DURATION_SECONDS: u32 = 4;
pub const MAX_DURATION_SECONDS: u32 = 15;
/// Lowest `17n+5` frame count that reaches 4 seconds at the fixed 24 FPS
/// rate.
///
/// The published model card states an output duration of **4-15 seconds**
/// (`MiniMax-AI/MiniMax-H3` `README.md:73`), and 107 frames is 4.458 s — the
/// first grid point at or above that floor, since 90 frames is only 3.75 s.
/// This was 124 while [`MIN_DURATION_SECONDS`] said 5, which refused the
/// 107-frame clip the model card permits.
///
/// This is the FAMILY floor. It is deliberately **not** the reviewed compact
/// runtime's frame count — see [`REVIEWED_COMPACT_FRAMES`], which is exactly
/// 124 and stayed there. The two were one constant until they were separated,
/// which is only invisible while they happen to be equal.
pub const MIN_FRAMES: u32 = 107;
/// Highest `17n+5` frame count that remains within 15 seconds at the fixed
/// 24 FPS rate. The next grid value, 362, is 15.083 seconds and is rejected by
/// the pinned upstream Diffusers oracle.
pub const MAX_FRAMES: u32 = 345;
/// The compact stack's DEFAULT clip length — not a gate.
///
/// It was an equality gate: the runtime envelope validated `frames` against it
/// and the generation profile pinned frames as a fixed control, so 124 was the
/// only clip a compact tag could render. It is now the default alone, and a
/// compact tag takes the family grid ([`valid_frame_count`], 107..=345 on
/// `17n+5`) like every other H3 identity. Its shape is still special in one
/// respect: `public_runtime_bounds` in `private_server.rs` was MEASURED at 124
/// frames on the default canvas, and every other clip length is priced by
/// scaling that measurement with the request's own packed-row count.
///
/// The name is retained because ~40 call sites read it as `unwrap_or` for an
/// absent `frames`; [`DEFAULT_COMPACT_FRAMES`] is the honest alias.
pub const REVIEWED_COMPACT_FRAMES: u32 = 124;
/// [`REVIEWED_COMPACT_FRAMES`] under the name that describes what it is.
pub const DEFAULT_COMPACT_FRAMES: u32 = REVIEWED_COMPACT_FRAMES;
/// The fewest steps a compact request may ask for.
///
/// The sampler builds a sigma GRID whose terminal point is zero, so two points
/// is one denoise evaluation and one point is not a schedule at all
/// (`H3DualSchedule::new` refuses `grid_points < 2`). A floor of 1 would be
/// refused by arithmetic rather than by contract.
pub const COMPACT_MIN_STEPS: u32 = 2;
/// The most steps a compact request may ask for.
///
/// The upstream schedule imposes no smaller bound — Comfy's
/// `BasicScheduler("simple")` samples a 1,000-entry flow table, so it degrades
/// only past 1,001 grid points — so this is the released model's own default
/// step count ([`DEFAULT_STEPS`]) used as the ceiling: past it a render costs
/// more time than any reviewed configuration and buys nothing anyone measured.
pub const COMPACT_MAX_STEPS: u32 = DEFAULT_STEPS;
pub const FRAME_STEP: u32 = 17;
pub const FRAME_OFFSET: u32 = 5;
pub const DIMENSION_ALIGNMENT: u32 = 32;
/// Short edge used by the released checkpoint's aspect-ratio resolver.
pub const CANVAS_SHORT_EDGE: u32 = 768;
/// Pre-rounding area budget used by the released checkpoint.
pub const CANVAS_MAX_PIXELS: u64 = 768 * 1344;
/// Largest post-rounding canvas produced by the official resolver over the
/// supported 1:4 through 4:1 aspect range.
///
/// The official algorithm applies the area budget first, then rounds both axes
/// independently to 32 pixels. That final rounding is explicitly allowed to
/// exceed [`CANVAS_MAX_PIXELS`]; the 576x1856 (and transposed) canvas is the
/// largest result at 1,069,056 pixels. Advertising the pre-rounding budget as a
/// hard request ceiling would reject a canvas produced by the oracle itself.
pub const MAX_PIXELS: u64 = 576 * 1856;
pub const MIN_ASPECT_RATIO: f64 = 0.25;
pub const MAX_ASPECT_RATIO: f64 = 4.0;
pub const NATIVE_BATCH_SIZES: &[u32] = &[1];
pub const CONDITION_POSTERIOR_SEED: u64 = 42;
pub const AUDIO_SAMPLE_RATE_HZ: u32 = 32_000;
pub const AUDIO_CHANNELS: u32 = 2;
pub const MAX_REFERENCE_IMAGES: usize = 9;
pub const MAX_REFERENCE_VIDEOS: usize = 3;
pub const MAX_REFERENCE_AUDIOS: usize = 3;
pub const MAX_REFERENCE_FILES: usize = 12;
pub const MIN_REFERENCE_DURATION_MS: u64 = 2_000;
pub const MAX_REFERENCE_DURATION_MS: u64 = 15_000;
pub const MAX_AGGREGATE_REFERENCE_VIDEO_MS: u64 = 15_000;
pub const MAX_AGGREGATE_REFERENCE_AUDIO_MS: u64 = 15_000;
/// Keep decoded inline reference data safely below the server's 64 MiB JSON
/// body ceiling after base64 expansion and request overhead. Larger media must
/// use the request-scoped upload or trusted-server-path authority.
pub const MAX_INLINE_REFERENCE_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_REFERENCE_UPLOAD_HANDLE_BYTES: usize = 256;
pub const MAX_REFERENCE_PATH_BYTES: usize = 4096;
pub const MAX_REFERENCE_NAME_BYTES: usize = 255;
/// Version 3 stops upscaling reference media: images scale DOWN to the 2048
/// short edge and videos DOWN to the 768-short-edge area-capped canvas, but a
/// source already inside those bounds keeps its native geometry (ComfyUI
/// `comfy_extras/nodes_minimax_h3.py:298-303` — `min(1.0, 2048/short)`,
/// "never upscaled" — and `:318-323`; the released processor config bounds
/// pixel AREA, `shortest_edge: 65536`/`longest_edge: 16777216`, and forces no
/// short edge). Version 2's unconditional scale blew two ~600x1200 phone
/// photos up to 2048x4224 canvases — 33k ViT patches each — and held the
/// render on an 82.7 GB host demand no 64 GB box can meet.
pub const REFERENCE_PREPROCESS_VERSION: u32 = 3;
pub const MAX_REFERENCE_DIMENSION: u32 = 65_535;
pub const MAX_REFERENCE_IMAGE_PIXELS: u64 = 100_000_000;
pub const MAX_REFERENCE_FPS: f64 = 240.0;
pub const MAX_REFERENCE_SAMPLE_RATE: u32 = 384_000;
pub const MAX_REFERENCE_CHANNELS: u16 = 32;

/// Payload-free, checked preprocessing authority used by placement, admission,
/// and durable output metadata. The counts describe the exact packed rows that
/// the versioned preprocessing policy will produce; callers must recompute this
/// from content-sniffed facts before admission rather than trusting the wire.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
pub struct GenerationReferencePreparedShape {
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalized_width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalized_height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalized_video_frames: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_frames: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qwen_video_frames: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio_samples_per_channel: Option<u64>,
    pub visual_rows: u64,
    pub audio_rows: u64,
}

/// Seed-domain version. Changing stream names/order or seed derivation must
/// mint a new version rather than silently changing seeded outputs.
pub const NOISE_DOMAIN_VERSION: &str = "mold.minimax-h3.noise.v1";
pub const NOISE_STREAMS: &[&str] = &[
    "condition-posterior",
    "condition-noise",
    "target-video",
    "target-audio",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseSeedSource {
    /// A fresh generator is recreated for each visual condition.
    FixedFreshPerVisualCondition(u64),
    /// The generator seeded from the request, shared in the declared draw order.
    RequestSeed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseDrawCardinality {
    PerVisualCondition,
    Once,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoiseDrawContract {
    pub name: &'static str,
    pub seed_source: NoiseSeedSource,
    pub cardinality: NoiseDrawCardinality,
}

/// Ordering is part of the seeded-output contract. The condition posterior is
/// deliberately outside the request generator: official preprocessing creates
/// a fresh seed-42 generator for every visual condition. Condition noise then
/// consumes the request generator in packed-condition order, followed by target
/// video and target audio noise.
pub const NOISE_DRAWS: &[NoiseDrawContract] = &[
    NoiseDrawContract {
        name: NOISE_STREAMS[0],
        seed_source: NoiseSeedSource::FixedFreshPerVisualCondition(CONDITION_POSTERIOR_SEED),
        cardinality: NoiseDrawCardinality::PerVisualCondition,
    },
    NoiseDrawContract {
        name: NOISE_STREAMS[1],
        seed_source: NoiseSeedSource::RequestSeed,
        cardinality: NoiseDrawCardinality::PerVisualCondition,
    },
    NoiseDrawContract {
        name: NOISE_STREAMS[2],
        seed_source: NoiseSeedSource::RequestSeed,
        cardinality: NoiseDrawCardinality::Once,
    },
    NoiseDrawContract {
        name: NOISE_STREAMS[3],
        seed_source: NoiseSeedSource::RequestSeed,
        cardinality: NoiseDrawCardinality::Once,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    Fl2va,
    Ref2va,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    OfficialBf16,
    ComfyPrunedInt8ConvrotNvfp4Awq,
    /// Pruned NVFP4 transformer beside the same NVFP4-AWQ conditioner.
    ///
    /// Recognized end to end as a model identity — manifest, provenance,
    /// artifact dtypes, storage, download, removal — and refused at the
    /// route, because no engine arm reads this weight layout.
    /// [`base_compact_model`] answers `None` for it, which is what stops
    /// admission before any checkpoint is opened.
    ComfyPrunedNvfp4ConvrotNvfp4Awq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    TextToAudioVideo,
    FirstFrameToAudioVideo,
    LastFrameToAudioVideo,
    FirstAndLastFrameToAudioVideo,
    ReferenceToAudioVideo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendQualification {
    /// The implementation target is CUDA, but no runnable engine is registered.
    ContractTarget,
    /// The backend is part of the supported public runtime.
    Supported,
    /// The execution path exists and is qualified for correctness only;
    /// throughput is deliberately unqualified. This mirrors
    /// `mold_inference::BackendQualification::CorrectnessOnly`, the tier Wan
    /// (#800) and LTX-2 landed Metal on before any perf UAT.
    CorrectnessOnly,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendApplicability {
    pub cuda: BackendQualification,
    pub metal: BackendQualification,
    pub cpu: BackendQualification,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    pub runtime_available: bool,
    pub backends: BackendApplicability,
    pub native_batch_sizes: &'static [u32],
    pub modes: &'static [Mode],
    pub synchronized_audio: bool,
    pub audio_disable_supported: bool,
    pub audio_sample_rate_hz: u32,
    pub audio_channels: u32,
    pub fixed_fps: u32,
    pub min_duration_seconds: u32,
    pub max_duration_seconds: u32,
    pub frame_step: u32,
    pub frame_offset: u32,
    pub min_frames: u32,
    pub max_frames: u32,
    pub dimension_alignment: u32,
    pub default_dimensions: (u32, u32),
    pub min_aspect_ratio: (u32, u32),
    pub max_aspect_ratio: (u32, u32),
    pub max_pixels: u64,
    pub noise_domain_version: &'static str,
    pub noise_streams: &'static [&'static str],
    pub noise_draws: &'static [NoiseDrawContract],
}

const FL2VA_MODES: &[Mode] = &[
    Mode::TextToAudioVideo,
    Mode::FirstFrameToAudioVideo,
    Mode::LastFrameToAudioVideo,
    Mode::FirstAndLastFrameToAudioVideo,
];
const REF2VA_MODES: &[Mode] = &[Mode::ReferenceToAudioVideo];
pub const ALL_MODES: &[Mode] = &[
    Mode::TextToAudioVideo,
    Mode::FirstFrameToAudioVideo,
    Mode::LastFrameToAudioVideo,
    Mode::FirstAndLastFrameToAudioVideo,
    Mode::ReferenceToAudioVideo,
];

pub const fn capabilities(task: Task) -> Capabilities {
    let fl2va_runtime = engine_is_built() && task_runtime_available(task);
    Capabilities {
        runtime_available: fl2va_runtime,
        backends: BackendApplicability {
            cuda: if fl2va_runtime {
                BackendQualification::Supported
            } else {
                BackendQualification::ContractTarget
            },
            // The Apple Silicon execution path landed in #1164: family-scoped
            // BF16, a folded audio-VAE reduction, chunked dense attention, the
            // portable INT8 arm, and fp8 refused by name. It is advertised as
            // correctness-only and stays that way until performance UAT, per
            // the Wan #800 precedent.
            metal: BackendQualification::CorrectnessOnly,
            cpu: BackendQualification::Unsupported,
        },
        native_batch_sizes: NATIVE_BATCH_SIZES,
        modes: match task {
            Task::Fl2va => FL2VA_MODES,
            Task::Ref2va => REF2VA_MODES,
        },
        synchronized_audio: true,
        audio_disable_supported: false,
        audio_sample_rate_hz: AUDIO_SAMPLE_RATE_HZ,
        audio_channels: AUDIO_CHANNELS,
        fixed_fps: FIXED_FPS,
        min_duration_seconds: MIN_DURATION_SECONDS,
        max_duration_seconds: MAX_DURATION_SECONDS,
        frame_step: FRAME_STEP,
        frame_offset: FRAME_OFFSET,
        min_frames: MIN_FRAMES,
        max_frames: MAX_FRAMES,
        dimension_alignment: DIMENSION_ALIGNMENT,
        default_dimensions: (DEFAULT_WIDTH, DEFAULT_HEIGHT),
        min_aspect_ratio: (1, 4),
        max_aspect_ratio: (4, 1),
        max_pixels: MAX_PIXELS,
        noise_domain_version: NOISE_DOMAIN_VERSION,
        noise_streams: NOISE_STREAMS,
        noise_draws: NOISE_DRAWS,
    }
}

/// Exact per-model capability authority for later server/surface advertising.
///
/// Compact acquisition manifests may be serialized without implying runtime
/// support. Callers that advertise *runnable* models must use
/// [`runnable_capability_contract_for_model`], which returns `None` until an
/// engine is registered and qualified. Keeping task and layout in this typed
/// value prevents UI/server code from guessing modes from family strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCapabilityContract {
    pub canonical_model: &'static str,
    pub task: Task,
    pub layout: Layout,
    pub generation: Capabilities,
}

/// Whether mold ships an engine arm that can read this weight layout.
///
/// [`capabilities`] is keyed on the task alone, because for every layout that
/// predates this one the task was the whole question. It is not any more: the
/// pruned NVFP4 transformer is an FL2VA/Ref2VA checkpoint mold can download,
/// verify, and store while having no loader for its linears. Without this
/// narrowing, `minimax-h3-fl2va:comfy-pruned-nvfp4` would advertise
/// `runtime_available: true` on a build whose route refuses it.
pub const fn layout_runtime_available(layout: Layout) -> bool {
    match layout {
        Layout::ComfyPrunedInt8ConvrotNvfp4Awq => true,
        // `official-bf16` is a qualification reference, not a runnable
        // checkpoint: `base_compact_model` answers `None` for it, so nothing
        // ever reaches the loader arms that name it. Saying `true` here was a
        // latent contradiction that only stayed invisible because the catalog
        // asked `base_compact_model` instead of this authority (#1276).
        Layout::OfficialBf16 | Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq => false,
    }
}

/// Whether this binary links the MiniMax H3 engine at all.
///
/// Split out of [`capabilities`] so a refusal can name *which* of the three
/// obstacles applies. Only the sm89 Linux and macOS Metal release recipes
/// enable `h3`; sm86, sm100, sm120, and Windows ship the catalog rows without
/// the engine, which is the case #1276 exists for.
pub const fn engine_is_built() -> bool {
    cfg!(feature = "h3")
}

/// Whether mold implements an execution path for this task partition.
///
/// Both released partitions execute since #825: FL2VA's boundary-endpoint
/// route and Ref2VA's ordered-reference route each carry their own compiled
/// runtime qualification in `mold_inference`, and `reviewed_h3_private_
/// runtime_available_for_task` is the gate that pairs with this one. The
/// variant survives because it is the axis a FUTURE task partition would be
/// refused on; it is deliberately not deleted along with its last `false`.
pub const fn task_runtime_available(task: Task) -> bool {
    matches!(task, Task::Fl2va | Task::Ref2va)
}

/// Why this build cannot execute an H3 identity.
///
/// Ordered from most to least permanent by [`runtime_availability_for`], so a
/// reason never over-promises: a task partition mold has no route for is a
/// property of every build, while "this binary was compiled without the H3
/// engine" is a property of one artifact, and naming the second where the
/// first applies would imply another artifact runs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeUnavailableReason {
    /// Mold has no engine arm that reads this checkpoint's weight layout.
    UnsupportedLayout,
    /// The task partition has no qualified runtime on any released build.
    UnsupportedTask,
    /// This binary was compiled without the H3 engine.
    EngineNotBuilt,
}

impl RuntimeUnavailableReason {
    /// One sentence naming the obstacle. Deliberately carries no license or
    /// authorization URL: nothing about any of these is the user's to resolve.
    pub const fn message(self) -> &'static str {
        match self {
            Self::UnsupportedLayout => {
                "MiniMax H3 has no runtime for this model's weight layout in this build. The \
                 checkpoint downloads and verifies normally; only generation is unavailable"
            }
            Self::UnsupportedTask => {
                "MiniMax H3 has no runtime for this model's task partition in this build. The \
                 checkpoint downloads and verifies normally; only generation is unavailable"
            }
            Self::EngineNotBuilt => {
                "This mold build was compiled without the MiniMax H3 engine. The checkpoint \
                 downloads and verifies normally; only generation is unavailable"
            }
        }
    }
}

/// Whether this build can execute one H3 identity, and why not when it cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeAvailability {
    Available,
    Unavailable(RuntimeUnavailableReason),
}

impl RuntimeAvailability {
    pub const fn is_available(self) -> bool {
        matches!(self, Self::Available)
    }

    pub const fn reason(self) -> Option<RuntimeUnavailableReason> {
        match self {
            Self::Available => None,
            Self::Unavailable(reason) => Some(reason),
        }
    }
}

/// The single authority for "can *this build* execute this H3 task/layout".
///
/// [`capabilities`] and [`generation_capabilities`] answer the same question
/// as one bool; this one names the obstacle. A test pins the two together so
/// a row, an activation refusal, and an engine registration can never
/// disagree.
pub const fn runtime_availability_for(task: Task, layout: Layout) -> RuntimeAvailability {
    if !layout_runtime_available(layout) {
        return RuntimeAvailability::Unavailable(RuntimeUnavailableReason::UnsupportedLayout);
    }
    if !task_runtime_available(task) {
        return RuntimeAvailability::Unavailable(RuntimeUnavailableReason::UnsupportedTask);
    }
    if !engine_is_built() {
        return RuntimeAvailability::Unavailable(RuntimeUnavailableReason::EngineNotBuilt);
    }
    RuntimeAvailability::Available
}

/// The single authority for "can *this build* execute this H3 model identity".
///
/// Callers must already have established that the identity is H3 (the catalog
/// asks [`is_family`] of the manifest family first). It fails closed for an
/// identity this build cannot resolve to a task and layout: an unresolvable
/// H3 name has no engine arm by definition.
pub fn model_runtime_availability(model: &str) -> RuntimeAvailability {
    match capability_contract_for_model(model) {
        Some(contract) => runtime_availability_for(contract.task, contract.layout),
        None => RuntimeAvailability::Unavailable(RuntimeUnavailableReason::UnsupportedLayout),
    }
}

fn generation_capabilities(task: Task, layout: Layout) -> Capabilities {
    let mut generation = capabilities(task);
    generation.runtime_available &= layout_runtime_available(layout);
    generation
}

pub fn capability_contract_for_model(model: &str) -> Option<ModelCapabilityContract> {
    let canonical_model = resolve_model_name(model)?;
    let task = task_for_model(canonical_model)?;
    let layout = layout_for_model(canonical_model)?;
    Some(ModelCapabilityContract {
        canonical_model,
        task,
        layout,
        generation: generation_capabilities(task, layout),
    })
}

/// Runtime-advertising boundary. This stays `None` for every H3 identity until
/// the engine, device qualification, and shipping policy are all real.
pub fn runnable_capability_contract_for_model(model: &str) -> Option<ModelCapabilityContract> {
    let contract = capability_contract_for_model(model)?;
    contract.generation.runtime_available.then_some(contract)
}

pub fn canonical_family(value: &str) -> Option<&'static str> {
    let normalized = value.trim().to_ascii_lowercase();
    FAMILY_ALIASES
        .contains(&normalized.as_str())
        .then_some(FAMILY)
}

pub fn is_family(value: &str) -> bool {
    canonical_family(value).is_some()
}

pub fn repo_revision(repo: &str) -> Option<&'static str> {
    match repo {
        OFFICIAL_REPO => Some(OFFICIAL_REVISION),
        COMFY_REPO => Some(COMFY_REVISION),
        NVFP4_REPO => Some(NVFP4_REVISION),
        LIGHTX2V_REPO => Some(LIGHTX2V_REVISION),
        DRBAPH_TURBO_LORA_REPO => Some(DRBAPH_TURBO_LORA_REVISION),
        _ => None,
    }
}

/// Pinned revision for one exact repository file.
///
/// A reviewed Turbo adapter carries its own source: Comfy-Org publishes its
/// re-hosts under `loras/` at a revision later than the compact base stack's,
/// and lightx2v publishes the 768p tiers at its own repository root. The
/// lookup keys on the exact `(repo, path)` pair the tier declares, so a
/// same-named file in a different repository is not that adapter. Every other
/// file keeps its repository-wide pinned revision.
pub fn file_revision(repo: &str, filename: &str) -> Option<&'static str> {
    if let Some(tier) = REVIEWED_TURBO_MANIFEST_TIERS
        .iter()
        .find(|tier| tier.adapter_hf_repo == repo && tier.adapter_hf_filename == filename)
    {
        return Some(tier.adapter_hf_revision);
    }
    repo_revision(repo)
}

pub fn task_for_model(model: &str) -> Option<Task> {
    let canonical = resolve_model_name(model)?;
    if canonical.starts_with("minimax-h3-ref2va:") {
        Some(Task::Ref2va)
    } else if canonical.starts_with("minimax-h3-fl2va:") {
        Some(Task::Fl2va)
    } else {
        None
    }
}

/// Whether `model` renders under the reviewed **compact** envelope: exactly
/// one canvas, one frame count, and a fixed per-tier step count.
///
/// This is the single authority for that question. `generation_profile`'s
/// recipe builder, the `/api/models` row builder, and the request validator
/// all ask it, because a row that advertises a wider envelope than the
/// profile beside it is a request the user was told would work and the
/// engine refuses after the load is paid for. `family` gates the question so
/// a non-H3 model can never answer it; within the family an unrecognized
/// identity answers `true` — fail toward the stricter envelope, the same
/// direction `presets_for_identity` takes.
pub fn uses_reviewed_compact_envelope(family: &str, model: &str) -> bool {
    is_family(family) && layout_for_model(model) != Some(Layout::OfficialBf16)
}

pub fn layout_for_model(model: &str) -> Option<Layout> {
    let canonical = resolve_model_name(model)?;
    if canonical.ends_with(":official-bf16") {
        Some(Layout::OfficialBf16)
    } else if canonical.ends_with(":comfy-pruned-nvfp4") {
        Some(Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq)
    } else if canonical.ends_with(":comfy-pruned-int8")
        || REVIEWED_TURBO_MANIFEST_TIERS
            .iter()
            .any(|tier| tier.model == canonical)
    {
        // A Turbo tag overlays a reviewed LoRA on the *same* compact INT8
        // checkpoint; nothing about the weight layout changes.
        Some(Layout::ComfyPrunedInt8ConvrotNvfp4Awq)
    } else {
        None
    }
}

pub fn resolve_model_name(input: &str) -> Option<&'static str> {
    let normalized = input.trim().to_ascii_lowercase().replace('_', "-");
    match normalized.as_str() {
        "minimax-h3" | "minimaxh3" | "minimax-h3-fl2va" => Some(FL2VA_COMFY),
        "minimax-h3-ref2va" => Some(REF2VA_COMFY),
        value if value == FL2VA_OFFICIAL => Some(FL2VA_OFFICIAL),
        value if value == REF2VA_OFFICIAL => Some(REF2VA_OFFICIAL),
        value if value == FL2VA_COMFY => Some(FL2VA_COMFY),
        value if value == REF2VA_COMFY => Some(REF2VA_COMFY),
        value if value == FL2VA_COMFY_NVFP4 => Some(FL2VA_COMFY_NVFP4),
        value if value == REF2VA_COMFY_NVFP4 => Some(REF2VA_COMFY_NVFP4),
        value if value == FL2VA_COMFY_TURBO_8STEP => Some(FL2VA_COMFY_TURBO_8STEP),
        value if value == FL2VA_COMFY_TURBO_4STEP_768P => Some(FL2VA_COMFY_TURBO_4STEP_768P),
        value if value == REF2VA_COMFY_TURBO_4STEP => Some(REF2VA_COMFY_TURBO_4STEP),
        value if value == FL2VA_COMFY_TURBO_4STEP_768P_V11 => {
            Some(FL2VA_COMFY_TURBO_4STEP_768P_V11)
        }
        value if value == FL2VA_COMFY_TURBO_8STEP_768P => Some(FL2VA_COMFY_TURBO_8STEP_768P),
        value if value == FL2VA_COMFY_TURBO_4STEP_768P_R21 => {
            Some(FL2VA_COMFY_TURBO_4STEP_768P_R21)
        }
        value if value == FL2VA_COMFY_TURBO_8STEP_R21 => Some(FL2VA_COMFY_TURBO_8STEP_R21),
        value if value == REF2VA_COMFY_TURBO_4STEP_R21 => Some(REF2VA_COMFY_TURBO_4STEP_R21),
        _ => None,
    }
}

/// Replace any released H3 alias with the exact task/layout manifest identity.
///
/// Server ingress calls this before it derives activation, upload-session,
/// admission, queue, or persistence identity. Non-H3 model names are left
/// byte-for-byte unchanged so catalog IDs and configured aliases retain their
/// existing authority.
pub fn canonicalize_request_model(request: &mut GenerateRequest) -> bool {
    let Some(canonical) = resolve_model_name(&request.model) else {
        return false;
    };
    if request.model == canonical {
        return false;
    }
    request.model = canonical.to_string();
    true
}

pub const fn valid_frame_count(frames: u32) -> bool {
    frames >= MIN_FRAMES
        && frames <= MAX_FRAMES
        && frames >= FRAME_OFFSET
        && (frames - FRAME_OFFSET).is_multiple_of(FRAME_STEP)
}

// ---------------------------------------------------------------------------
// Packed-row geometry.
//
// The prepared request's packed sequence is four row counts summed, and three
// of them are pure functions of the canvas and the clip length. That
// arithmetic was written out three times — the server's admission authority,
// `mold-inference`'s target-budget validator, and the reference prepared-shape
// resolver — and the runtime envelope transcribed its results as constants.
// While the canvas and the frame count were both pinned that was invisible;
// once a request may name either, a divergence between the envelope that
// GRANTS memory and the admission that CHARGES it is a silent over- or
// under-admit. These are the one authority all of them call.
//
// Every function returns `Option` rather than a crate error type so each
// caller keeps its own overflow reporting.
// ---------------------------------------------------------------------------

/// Latent frames the video VAE produces for a clip of `frames` pixel frames.
///
/// `(frames - 5) / 17 * 5 + 2`, the released checkpoint's own temporal
/// geometry. Only defined on the family grid; a frame count off it truncates,
/// which is exactly why [`valid_frame_count`] runs first at every door.
pub fn video_latent_frames(frames: u32) -> Option<u64> {
    let frames = u64::from(frames);
    Some(frames.checked_sub(u64::from(FRAME_OFFSET))? / u64::from(FRAME_STEP) * 5 + 2)
}

/// Packed rows one video latent frame occupies on a canvas.
///
/// `(width / stride) * (height / stride)` — see [`VIDEO_ROW_STRIDE`].
pub fn rows_per_video_latent(width: u32, height: u32) -> Option<u64> {
    let stride = u64::from(VIDEO_ROW_STRIDE);
    u64::from(width)
        .checked_div(stride)?
        .checked_mul(u64::from(height) / stride)
}

/// Total generated-video rows a request packs.
pub fn target_video_rows(width: u32, height: u32, frames: u32) -> Option<u64> {
    video_latent_frames(frames)?.checked_mul(rows_per_video_latent(width, height)?)
}

/// Audio latents per channel for a clip of `frames` pixel frames.
///
/// `round(frames / 24 * 40)` in exact integer arithmetic: `(frames * 5 + 1) / 3`.
/// Valid H3 frame counts never land on a half tie.
pub fn audio_latents_per_channel(frames: u32) -> Option<u64> {
    u64::from(frames)
        .checked_mul(5)?
        .checked_add(1)
        .map(|v| v / 3)
}

/// Total generated-audio rows a request packs.
pub fn target_audio_rows(frames: u32) -> Option<u64> {
    audio_latents_per_channel(frames)?.checked_mul(u64::from(AUDIO_CHANNELS))
}

/// Audio samples per channel implied by the clip's DURATION.
///
/// `round(frames / 24 * 32000)`, exactly: `(frames * 4000 + 1) / 3`. This is
/// the AAC mux staging size — how much audio the finished clip carries.
pub fn audio_samples_per_channel(frames: u32) -> Option<u64> {
    u64::from(frames)
        .checked_mul(4_000)?
        .checked_add(1)
        .map(|value| value / 3)
}

/// Audio samples per channel the VOCODER emits.
///
/// `audio_latents_per_channel * 800`, which is deliberately NOT
/// [`audio_samples_per_channel`]: the latent count is the rounded duration and
/// the vocoder expands each latent by a fixed 800, so the two differ by up to
/// 800 samples (267 at the default 124 frames). One is what the clip is worth,
/// the other is what the decoder produces; unifying them would make the
/// factory's prepared-request check reject every valid request.
pub fn vocoder_audio_samples_per_channel(frames: u32) -> Option<u64> {
    audio_latents_per_channel(frames)?.checked_mul(800)
}

/// The single frame count `model` may render, when its layout admits exactly
/// one.
///
/// Nothing answers `Some` today: the compact layouts used to, because their
/// runtime envelope validated the clip length by equality, and now they take
/// the family grid like every other H3 identity. The helper survives because
/// it is the shape a future single-length layout would need, and because
/// removing it would push a `None` literal into each of its callers.
pub fn fixed_frames_for_model(_family: &str, _model: &str) -> Option<u32> {
    None
}

/// [`recommended_frames`], narrowed by the concrete model.
///
/// Normalizing a stale or off-grid frame count for a compact tag snaps to the
/// family grid, which is what the runtime now admits.
pub fn recommended_frames_for_model(family: &str, model: &str, frames: u32) -> u32 {
    fixed_frames_for_model(family, model).unwrap_or_else(|| recommended_frames(frames))
}

/// [`valid_frame_count`], narrowed by the concrete model. A compact tag admits
/// exactly one clip length; every other H3 identity takes the family grid.
pub fn valid_frame_count_for_model(family: &str, model: &str, frames: u32) -> bool {
    match fixed_frames_for_model(family, model) {
        Some(fixed) => frames == fixed,
        None => valid_frame_count(frames),
    }
}

pub fn recommended_frames(frames: u32) -> u32 {
    if frames <= MIN_FRAMES {
        return MIN_FRAMES;
    }
    if frames >= MAX_FRAMES {
        return MAX_FRAMES;
    }
    let lower = FRAME_OFFSET + ((frames - FRAME_OFFSET) / FRAME_STEP) * FRAME_STEP;
    let upper = (lower + FRAME_STEP).min(MAX_FRAMES);
    if frames - lower <= upper - frames {
        lower.max(MIN_FRAMES)
    } else {
        upper
    }
}

/// The canvas an H3 request should submit when the caller attached a source
/// image without explicit dimensions. The official BF16 reference keeps its
/// flexible short-edge/area canvas ([`recommended_dimensions`]); every
/// compact layout — the base task partitions and the Turbo tiers — takes
/// [`largest_admitted_compact_canvas`], which renders the source's own aspect
/// as large as the compact area ceiling allows. A model the layout resolver
/// does not recognize also takes the compact rule: fail toward the stricter
/// contract.
///
/// This was a nearest-of-two-fixed-canvases choice, which letterboxed every
/// source that was neither 7:4 nor square. With the canvas rule in place the
/// source no longer has to be fitted into someone else's shape.
pub fn source_fit_dimensions(model: &str, width: u32, height: u32) -> (u32, u32) {
    match layout_for_model(model) {
        Some(Layout::OfficialBf16) => recommended_dimensions(width, height),
        _ => largest_admitted_compact_canvas(width, height),
    }
}

pub fn recommended_dimensions(width: u32, height: u32) -> (u32, u32) {
    if width == 0 || height == 0 {
        return (DEFAULT_WIDTH, DEFAULT_HEIGHT);
    }
    let aspect = (width as f64 / height as f64).clamp(MIN_ASPECT_RATIO, MAX_ASPECT_RATIO);
    let short_edge = f64::from(CANVAS_SHORT_EDGE);
    let (mut target_width, mut target_height) = if aspect >= 1.0 {
        (short_edge * aspect, short_edge)
    } else {
        (short_edge, short_edge / aspect)
    };
    let area = target_width * target_height;
    if area > CANVAS_MAX_PIXELS as f64 {
        let scale = (CANVAS_MAX_PIXELS as f64 / area).sqrt();
        target_width *= scale;
        target_height *= scale;
    }
    let round_axis = |axis: f64| {
        ((axis / f64::from(DIMENSION_ALIGNMENT))
            .round_ties_even()
            .max(1.0) as u32)
            * DIMENSION_ALIGNMENT
    };
    (round_axis(target_width), round_axis(target_height))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractError {
    pub code: &'static str,
    pub message: String,
    pub recommended_frames: Option<u32>,
    pub recommended_dimensions: Option<(u32, u32)>,
}

/// Structured one-based error for an ordered Ref2VA reference contract.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, utoipa::ToSchema)]
pub struct ReferenceContractError {
    pub code: &'static str,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field: Option<&'static str>,
    pub message: String,
}

impl std::fmt::Display for ReferenceContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ReferenceContractError {}

fn reference_violation(
    index: Option<usize>,
    code: &'static str,
    field: Option<&'static str>,
    message: impl Into<String>,
) -> ReferenceContractError {
    let reference = index.and_then(|index| u32::try_from(index).ok()?.checked_add(1));
    let detail = message.into();
    let message = reference.map_or_else(
        || detail.clone(),
        |reference| format!("reference {reference}: {detail}"),
    );
    ReferenceContractError {
        code,
        reference,
        field,
        message,
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_reference_provenance(
    index: usize,
    reference: &GenerationReference,
    placement_preview: bool,
) -> Result<(), ReferenceContractError> {
    let provenance = reference.provenance();
    if let Some(name) = provenance.name.as_deref() {
        let name = name.trim();
        if name.is_empty()
            || name.len() > MAX_REFERENCE_NAME_BYTES
            || name == "."
            || name == ".."
            || name.contains(['/', '\\'])
            || name.chars().any(char::is_control)
        {
            return Err(reference_violation(
                Some(index),
                "MINIMAX_H3_REFERENCE_NAME",
                Some("provenance.name"),
                "provenance name must be a display-only filename, not a path",
            ));
        }
    }
    if let Some(digest) = provenance.sha256.as_deref() {
        if !valid_sha256(digest) {
            return Err(reference_violation(
                Some(index),
                "MINIMAX_H3_REFERENCE_DIGEST",
                Some("provenance.sha256"),
                "sha256 must contain exactly 64 hexadecimal characters",
            ));
        }
    }
    if let Some(crop) = provenance.crop.as_ref() {
        let checked = match reference {
            GenerationReference::Image { width, height, .. } => {
                crop.validate_for_image(*width, *height)
            }
            GenerationReference::Video { .. } | GenerationReference::Audio { .. } => {
                Err("a crop is provenance for image references only")
            }
        };
        if let Err(reason) = checked {
            return Err(reference_violation(
                Some(index),
                "MINIMAX_H3_REFERENCE_CROP",
                Some("provenance.crop"),
                reason,
            ));
        }
    }
    match reference.media() {
        GenerationReferenceAuthority::Descriptor => {
            if !placement_preview {
                return Err(reference_violation(
                    Some(index),
                    "MINIMAX_H3_REFERENCE_DESCRIPTOR_ONLY",
                    Some("media.authority"),
                    "descriptor authority is valid only for placement preview",
                ));
            }
            if provenance.sha256.is_none() {
                return Err(reference_violation(
                    Some(index),
                    "MINIMAX_H3_REFERENCE_DIGEST_REQUIRED",
                    Some("provenance.sha256"),
                    "placement descriptors require a content sha256",
                ));
            }
        }
        GenerationReferenceAuthority::Inline { data } => {
            if placement_preview {
                return Err(reference_violation(
                    Some(index),
                    "MINIMAX_H3_REFERENCE_PREVIEW_MEDIA",
                    Some("media.authority"),
                    "placement preview accepts descriptors only, never raw reference media",
                ));
            }
            if data.is_empty() {
                return Err(reference_violation(
                    Some(index),
                    "MINIMAX_H3_REFERENCE_EMPTY",
                    Some("media.data"),
                    "inline media is empty",
                ));
            }
            if let Some(declared) = provenance.sha256.as_deref() {
                let observed = reference
                    .content_sha256()
                    .expect("inline reference always has a digest");
                if !declared.eq_ignore_ascii_case(&observed) {
                    return Err(reference_violation(
                        Some(index),
                        "MINIMAX_H3_REFERENCE_DIGEST_MISMATCH",
                        Some("provenance.sha256"),
                        "declared sha256 does not match the inline media bytes",
                    ));
                }
            }
        }
        GenerationReferenceAuthority::Upload { handle } => {
            if placement_preview {
                return Err(reference_violation(
                    Some(index),
                    "MINIMAX_H3_REFERENCE_PREVIEW_MEDIA",
                    Some("media.authority"),
                    "placement preview accepts descriptors only, never upload handles",
                ));
            }
            if handle.is_empty()
                || handle.len() > MAX_REFERENCE_UPLOAD_HANDLE_BYTES
                || !handle.bytes().all(|byte| {
                    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b':' | b'.')
                })
            {
                return Err(reference_violation(
                    Some(index),
                    "MINIMAX_H3_REFERENCE_UPLOAD_HANDLE",
                    Some("media.handle"),
                    "upload handle is malformed",
                ));
            }
            if provenance.sha256.is_none() {
                return Err(reference_violation(
                    Some(index),
                    "MINIMAX_H3_REFERENCE_DIGEST_REQUIRED",
                    Some("provenance.sha256"),
                    "upload references require a content sha256 before admission",
                ));
            }
        }
        GenerationReferenceAuthority::ServerPath { path } => {
            if placement_preview {
                return Err(reference_violation(
                    Some(index),
                    "MINIMAX_H3_REFERENCE_PREVIEW_MEDIA",
                    Some("media.authority"),
                    "placement preview accepts descriptors only, never server paths",
                ));
            }
            if path.trim().is_empty()
                || path.len() > MAX_REFERENCE_PATH_BYTES
                || path.contains('\0')
            {
                return Err(reference_violation(
                    Some(index),
                    "MINIMAX_H3_REFERENCE_PATH",
                    Some("media.path"),
                    "server path is empty or malformed",
                ));
            }
            if provenance.sha256.is_none() {
                return Err(reference_violation(
                    Some(index),
                    "MINIMAX_H3_REFERENCE_DIGEST_REQUIRED",
                    Some("provenance.sha256"),
                    "server-path references require a content sha256 before admission",
                ));
            }
        }
    }
    Ok(())
}

fn validate_reference_duration(
    index: usize,
    field: &'static str,
    duration_ms: u64,
) -> Result<(), ReferenceContractError> {
    if !(MIN_REFERENCE_DURATION_MS..=MAX_REFERENCE_DURATION_MS).contains(&duration_ms) {
        return Err(reference_violation(
            Some(index),
            "MINIMAX_H3_REFERENCE_DURATION",
            Some(field),
            format!(
                "{field} must be between {MIN_REFERENCE_DURATION_MS} and {MAX_REFERENCE_DURATION_MS} ms"
            ),
        ));
    }
    Ok(())
}

fn checked_duration_sum(
    current: u64,
    add: u64,
    index: usize,
    field: &'static str,
) -> Result<u64, ReferenceContractError> {
    current.checked_add(add).ok_or_else(|| {
        reference_violation(
            Some(index),
            "MINIMAX_H3_REFERENCE_DURATION_OVERFLOW",
            Some(field),
            "aggregate duration overflowed",
        )
    })
}

fn aligned_dimension(value: f64) -> Result<u32, ReferenceContractError> {
    if !value.is_finite() || value <= 0.0 || value > f64::from(u32::MAX) {
        return Err(reference_violation(
            None,
            "MINIMAX_H3_REFERENCE_PREPARED_SHAPE",
            Some("references"),
            "reference preprocessing produced an invalid dimension",
        ));
    }
    let aligned =
        (value / f64::from(DIMENSION_ALIGNMENT)).round_ties_even() * f64::from(DIMENSION_ALIGNMENT);
    Ok((aligned as u32).max(DIMENSION_ALIGNMENT))
}

fn reference_image_dimensions(
    width: u32,
    height: u32,
) -> Result<(u32, u32), ReferenceContractError> {
    const SHORT_EDGE: f64 = 2048.0;
    let short = f64::from(width.min(height));
    // Down-only: an image larger than the 2048-short-edge canvas scales onto
    // it; a smaller one keeps its native geometry. ComfyUI's reference node is
    // explicit — `scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))`, with
    // the input tooltip reading "never upscaled"
    // (`comfy_extras/nodes_minimax_h3.py:298-303`, `:267`) — and the released
    // processor bounds pixel AREA (`preprocessor_config.json`:
    // `shortest_edge: 65536`, `longest_edge: 16777216`), forcing no short
    // edge at all. Unconditional upscaling turned a 582x1200 phone photo into
    // a 2048x4224 canvas whose ViT patch count made the host admission demand
    // unmeetable on any 64 GB machine, and handed the conditioner a 3.5x
    // lanczos blow-up no reference implementation would produce.
    let scale = (SHORT_EDGE / short).min(1.0);
    Ok((
        aligned_dimension(f64::from(width) * scale)?,
        aligned_dimension(f64::from(height) * scale)?,
    ))
}

fn reference_video_dimensions(
    width: u32,
    height: u32,
) -> Result<(u32, u32), ReferenceContractError> {
    let ratio = f64::from(width) / f64::from(height);
    let (mut nominal_width, mut nominal_height) = if ratio >= 1.0 {
        (
            f64::from(CANVAS_SHORT_EDGE) * ratio,
            f64::from(CANVAS_SHORT_EDGE),
        )
    } else {
        (
            f64::from(CANVAS_SHORT_EDGE),
            f64::from(CANVAS_SHORT_EDGE) / ratio,
        )
    };
    let pixels = nominal_width * nominal_height;
    if pixels > CANVAS_MAX_PIXELS as f64 {
        let scale = (CANVAS_MAX_PIXELS as f64 / pixels).sqrt();
        nominal_width *= scale;
        nominal_height *= scale;
    }
    // Down-only, by area: a source video already smaller than the canvas the
    // aspect rule would hand it keeps its native geometry instead of being
    // upscaled onto it (ComfyUI `comfy_extras/nodes_minimax_h3.py:318-323` —
    // `if vw * vh < cw * ch:` round the NATIVE axes to the 32 grid).
    let native_pixels = f64::from(width) * f64::from(height);
    if native_pixels < nominal_width * nominal_height {
        nominal_width = f64::from(width);
        nominal_height = f64::from(height);
    }
    let normalized = (
        aligned_dimension(nominal_width)?,
        aligned_dimension(nominal_height)?,
    );
    Ok(normalized)
}

fn normalized_reference_video_frames(
    frame_count: u32,
    fps: f64,
    target_frames: u32,
) -> Result<(u32, u32), ReferenceContractError> {
    if frame_count == 0 || !fps.is_finite() || fps <= 0.0 || target_frames == 0 {
        return Err(reference_violation(
            None,
            "MINIMAX_H3_REFERENCE_EXACT_VIDEO_SHAPE",
            Some("frame_count"),
            "reference video requires a positive decoded frame count, fps, and target frame count",
        ));
    }
    // Diffusers reproduces ffmpeg's CFR filter by assigning source frame i to
    // floor(i * target/source + 0.5) and holding it until the next slot. The
    // resulting count telescopes to the terminal slot below; duration_ms and
    // a rounded rate cannot recover it at boundaries.
    let resampled = (f64::from(frame_count) * f64::from(FIXED_FPS) / fps + 0.5).floor();
    if !resampled.is_finite() || resampled <= 0.0 || resampled > f64::from(u32::MAX) {
        return Err(reference_violation(
            None,
            "MINIMAX_H3_REFERENCE_PREPARED_SHAPE",
            Some("frame_count"),
            "reference CFR frame count exceeds the supported range",
        ));
    }
    let normalized = (resampled as u32).min(target_frames);
    // The Qwen conditioner reads the normalized sequence before the visual
    // VAE's 17n+5 snap-down.
    let mut vae_frames = normalized;
    if vae_frames < FRAME_OFFSET {
        return Err(reference_violation(
            None,
            "MINIMAX_H3_REFERENCE_PREPARED_SHAPE",
            Some("frame_count"),
            "reference video is too short for the visual VAE",
        ));
    }
    vae_frames = ((vae_frames - FRAME_OFFSET) / FRAME_STEP)
        .max(1)
        .checked_mul(FRAME_STEP)
        .and_then(|frames| frames.checked_add(FRAME_OFFSET))
        .ok_or_else(|| {
            reference_violation(
                None,
                "MINIMAX_H3_REFERENCE_PREPARED_SHAPE",
                Some("frame_count"),
                "reference VAE frame count overflowed",
            )
        })?;
    if vae_frames > normalized {
        return Err(reference_violation(
            None,
            "MINIMAX_H3_REFERENCE_PREPARED_SHAPE",
            Some("frame_count"),
            "reference video is too short for one 17n+5 VAE chunk",
        ));
    }
    Ok((normalized, vae_frames))
}

fn exact_reference_frame_count(
    index: usize,
    frame_count: Option<u32>,
) -> Result<u32, ReferenceContractError> {
    frame_count.filter(|frames| *frames > 0).ok_or_else(|| {
        reference_violation(
            Some(index),
            "MINIMAX_H3_REFERENCE_EXACT_VIDEO_SHAPE",
            Some("frame_count"),
            "video references require the exact decoded frame_count before placement or generation",
        )
    })
}

fn audio_shape(
    index: usize,
    sample_count: Option<u64>,
    sample_rate: u32,
    target_frames: u32,
) -> Result<(u64, u64), ReferenceContractError> {
    let source_samples = sample_count.filter(|samples| *samples > 0).ok_or_else(|| {
        reference_violation(
            Some(index),
            "MINIMAX_H3_REFERENCE_EXACT_AUDIO_SHAPE",
            Some("sample_count"),
            "audio references require the exact decoded samples per channel before placement or generation",
        )
    })?;
    if sample_rate == 0 {
        return Err(reference_violation(
            Some(index),
            "MINIMAX_H3_REFERENCE_EXACT_AUDIO_SHAPE",
            Some("sample_rate"),
            "audio references require a positive decoded sample rate",
        ));
    }
    // Diffusers truncates in the decoded waveform's native sample domain
    // before its single torchaudio resample onto 32 kHz.
    let target_native_samples = u64::from(target_frames)
        .checked_mul(u64::from(sample_rate))
        .ok_or_else(|| {
            reference_violation(
                Some(index),
                "MINIMAX_H3_REFERENCE_PREPARED_SHAPE",
                Some("sample_count"),
                "target-native audio sample count overflowed",
            )
        })?
        / u64::from(FIXED_FPS);
    let truncated_native = source_samples.min(target_native_samples);
    let resampled_samples = truncated_native
        .checked_mul(u64::from(AUDIO_SAMPLE_RATE_HZ))
        .ok_or_else(|| {
            reference_violation(
                Some(index),
                "MINIMAX_H3_REFERENCE_PREPARED_SHAPE",
                Some("sample_count"),
                "reference audio resample count overflowed",
            )
        })?
        .div_ceil(u64::from(sample_rate));
    let latent_frames = resampled_samples.div_ceil(800);
    Ok((
        resampled_samples,
        latent_frames.saturating_mul(u64::from(AUDIO_CHANNELS)),
    ))
}

fn reference_prepared_shape_at(
    index: usize,
    reference: &GenerationReference,
    target_frames: u32,
) -> Result<GenerationReferencePreparedShape, ReferenceContractError> {
    let result = match reference {
        GenerationReference::Image { width, height, .. } => {
            let (width, height) = reference_image_dimensions(*width, *height)?;
            // Visual VAE downsamples 16x, then the DiT packs 2x2 latent
            // patches: one row therefore represents a 32x32 pixel cell.
            // [`VIDEO_ROW_STRIDE`] names that, and `rows_per_video_latent` is
            // the one function every consumer of it calls.
            let visual_rows = rows_per_video_latent(width, height).ok_or_else(|| {
                reference_violation(
                    Some(index),
                    "MINIMAX_H3_REFERENCE_PREPARED_SHAPE",
                    Some("width"),
                    "reference image row count overflowed",
                )
            })?;
            GenerationReferencePreparedShape {
                version: REFERENCE_PREPROCESS_VERSION,
                normalized_width: Some(width),
                normalized_height: Some(height),
                normalized_video_frames: None,
                video_frames: None,
                qwen_video_frames: None,
                audio_samples_per_channel: None,
                visual_rows,
                audio_rows: 0,
            }
        }
        GenerationReference::Video {
            width,
            height,
            frame_count,
            fps,
            has_audio,
            audio_sample_count,
            audio_sample_rate,
            ..
        } => {
            let (width, height) = reference_video_dimensions(*width, *height)?;
            let source_frames = exact_reference_frame_count(index, *frame_count)?;
            let (normalized_frames, frames) =
                normalized_reference_video_frames(source_frames, *fps, target_frames)?;
            let latent_t = if frames <= 5 {
                2
            } else {
                ((frames - 5) / FRAME_STEP) * 5 + 2
            };
            let visual_rows = u64::from(latent_t)
                .checked_mul(u64::from(width / 32))
                .and_then(|rows| rows.checked_mul(u64::from(height / 32)))
                .ok_or_else(|| {
                    reference_violation(
                        Some(index),
                        "MINIMAX_H3_REFERENCE_PREPARED_SHAPE",
                        Some("duration_ms"),
                        "reference video row count overflowed",
                    )
                })?;
            let (audio_samples_per_channel, audio_rows) = if *has_audio {
                let (samples, rows) = audio_shape(
                    index,
                    *audio_sample_count,
                    audio_sample_rate.unwrap_or_default(),
                    target_frames,
                )?;
                (Some(samples), rows)
            } else {
                (None, 0)
            };
            GenerationReferencePreparedShape {
                version: REFERENCE_PREPROCESS_VERSION,
                normalized_width: Some(width),
                normalized_height: Some(height),
                normalized_video_frames: Some(normalized_frames),
                video_frames: Some(frames),
                qwen_video_frames: Some(normalized_frames.div_ceil(FIXED_FPS / 2)),
                audio_samples_per_channel,
                visual_rows,
                audio_rows,
            }
        }
        GenerationReference::Audio {
            sample_rate,
            sample_count,
            ..
        } => {
            let (samples, audio_rows) =
                audio_shape(index, *sample_count, *sample_rate, target_frames)?;
            GenerationReferencePreparedShape {
                version: REFERENCE_PREPROCESS_VERSION,
                normalized_width: None,
                normalized_height: None,
                normalized_video_frames: None,
                video_frames: None,
                qwen_video_frames: None,
                audio_samples_per_channel: Some(samples),
                visual_rows: 0,
                audio_rows,
            }
        }
    };
    Ok(result)
}

pub fn reference_prepared_shape(
    reference: &GenerationReference,
) -> Result<GenerationReferencePreparedShape, ReferenceContractError> {
    reference_prepared_shape_at(0, reference, MAX_FRAMES)
}

/// Exact preprocessing shape for one concrete generated duration. Reference
/// video and soundtrack media are truncated to this target before their VAE
/// row counts become admission authority.
pub fn reference_prepared_shape_for_target(
    reference: &GenerationReference,
    target_frames: u32,
) -> Result<GenerationReferencePreparedShape, ReferenceContractError> {
    if !valid_frame_count(target_frames) {
        return Err(reference_violation(
            None,
            "MINIMAX_H3_REFERENCE_TARGET_FRAMES",
            Some("frames"),
            format!("target frames must use the MiniMax H3 {FRAME_STEP}n+{FRAME_OFFSET} grid"),
        ));
    }
    reference_prepared_shape_at(0, reference, target_frames)
}

pub fn reference_prepared_shapes(
    references: &[GenerationReference],
) -> Result<Vec<GenerationReferencePreparedShape>, ReferenceContractError> {
    references
        .iter()
        .enumerate()
        .map(|(index, reference)| reference_prepared_shape_at(index, reference, MAX_FRAMES))
        .collect()
}

pub fn reference_prepared_shapes_for_target(
    references: &[GenerationReference],
    target_frames: u32,
) -> Result<Vec<GenerationReferencePreparedShape>, ReferenceContractError> {
    if !valid_frame_count(target_frames) {
        return Err(reference_violation(
            None,
            "MINIMAX_H3_REFERENCE_TARGET_FRAMES",
            Some("frames"),
            format!("target frames must use the MiniMax H3 {FRAME_STEP}n+{FRAME_OFFSET} grid"),
        ));
    }
    references
        .iter()
        .enumerate()
        .map(|(index, reference)| reference_prepared_shape_at(index, reference, target_frames))
        .collect()
}

/// Validate the complete, ordered Ref2VA reference list before media decode,
/// model download, or queue mutation.
pub fn validate_references(
    references: &[GenerationReference],
) -> Result<(), ReferenceContractError> {
    validate_reference_set(references, false)
}

/// Validate the payload-free projection accepted by placement preview. Every
/// entry must retain its content digest and planning descriptors, but any raw
/// bytes, upload handle, or server-local path fails closed.
pub fn validate_reference_descriptors(
    references: &[GenerationReference],
) -> Result<(), ReferenceContractError> {
    validate_reference_set(references, true)
}

fn validate_reference_set(
    references: &[GenerationReference],
    placement_preview: bool,
) -> Result<(), ReferenceContractError> {
    if references.is_empty() {
        return Err(reference_violation(
            None,
            "MINIMAX_H3_REFERENCE_REQUIRED",
            Some("references"),
            "Ref2VA requires at least one image or video reference",
        ));
    }
    if references.len() > MAX_REFERENCE_FILES {
        return Err(reference_violation(
            None,
            "MINIMAX_H3_REFERENCE_COUNT",
            Some("references"),
            format!("Ref2VA accepts at most {MAX_REFERENCE_FILES} total references"),
        ));
    }

    let mut images = 0usize;
    let mut videos = 0usize;
    let mut audios = 0usize;
    let mut inline_bytes = 0usize;
    let mut video_duration_ms = 0u64;
    let mut audio_duration_ms = 0u64;

    for (index, reference) in references.iter().enumerate() {
        validate_reference_provenance(index, reference, placement_preview)?;
        if let GenerationReferenceAuthority::Inline { data } = reference.media() {
            inline_bytes = inline_bytes.checked_add(data.len()).ok_or_else(|| {
                reference_violation(
                    Some(index),
                    "MINIMAX_H3_REFERENCE_INLINE_BYTES",
                    Some("media.data"),
                    "aggregate inline media size overflowed",
                )
            })?;
            if inline_bytes > MAX_INLINE_REFERENCE_BYTES {
                return Err(reference_violation(
                    Some(index),
                    "MINIMAX_H3_REFERENCE_INLINE_BYTES",
                    Some("media.data"),
                    format!(
                        "aggregate inline reference media exceeds {} MiB; use an upload handle or trusted server path",
                        MAX_INLINE_REFERENCE_BYTES / (1024 * 1024)
                    ),
                ));
            }
        }

        match reference {
            GenerationReference::Image {
                mime_type,
                width,
                height,
                ..
            } => {
                images += 1;
                if !mime_type.trim().to_ascii_lowercase().starts_with("image/") {
                    return Err(reference_violation(
                        Some(index),
                        "MINIMAX_H3_REFERENCE_MEDIA_TYPE",
                        Some("mime_type"),
                        "image reference must declare an image MIME type",
                    ));
                }
                if *width == 0
                    || *height == 0
                    || *width > MAX_REFERENCE_DIMENSION
                    || *height > MAX_REFERENCE_DIMENSION
                    || u64::from(*width) * u64::from(*height) > MAX_REFERENCE_IMAGE_PIXELS
                {
                    return Err(reference_violation(
                        Some(index),
                        "MINIMAX_H3_REFERENCE_DIMENSIONS",
                        Some("width"),
                        format!(
                            "image dimensions must be positive, at most {MAX_REFERENCE_DIMENSION} pixels per axis, and at most {MAX_REFERENCE_IMAGE_PIXELS} total pixels"
                        ),
                    ));
                }
            }
            GenerationReference::Video {
                mime_type,
                width,
                height,
                duration_ms,
                fps,
                has_audio,
                audio_duration_ms: soundtrack_duration,
                audio_sample_count,
                audio_sample_rate,
                audio_channels,
                ..
            } => {
                videos += 1;
                if !mime_type.trim().to_ascii_lowercase().starts_with("video/") {
                    return Err(reference_violation(
                        Some(index),
                        "MINIMAX_H3_REFERENCE_MEDIA_TYPE",
                        Some("mime_type"),
                        "video reference must declare a video MIME type",
                    ));
                }
                if *width == 0
                    || *height == 0
                    || *width > MAX_REFERENCE_DIMENSION
                    || *height > MAX_REFERENCE_DIMENSION
                    || u64::from(*width) * u64::from(*height) > MAX_REFERENCE_IMAGE_PIXELS
                    || !fps.is_finite()
                    || *fps <= 0.0
                    || *fps > MAX_REFERENCE_FPS
                {
                    return Err(reference_violation(
                        Some(index),
                        "MINIMAX_H3_REFERENCE_VIDEO_SHAPE",
                        Some("fps"),
                        format!(
                            "video dimensions must be at most {MAX_REFERENCE_DIMENSION} pixels per axis and fps must be in (0, {MAX_REFERENCE_FPS}]"
                        ),
                    ));
                }
                validate_reference_duration(index, "duration_ms", *duration_ms)?;
                video_duration_ms =
                    checked_duration_sum(video_duration_ms, *duration_ms, index, "duration_ms")?;
                match (*has_audio, *soundtrack_duration) {
                    (true, Some(soundtrack_ms)) => {
                        validate_reference_duration(index, "audio_duration_ms", soundtrack_ms)?;
                        audio_duration_ms = checked_duration_sum(
                            audio_duration_ms,
                            soundtrack_ms,
                            index,
                            "audio_duration_ms",
                        )?;
                    }
                    (true, None) => {
                        return Err(reference_violation(
                            Some(index),
                            "MINIMAX_H3_REFERENCE_SOUNDTRACK_DURATION",
                            Some("audio_duration_ms"),
                            "video with audio must declare its soundtrack duration",
                        ));
                    }
                    (false, Some(_)) => {
                        return Err(reference_violation(
                            Some(index),
                            "MINIMAX_H3_REFERENCE_SOUNDTRACK_MISMATCH",
                            Some("audio_duration_ms"),
                            "audio_duration_ms is only valid when has_audio is true",
                        ));
                    }
                    (false, None) => {}
                }
                match (
                    *has_audio,
                    *audio_sample_count,
                    *audio_sample_rate,
                    *audio_channels,
                ) {
                    (true, Some(samples), Some(rate), Some(channels))
                        if samples > 0
                            && rate > 0
                            && rate <= MAX_REFERENCE_SAMPLE_RATE
                            && (channels == 1 || channels == AUDIO_CHANNELS as u16) => {}
                    (true, ..) => {
                        return Err(reference_violation(
                            Some(index),
                            "MINIMAX_H3_REFERENCE_EXACT_AUDIO_SHAPE",
                            Some("audio_sample_count"),
                            "video soundtracks require exact positive decoded sample_count, sample_rate, and mono/stereo channels",
                        ));
                    }
                    (false, None, None, None) => {}
                    (false, ..) => {
                        return Err(reference_violation(
                            Some(index),
                            "MINIMAX_H3_REFERENCE_SOUNDTRACK_MISMATCH",
                            Some("audio_sample_count"),
                            "soundtrack sample fields are only valid when has_audio is true",
                        ));
                    }
                }
            }
            GenerationReference::Audio {
                mime_type,
                duration_ms,
                sample_rate,
                channels,
                sample_count,
                ..
            } => {
                audios += 1;
                if !mime_type.trim().to_ascii_lowercase().starts_with("audio/") {
                    return Err(reference_violation(
                        Some(index),
                        "MINIMAX_H3_REFERENCE_MEDIA_TYPE",
                        Some("mime_type"),
                        "audio reference must declare an audio MIME type",
                    ));
                }
                validate_reference_duration(index, "duration_ms", *duration_ms)?;
                if *sample_rate == 0
                    || *sample_rate > MAX_REFERENCE_SAMPLE_RATE
                    || *channels == 0
                    || *channels > AUDIO_CHANNELS as u16
                {
                    return Err(reference_violation(
                        Some(index),
                        "MINIMAX_H3_REFERENCE_AUDIO_SHAPE",
                        Some("sample_rate"),
                        format!(
                            "audio sample rate must be in 1..={MAX_REFERENCE_SAMPLE_RATE} and channels must be mono or stereo"
                        ),
                    ));
                }
                if sample_count.is_none_or(|samples| samples == 0) {
                    return Err(reference_violation(
                        Some(index),
                        "MINIMAX_H3_REFERENCE_EXACT_AUDIO_SHAPE",
                        Some("sample_count"),
                        "audio references require the exact decoded samples per channel",
                    ));
                }
                audio_duration_ms =
                    checked_duration_sum(audio_duration_ms, *duration_ms, index, "duration_ms")?;
            }
        }
        let _ = reference_prepared_shape_at(index, reference, MAX_FRAMES)?;
    }

    for (observed, maximum, kind) in [
        (images, MAX_REFERENCE_IMAGES, "image"),
        (videos, MAX_REFERENCE_VIDEOS, "video"),
        (audios, MAX_REFERENCE_AUDIOS, "audio"),
    ] {
        if observed > maximum {
            return Err(reference_violation(
                None,
                "MINIMAX_H3_REFERENCE_KIND_COUNT",
                Some("references"),
                format!("Ref2VA accepts at most {maximum} {kind} references; received {observed}"),
            ));
        }
    }
    if images + videos == 0 {
        return Err(reference_violation(
            None,
            "MINIMAX_H3_REFERENCE_AUDIO_ONLY",
            Some("references"),
            "Ref2VA audio references require at least one image or video reference",
        ));
    }
    if video_duration_ms > MAX_AGGREGATE_REFERENCE_VIDEO_MS {
        return Err(reference_violation(
            None,
            "MINIMAX_H3_REFERENCE_VIDEO_DURATION_TOTAL",
            Some("references"),
            format!(
                "aggregate reference video duration exceeds {MAX_AGGREGATE_REFERENCE_VIDEO_MS} ms"
            ),
        ));
    }
    if audio_duration_ms > MAX_AGGREGATE_REFERENCE_AUDIO_MS {
        return Err(reference_violation(
            None,
            "MINIMAX_H3_REFERENCE_AUDIO_DURATION_TOTAL",
            Some("references"),
            format!(
                "aggregate reference audio duration exceeds {MAX_AGGREGATE_REFERENCE_AUDIO_MS} ms"
            ),
        ));
    }
    Ok(())
}

impl std::fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ContractError {}

fn violation(code: &'static str, message: impl Into<String>) -> ContractError {
    ContractError {
        code,
        message: message.into(),
        recommended_frames: None,
        recommended_dimensions: None,
    }
}

/// Validate the H3-specific part of a normalized generation request.
///
/// Public generation paths must call `require_model_activation` first.  This
/// helper is intentionally available for table tests and future engine work;
/// it does not authorize H3 or bypass the compliance gate.
pub fn validate_request_contract(req: &GenerateRequest, task: Task) -> Result<Mode, ContractError> {
    validate_request_contract_with_reference_authority(req, task, false)
}

/// Validate the internal, payload-free request retained after authenticated
/// Ref2VA ingress has resolved every media authority to a private binding.
///
/// This does not authorize H3 execution. It differs from
/// [`validate_request_contract`] only by requiring descriptor authorities for
/// Ref2VA, so a queued request never needs to carry server-local paths.
/// The reviewed-canvas gate for one concrete compact identity, as a
/// [`ContractError`] carrying its own repair.
///
/// Deliberately NOT part of [`validate_request_contract`]. That function is
/// the FAMILY/model contract — the engine's own `prepare_request` runs it,
/// and the synthetic pipeline tests that pin engine behaviour do so on tiny
/// canvases where the reviewed set is meaningless. The reviewed canvases are
/// a RUNTIME QUALIFICATION fact, so this is asked at the request door
/// instead: `validation::validate_h3_private_uat_request` for authenticated
/// private ingress (which skips generation-profile validation), and the
/// generation profile's `Buckets` + `OffBucketPolicy::Reject` for every
/// ordinary client. `private_server.rs` keeps the last word either way.
pub fn validate_reviewed_canvas(req: &GenerateRequest) -> Result<(), ContractError> {
    if valid_dimensions_for_model(FAMILY, &req.model, req.width, req.height) {
        return Ok(());
    }
    let mut error = violation(
        "MINIMAX_H3_DIMENSIONS",
        format!(
            "{} renders canvases whose axes are multiples of {VIDEO_ROW_STRIDE}, at least \
             {MIN_COMPACT_AXIS_PIXELS} px, at most {COMPACT_MAX_PIXELS} pixels in total, with \
             aspect ratio in [{MIN_ASPECT_RATIO}, {MAX_ASPECT_RATIO}]; received {}x{}",
            req.model, req.width, req.height
        ),
    );
    error.recommended_dimensions = Some(recommended_dimensions_for_model(
        FAMILY, &req.model, req.width, req.height,
    ));
    Err(error)
}

pub fn validate_resolved_request_contract(
    req: &GenerateRequest,
    task: Task,
) -> Result<Mode, ContractError> {
    validate_request_contract_with_reference_authority(req, task, true)
}

/// Media the persisted (scrubbed) form of a request no longer carries but
/// the queue's media store still holds for it.
///
/// `scrub_request_media` strips every media payload from a durable row and
/// the scheduler resolves that row, so a FL2VA job whose first frame is in
/// the encrypted media set reads as `source_image: None` — and therefore as
/// `TextToAudioVideo` — unless the resolver says what the store holds. This
/// is the H3 shape of the queue-media projection; a caller holding a hydrated
/// request passes `from_request`, which is what the worker does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResolvedMediaPresence {
    pub source_image: bool,
}

impl ResolvedMediaPresence {
    pub fn from_request(request: &GenerateRequest) -> Self {
        Self {
            source_image: request.source_image.is_some(),
        }
    }
}

/// [`validate_resolved_request_contract`] for a request whose media may have
/// been scrubbed into the queue-media store: the boundary endpoints are
/// derived from what the request carries OR what `media` says the store
/// holds for it.
pub fn validate_resolved_request_contract_with_media(
    req: &GenerateRequest,
    task: Task,
    media: ResolvedMediaPresence,
) -> Result<Mode, ContractError> {
    validate_request_contract_with_authorities(req, task, true, media)
}

/// [`validate_request_contract`] for a request whose media may have been
/// scrubbed into the queue's media store — the FL2VA twin of
/// [`validate_resolved_request_contract_with_media`].
pub fn validate_request_contract_with_media(
    req: &GenerateRequest,
    task: Task,
    media: ResolvedMediaPresence,
) -> Result<Mode, ContractError> {
    validate_request_contract_with_authorities(req, task, false, media)
}

fn validate_request_contract_with_reference_authority(
    req: &GenerateRequest,
    task: Task,
    resolved_references: bool,
) -> Result<Mode, ContractError> {
    validate_request_contract_with_authorities(
        req,
        task,
        resolved_references,
        ResolvedMediaPresence::from_request(req),
    )
}

fn validate_request_contract_with_authorities(
    req: &GenerateRequest,
    task: Task,
    resolved_references: bool,
    media: ResolvedMediaPresence,
) -> Result<Mode, ContractError> {
    let fps = req.fps.unwrap_or(FIXED_FPS);
    if fps != FIXED_FPS {
        return Err(violation(
            "MINIMAX_H3_FIXED_FPS",
            format!("MiniMax H3 requires {FIXED_FPS} fps; received {fps}"),
        ));
    }

    // An omitted frame count means the shipped default clip length, NOT the
    // family floor. Reading the floor here would interpret the same request as
    // 107 frames while the profile, the manifest default, and the renderer all
    // read 124 — so an advertised final keyframe at index 123 would be refused
    // for exceeding a duration nothing else believes in.
    let frames = req.frames.unwrap_or(REVIEWED_COMPACT_FRAMES);
    // Model-aware, because authenticated private H3 ingress bypasses
    // generation-profile validation: a compact request carrying the family
    // floor would otherwise enter admission and the queue before failing
    // against the runtime envelope, which is a refusal after the load is paid
    // for rather than at the door.
    if !valid_frame_count_for_model(FAMILY, &req.model, frames) {
        let mut error = match fixed_frames_for_model(FAMILY, &req.model) {
            Some(fixed) => violation(
                "MINIMAX_H3_FRAME_GRID",
                format!(
                    "{} renders exactly {fixed} frames; received {frames}",
                    req.model
                ),
            ),
            None => violation(
                "MINIMAX_H3_FRAME_GRID",
                format!(
                    "MiniMax H3 frames must be {FRAME_STEP}n+{FRAME_OFFSET} from {MIN_FRAMES} through {MAX_FRAMES}; received {frames}"
                ),
            ),
        };
        error.recommended_frames = Some(recommended_frames_for_model(FAMILY, &req.model, frames));
        return Err(error);
    }

    if req.width == 0
        || req.height == 0
        || !req.width.is_multiple_of(DIMENSION_ALIGNMENT)
        || !req.height.is_multiple_of(DIMENSION_ALIGNMENT)
        || u64::from(req.width) * u64::from(req.height) > MAX_PIXELS
        || !(MIN_ASPECT_RATIO..=MAX_ASPECT_RATIO).contains(&(req.width as f64 / req.height as f64))
    {
        let mut error = violation(
            "MINIMAX_H3_DIMENSIONS",
            format!(
                "MiniMax H3 dimensions must be positive multiples of {DIMENSION_ALIGNMENT}, at most {MAX_PIXELS} pixels, with aspect ratio in [{MIN_ASPECT_RATIO}, {MAX_ASPECT_RATIO}]"
            ),
        );
        error.recommended_dimensions = Some(recommended_dimensions(req.width, req.height));
        return Err(error);
    }

    if req.enable_audio == Some(false) {
        return Err(violation(
            "MINIMAX_H3_SYNCHRONIZED_AUDIO_REQUIRED",
            "MiniMax H3 always generates synchronized audio; enable_audio=false is unsupported",
        ));
    }
    if req
        .output_format
        .is_some_and(|format| format != OutputFormat::Mp4)
    {
        return Err(violation(
            "MINIMAX_H3_MP4_REQUIRED",
            "MiniMax H3 synchronized audio-video output requires mp4",
        ));
    }
    if req.guidance != 0.0 {
        return Err(violation(
            "MINIMAX_H3_NO_CFG",
            "MiniMax H3 does not use classifier-free guidance; guidance must be 0",
        ));
    }
    if req.scheduler.is_some() {
        return Err(violation(
            "MINIMAX_H3_FIXED_DUAL_SCHEDULE",
            "MiniMax H3 uses its dedicated synchronized video/audio flow schedules; generic scheduler overrides are unsupported",
        ));
    }
    if req.steps < 2 {
        return Err(violation(
            "MINIMAX_H3_GRID_POINTS",
            "MiniMax H3 steps count terminal-inclusive sigma grid points and must be at least 2",
        ));
    }
    if req
        .negative_prompt
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty())
    {
        return Err(violation(
            "MINIMAX_H3_NO_NEGATIVE_PROMPT",
            "MiniMax H3 has no negative-prompt branch",
        ));
    }
    if req.strength != 1.0 {
        return Err(violation(
            "MINIMAX_H3_FIXED_STRENGTH",
            "MiniMax H3 generation has no denoise-strength control; strength must be 1",
        ));
    }
    if req.source_video.is_some()
        || req.source_video_path.is_some()
        || req.audio_file.is_some()
        || req.audio_file_path.is_some()
        || req.retake_range.is_some()
        || req.is_extend()
    {
        return Err(violation(
            "MINIMAX_H3_CONDITIONING_UNSUPPORTED",
            "MiniMax H3 supports text, FL2VA boundary frames, or Ref2VA references; source video/audio, retake, and extend are unsupported",
        ));
    }
    if req.mask_image.is_some()
        || req.control_image.is_some()
        || req.control_model.is_some()
        || req.control_scale != 1.0
        || req.cfg_plus.is_some()
        || req.lora.is_some()
        || req.loras.as_ref().is_some_and(|items| !items.is_empty())
        || req.pipeline.is_some()
        || req.ic_lora_control.is_some()
        || req.hdr_exr_dir.is_some()
        || req.hdr_exr_full_float
        || req.spatial_upscale.is_some()
        || req.temporal_upscale.is_some()
        || req.guidance_overrides.is_some()
        || req.extend_overlap_frames.is_some()
    {
        return Err(violation(
            "MINIMAX_H3_FOREIGN_PIPELINE_FIELD",
            "MiniMax H3 does not accept mask, ControlNet, CFG+, LoRA, LTX-2 pipeline, HDR, latent upscale-stage, extend-overlap, or guidance-override fields",
        ));
    }
    if req.source_image.is_none() && !media.source_image && req.source_image_name.is_some() {
        return Err(violation(
            "MINIMAX_H3_ORPHAN_SOURCE_NAME",
            "MiniMax H3 source_image_name requires a first-frame source image",
        ));
    }

    match task {
        Task::Ref2va => {
            if req.source_image.is_some() || req.keyframes.as_ref().is_some_and(|v| !v.is_empty()) {
                return Err(violation(
                    "MINIMAX_H3_TASK_MISMATCH",
                    "Ref2VA accepts reference inputs, not FL2VA boundary frames",
                ));
            }
            if req
                .edit_images
                .as_ref()
                .is_some_and(|items| !items.is_empty())
            {
                return Err(violation(
                    "MINIMAX_H3_ORDERED_REFERENCES_REQUIRED",
                    "Ref2VA uses the ordered references contract; edit_images is not authoritative",
                ));
            }
            let references = req.references.as_deref().ok_or_else(|| {
                violation(
                    "MINIMAX_H3_REFERENCE_REQUIRED",
                    "Ref2VA requires at least one ordered reference",
                )
            })?;
            let reference_validation = if resolved_references {
                validate_reference_descriptors(references)
            } else {
                validate_references(references)
            };
            if let Err(error) = reference_validation {
                return Err(violation(error.code, error.message));
            }
            Ok(Mode::ReferenceToAudioVideo)
        }
        Task::Fl2va => {
            if req
                .references
                .as_ref()
                .is_some_and(|items| !items.is_empty())
            {
                return Err(violation(
                    "MINIMAX_H3_TASK_MISMATCH",
                    "FL2VA does not accept Ref2VA ordered references",
                ));
            }
            if req
                .edit_images
                .as_ref()
                .is_some_and(|items| !items.is_empty())
            {
                return Err(violation(
                    "MINIMAX_H3_TASK_MISMATCH",
                    "FL2VA does not accept Ref2VA reference inputs",
                ));
            }
            let last = frames - 1;
            let mut first = req.source_image.is_some() || media.source_image;
            let mut end = false;
            for keyframe in req.keyframes.as_deref().unwrap_or_default() {
                match keyframe.frame {
                    0 if !first => first = true,
                    0 => {
                        return Err(violation(
                            "MINIMAX_H3_DUPLICATE_BOUNDARY",
                            "FL2VA received more than one first-frame condition",
                        ))
                    }
                    frame if frame == last && !end => end = true,
                    frame if frame == last => {
                        return Err(violation(
                            "MINIMAX_H3_DUPLICATE_BOUNDARY",
                            "FL2VA received more than one last-frame condition",
                        ))
                    }
                    frame => {
                        return Err(violation(
                            "MINIMAX_H3_BOUNDARY_FRAME_REQUIRED",
                            format!(
                                "FL2VA keyframes may target only frame 0 or final frame {last}; received {frame}"
                            ),
                        ))
                    }
                }
            }
            Ok(match (first, end) {
                (false, false) => Mode::TextToAudioVideo,
                (true, false) => Mode::FirstFrameToAudioVideo,
                (false, true) => Mode::LastFrameToAudioVideo,
                (true, true) => Mode::FirstAndLastFrameToAudioVideo,
            })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactRole {
    TaskTransformer,
    /// Reviewed Turbo LoRA adapter overlaid on a task transformer.
    TurboLoraAdapter,
    Qwen3VlConditioner,
    VideoVae,
    AudioVae,
    Processor,
    VideoScheduler,
    AudioScheduler,
    SharedConfig,
    TaskConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManifestContract<'a> {
    pub manifest_name: &'a str,
    pub task: Task,
    pub layout: Layout,
    pub source_repo: &'static str,
    pub source_revision: &'static str,
    pub license_url: &'static str,
    pub license_sha256: &'static str,
    pub implementation_repo: &'static str,
    pub implementation_revision: &'static str,
    pub diffusers_reference_repo: &'static str,
    pub diffusers_reference_revision: &'static str,
    pub shared_identity_scheme: &'static str,
    pub runtime_available: bool,
}

pub fn manifest_contract(manifest: &ModelManifest) -> Option<ManifestContract<'_>> {
    if manifest.family != FAMILY {
        return None;
    }
    let task = task_for_model(&manifest.name)?;
    let layout = layout_for_model(&manifest.name)?;
    let (source_repo, source_revision, implementation_repo, implementation_revision) = match layout
    {
        Layout::OfficialBf16 => (
            OFFICIAL_REPO,
            OFFICIAL_REVISION,
            OFFICIAL_IMPLEMENTATION_REPO,
            OFFICIAL_IMPLEMENTATION_REVISION,
        ),
        Layout::ComfyPrunedInt8ConvrotNvfp4Awq => (
            COMFY_REPO,
            COMFY_REVISION,
            COMFY_IMPLEMENTATION_REPO,
            COMFY_IMPLEMENTATION_REVISION,
        ),
        // Only the transformer is third-party. The implementation reference
        // stays Comfy-Org: the quantization schema is comfy-kitchen's and
        // every other artifact in the stack is Comfy-Org's own.
        Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq => (
            NVFP4_REPO,
            NVFP4_REVISION,
            COMFY_IMPLEMENTATION_REPO,
            COMFY_IMPLEMENTATION_REVISION,
        ),
    };
    Some(ManifestContract {
        manifest_name: &manifest.name,
        task,
        layout,
        source_repo,
        source_revision,
        license_url: MINIMAX_H3_LICENSE_URL,
        license_sha256: LICENSE_SHA256,
        implementation_repo,
        implementation_revision,
        diffusers_reference_repo: DIFFUSERS_REFERENCE_REPO,
        diffusers_reference_revision: DIFFUSERS_REFERENCE_REVISION,
        shared_identity_scheme: "hf-repo+revision+path+sha256",
        runtime_available: generation_capabilities(task, layout).runtime_available,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactIdentity<'a> {
    pub source_repo: &'a str,
    pub source_revision: &'static str,
    pub source_path: &'a str,
    pub sha256: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactContract<'a> {
    pub identity: ArtifactIdentity<'a>,
    pub role: ArtifactRole,
    pub license_url: &'static str,
    pub license_sha256: &'static str,
    pub dtype: &'static str,
    pub shape: &'static str,
    pub compatible_tasks: &'static [Task],
}

const BOTH_TASKS: &[Task] = &[Task::Fl2va, Task::Ref2va];
const FL2VA_ONLY: &[Task] = &[Task::Fl2va];
const REF2VA_ONLY: &[Task] = &[Task::Ref2va];

pub fn artifact_contract<'a>(
    manifest: &'a ModelManifest,
    file: &'a ModelFile,
) -> Option<ArtifactContract<'a>> {
    if manifest.family != FAMILY {
        return None;
    }
    if !manifest.files.iter().any(|candidate| {
        candidate.hf_repo == file.hf_repo
            && candidate.hf_filename == file.hf_filename
            && candidate.component == file.component
            && candidate.size_bytes == file.size_bytes
            && candidate.gated == file.gated
            && candidate.sha256 == file.sha256
    }) {
        return None;
    }
    let task = task_for_model(&manifest.name)?;
    let layout = layout_for_model(&manifest.name)?;
    let compatible_tasks = if matches!(
        file.component,
        ModelComponent::Transformer
            | ModelComponent::TransformerShard
            | ModelComponent::DistilledLora
            | ModelComponent::TaskConfig
    ) {
        match task {
            Task::Fl2va => FL2VA_ONLY,
            Task::Ref2va => REF2VA_ONLY,
        }
    } else {
        BOTH_TASKS
    };
    let (role, dtype, shape) = match file.component {
        ModelComponent::Transformer | ModelComponent::TransformerShard => (
            ArtifactRole::TaskTransformer,
            match layout {
                Layout::OfficialBf16 => "bf16",
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq => "int8-convrot-pruned",
                Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq => "nvfp4-pruned",
            },
            "50 blocks; hidden=5376; 56 heads x 128",
        ),
        // The shape is a per-tier fact looked up on the exact `(repo, path)`
        // pair the tier declares — the same key `file_revision` uses — because
        // an SVD-resized derivative and the rank-128 adapter it approximates
        // are both `DistilledLora` files of the same 208 modules. A file no
        // reviewed tier claims falls back to the uniform sentence, which is
        // what every adapter mold pinned before the resized tiers existed.
        ModelComponent::DistilledLora => (
            ArtifactRole::TurboLoraAdapter,
            "bf16",
            REVIEWED_TURBO_MANIFEST_TIERS
                .iter()
                .find(|tier| {
                    tier.adapter_hf_repo == file.hf_repo
                        && tier.adapter_hf_filename == file.hf_filename
                })
                .map_or(UNIFORM_TURBO_ADAPTER_SHAPE, |tier| tier.adapter_shape_label),
        ),
        ModelComponent::TextEncoder => (
            ArtifactRole::Qwen3VlConditioner,
            match layout {
                Layout::OfficialBf16 => "bf16",
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq
                | Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq => "nvfp4-awq",
            },
            "Qwen3-VL-32B; H3 hidden-state contract",
        ),
        ModelComponent::Vae => (
            ArtifactRole::VideoVae,
            match layout {
                Layout::OfficialBf16 => "fp32",
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq
                | Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq => "fp16",
            },
            "24 latent channels; spatial /16; temporal 17->5 (+2)",
        ),
        ModelComponent::AudioVae => (
            ArtifactRole::AudioVae,
            "fp32",
            "32 kHz stereo; 40 latent rows/second",
        ),
        ModelComponent::Processor => (
            ArtifactRole::Processor,
            "data",
            "Qwen3-VL tokenizer/processor",
        ),
        ModelComponent::VideoScheduler => (
            ArtifactRole::VideoScheduler,
            "config",
            "rectified-flow schedule config; video shift=12",
        ),
        ModelComponent::AudioScheduler => (
            ArtifactRole::AudioScheduler,
            "config",
            "rectified-flow schedule config; native audio shift=3",
        ),
        ModelComponent::ModelConfig => (
            ArtifactRole::SharedConfig,
            "config",
            "official module config",
        ),
        ModelComponent::TaskConfig => (
            ArtifactRole::TaskConfig,
            "config",
            "task transformer config/index",
        ),
        _ => return None,
    };
    Some(ArtifactContract {
        identity: ArtifactIdentity {
            source_repo: &file.hf_repo,
            source_revision: file_revision(&file.hf_repo, &file.hf_filename)?,
            source_path: &file.hf_filename,
            sha256: file.sha256?,
        },
        role,
        license_url: MINIMAX_H3_LICENSE_URL,
        license_sha256: LICENSE_SHA256,
        dtype,
        shape,
        compatible_tasks,
    })
}

fn file(
    repo: &str,
    filename: &str,
    component: ModelComponent,
    size_bytes: u64,
    sha256: &'static str,
) -> ModelFile {
    ModelFile {
        hf_repo: repo.to_string(),
        hf_filename: filename.to_string(),
        component,
        size_bytes,
        gated: false,
        sha256: Some(sha256),
    }
}

const FL2VA_TRANSFORMER: &[(u64, &str)] = &[
    (
        4_825_958_704,
        "2d847200c45c09dd7f973c1b096663068408ef851ee0b3711d059b6dc5dcd028",
    ),
    (
        4_702_158_032,
        "2c4d362eddd2802180ac9c744849eb9ba8d9c8b984bdf9822cb02ed004b29184",
    ),
    (
        4_933_368_192,
        "949c5aafbbfa5654da730a6a7fafd75adb164d0857b095a30e8bb6d390887d69",
    ),
    (
        4_567_069_608,
        "eef7616790105ee839766bb2027203bf2c0d87c6aa038dca84145a8675f5ce28",
    ),
    (
        4_702_158_080,
        "43fdf42d638e8bc6745f713fae80c93bb301807a1a5ae7249344ce28e202a494",
    ),
    (
        4_933_368_232,
        "6442510b34d173653f0cce5c964b935395a8f7accf0b9cc0aa31aec59805239d",
    ),
    (
        4_567_069_608,
        "29f48f535c91dac76496ca821eeb16ca24bc4caf3f0cae8b920a89b1f966da6d",
    ),
    (
        4_702_158_080,
        "c711b096c764bd60f0b8b6ad49518bfab6d614fb788c725add8741c0674a4cd8",
    ),
    (
        4_933_368_232,
        "44428defe3976cbb87635ad200b958199e739986697cd29fdf27aeb7294b5944",
    ),
    (
        4_567_069_608,
        "3d44939c374c9da382e9c6877e1946adf7b84e08c7a881c068f228d6849411c9",
    ),
    (
        4_702_158_080,
        "224d24430b58127a5577721084e0e704a0e74ec96dd7c35bc6fc0994ebd87c33",
    ),
    (
        4_933_368_232,
        "48fa2bd8fe134eef565ab2464f1c2589a6657cba0d14283dfc06b532f8961f3c",
    ),
    (
        4_567_069_608,
        "be5b4b1809f9d546ffd4b3fcf41e5c1e02b819125caa6bc105c109b04c051bd3",
    ),
    (
        4_644_161_920,
        "8fbd5e6c1fb1df7ce988ca90f3d59e7610e465c7517e4b344eda4a214ba4b97d",
    ),
];

const REF2VA_TRANSFORMER: &[(u64, &str)] = &[
    (
        4_825_958_704,
        "7a3fcad885f51560e550b2e84c9a8d8b35e62996cfd9076937e992bd23478df9",
    ),
    (
        4_702_158_032,
        "1638ae1dc8ae26c4ba43ad28a6d851ad8983847324bb2b468719c7c81f219706",
    ),
    (
        4_933_368_192,
        "1ef3c4954ffe5a664c2e3028e2a3241190d9c159dce6ba1136002c6af1db5353",
    ),
    (
        4_567_069_608,
        "12d92f2975cfd5c5b786126385c52e5bf64884d4b4d6e60c3ef5d857c3f7469f",
    ),
    (
        4_702_158_080,
        "304d41ce03d59ac94bceb055935bf4e034df0badf8b0df4ded327c08a288a4cc",
    ),
    (
        4_933_368_232,
        "12a134b7c76d86edbe8fa2dc315f6cdaf4e1aca1b6ea4dfe4cad92df03d42eeb",
    ),
    (
        4_567_069_608,
        "b96395261359937c00fb42f4eb29306dc59b1a3368eeba52af4fb66e3e142c69",
    ),
    (
        4_702_158_080,
        "1897a6bf3b4fc834bb82d73ca02a7afc7d38c07f50ec5382cd54cd2f91b604d1",
    ),
    (
        4_933_368_232,
        "edfb38235adc96b99f55a401849befce59075a745e99c2d8c63ff358dd36443d",
    ),
    (
        4_567_069_608,
        "f8710775cf3413670edd7e23861b650a3431a71a6cc14cb1080623ab6b052385",
    ),
    (
        4_702_158_080,
        "9e18acc09f84edb5b34df9628efa15cfcab8bb76e8e20c1c2e979a107a0f7215",
    ),
    (
        4_933_368_232,
        "ea2e18228f8bdba1a4e0f32b155e4586df055997c45356213d05b971ba13e2f4",
    ),
    (
        4_567_069_608,
        "1e12083b1875678f7414ff55b09cd8bb1c30b861243f9bb7ff1e75b6ad3f1bdc",
    ),
    (
        4_644_161_920,
        "b340f44b5690cc745d48ae399381ec15b26a4fe25d483f677ccb4960dadb50d4",
    ),
];

const TEXT_ENCODER: &[(u64, &str)] = &[
    (
        4_932_328_944,
        "6b9dfbc930e505402ae9d7e5091a9d7d656cda5f34614f01cfe70bfb0cca27cb",
    ),
    (
        4_875_990_528,
        "d8bb44b4ff303fe76fe9e894022fb3dc71b15a2e716592790fe0e3c3e60478fa",
    ),
    (
        4_875_990_552,
        "54f22e8b3168f8dc962fac0d313607ebf52a12b433d4cf3098a0d82d9f042940",
    ),
    (
        4_875_990_584,
        "ad09c74d3c13ee29b5d0d84548fd8a3424a651564eaccd519946c296e59c557f",
    ),
    (
        4_875_990_584,
        "fc993c8a0e2a5b0570f383e1a95dc3a1281d1b224b6f3ee908f4827941e1dfc2",
    ),
    (
        4_875_990_584,
        "82f05620d1f718a90c362b221d6a184ff1a0f53301d706882d3df49695fa1974",
    ),
    (
        4_875_990_584,
        "fb91da8cb01ff4de3eef0eab1c3e769a734b3a1aafc61734068638a0d6c86934",
    ),
    (
        4_875_990_584,
        "431ca56535c8781944ce3801f5eb61c45531e853ecc5846d936ebaf4761b764f",
    ),
    (
        4_875_990_584,
        "3825e3f4302f4d2f7d76aa7430d2ce0864fde6b9e540a5806bf0d8e38e4d9f47",
    ),
    (
        4_875_990_584,
        "aded5a4d1d5e22dbd8b6f79266b6eb88c840411b09527c53917a1419ace22e2f",
    ),
    (
        4_875_990_584,
        "3820ffe8d8d6477f6fe8d614ef3c87abb264ee39accebf43a1507b970d80946f",
    ),
    (
        4_875_990_584,
        "05ad2d08ce71963121c9b03f1d9ec5d7641052f4b23c6c12b80d71065eb8e98e",
    ),
    (
        4_875_990_584,
        "b64f2289871261fdd1abbd3b78bcd66011b341de3dc8eeb2ed1a473ee7c8d95c",
    ),
    (
        3_270_697_008,
        "e45b6c9998c77ee5a6577f9f47bc76416c1d4d387169e50c4c9d3134ea51b13b",
    ),
];

fn official_shared_files() -> Vec<ModelFile> {
    let mut files: Vec<_> = TEXT_ENCODER
        .iter()
        .enumerate()
        .map(|(index, (size, sha))| {
            file(
                OFFICIAL_REPO,
                &format!("text_encoder/model-{:05}-of-00014.safetensors", index + 1),
                ModelComponent::TextEncoder,
                *size,
                sha,
            )
        })
        .collect();
    files.extend([
        file(
            OFFICIAL_REPO,
            "vae/diffusion_pytorch_model-00001-of-00003.safetensors",
            ModelComponent::Vae,
            5_061_033_024,
            "72f4c6be84ac0674f27398cde991dd9d719762f3952c4921aa66b2ce542f6374",
        ),
        file(
            OFFICIAL_REPO,
            "vae/diffusion_pytorch_model-00002-of-00003.safetensors",
            ModelComponent::Vae,
            4_955_986_528,
            "2e05e8bc23fa4071043e17fd242be8acd0685e781a43987432b2eae925be4198",
        ),
        file(
            OFFICIAL_REPO,
            "vae/diffusion_pytorch_model-00003-of-00003.safetensors",
            ModelComponent::Vae,
            398_539_336,
            "c05d6ac4b1a33de372799d708531da6320f6a3ce6d1ce6d895e770988e004a39",
        ),
        file(
            OFFICIAL_REPO,
            "audio_vae/diffusion_pytorch_model.safetensors",
            ModelComponent::AudioVae,
            605_429_340,
            "52c59e67ba8de5477c81bfbced0327aabf500f1bfdeefd5ee754529241cb26cb",
        ),
        file(
            OFFICIAL_REPO,
            "model_index.json",
            ModelComponent::ModelConfig,
            2_936,
            "5a587fe13b2371427415ac892463142683aefcd8d322e274a3a095eac37ac7d2",
        ),
        file(
            OFFICIAL_REPO,
            "modular_model_index.json",
            ModelComponent::ModelConfig,
            2_935,
            "a2b6a210e482ffb78e613b553f570c44e101afce6741bd4ed91429d0559af031",
        ),
        file(
            OFFICIAL_REPO,
            "text_encoder/config.json",
            ModelComponent::ModelConfig,
            1_474,
            "d2dd0c60d01b9e195d9447c52da61c7302d28828524914c044d9c6e1b81d0427",
        ),
        file(
            OFFICIAL_REPO,
            "text_encoder/model.safetensors.index.json",
            ModelComponent::ModelConfig,
            97_831,
            "06c952c569285870b811989b794b9766493e280fb77fbcb957fc4e5fcf25403a",
        ),
        file(
            OFFICIAL_REPO,
            "vae/config.json",
            ModelComponent::ModelConfig,
            2_011,
            "78f67deec3d63aae807f2bfe7154bc1e26f6372cb20b63265fcbae1b62bb5745",
        ),
        file(
            OFFICIAL_REPO,
            "vae/diffusion_pytorch_model.safetensors.index.json",
            ModelComponent::ModelConfig,
            74_228,
            "15f6d44553c3c616b0dc999920aa784f92ecee7e4201f1f99ac405cfbf3061ca",
        ),
        file(
            OFFICIAL_REPO,
            "audio_vae/config.json",
            ModelComponent::ModelConfig,
            2_271,
            "9a3c645ff892b376c6f5f4c8685964cd75474731af594ff058492a0000caabb6",
        ),
        file(
            OFFICIAL_REPO,
            "scheduler/scheduler_config.json",
            ModelComponent::VideoScheduler,
            97,
            "8fa6c3aa70dc9e691e1a6df899fd1b6f75f70481a27cee6e18a303817075c304",
        ),
        file(
            OFFICIAL_REPO,
            "audio_scheduler/scheduler_config.json",
            ModelComponent::AudioScheduler,
            96,
            "804780f7133477067bd6bbfbc02dc8b3cf9feeb400f97c08f5b1d5f6cbab3840",
        ),
        file(
            OFFICIAL_REPO,
            "processor/chat_template.json",
            ModelComponent::Processor,
            5_499,
            "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce",
        ),
        file(
            OFFICIAL_REPO,
            "processor/merges.txt",
            ModelComponent::Processor,
            1_671_839,
            "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        ),
        file(
            OFFICIAL_REPO,
            "processor/preprocessor_config.json",
            ModelComponent::Processor,
            390,
            "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        ),
        file(
            OFFICIAL_REPO,
            "processor/tokenizer.json",
            ModelComponent::Processor,
            7_032_403,
            "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7",
        ),
        file(
            OFFICIAL_REPO,
            "processor/tokenizer_config.json",
            ModelComponent::Processor,
            11_003,
            "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68",
        ),
        file(
            OFFICIAL_REPO,
            "processor/video_preprocessor_config.json",
            ModelComponent::Processor,
            385,
            "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
        ),
        file(
            OFFICIAL_REPO,
            "processor/vocab.json",
            ModelComponent::Processor,
            2_776_833,
            "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        ),
    ]);
    files
}

fn official_files(task: Task) -> Vec<ModelFile> {
    let (directory, shards) = match task {
        Task::Fl2va => ("transformer", FL2VA_TRANSFORMER),
        Task::Ref2va => ("transformer_ref", REF2VA_TRANSFORMER),
    };
    let mut files: Vec<_> = shards
        .iter()
        .enumerate()
        .map(|(index, (size, sha))| {
            file(
                OFFICIAL_REPO,
                &format!(
                    "{directory}/diffusion_pytorch_model-{:05}-of-00014.safetensors",
                    index + 1
                ),
                ModelComponent::TransformerShard,
                *size,
                sha,
            )
        })
        .collect();
    files.push(official_task_config(task));
    files.push(file(
        OFFICIAL_REPO,
        &format!("{directory}/diffusion_pytorch_model.safetensors.index.json"),
        ModelComponent::TaskConfig,
        64_488,
        "ac30a3b58963f2e735d493475fbb81853a5735ec947619648b3e045acda6783e",
    ));
    files.extend(official_shared_files());
    files
}

/// The task architecture config is valid for both the official and Comfy
/// weight layouts. It is deliberately separate from each task's sharded
/// weight index: the latter describes only the official BF16 files and must
/// never be attached to a Comfy single-file transformer.
fn official_task_config(task: Task) -> ModelFile {
    let directory = match task {
        Task::Fl2va => "transformer",
        Task::Ref2va => "transformer_ref",
    };
    file(
        OFFICIAL_REPO,
        &format!("{directory}/config.json"),
        ModelComponent::TaskConfig,
        546,
        "74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae",
    )
}

/// Pinned non-weight runtime assets reused by the Comfy layouts.
///
/// ComfyUI supplies equivalent tokenizer code/data from its Python package,
/// but Mold is a standalone binary and cannot assume that package exists.
/// Reuse the official model's exact processor and architecture/scheduler
/// files. Root model indexes and sharded-weight indexes are excluded because
/// they describe the official BF16 weight graph, not the Comfy layout.
fn official_runtime_support_files() -> Vec<ModelFile> {
    official_shared_files()
        .into_iter()
        .filter(|file| match file.component {
            ModelComponent::Processor
            | ModelComponent::VideoScheduler
            | ModelComponent::AudioScheduler => true,
            ModelComponent::ModelConfig => matches!(
                file.hf_filename.as_str(),
                "text_encoder/config.json" | "vae/config.json" | "audio_vae/config.json"
            ),
            _ => false,
        })
        .collect()
}

/// The compact shared graph: the NVFP4-AWQ conditioner, both VAEs, the
/// task architecture config, and the runtime support files.
///
/// Every compact layout names byte-identical entries here, which is what
/// makes them share one on-disk copy under `shared/minimax-h3/` and lets
/// removal ref-counting protect those bytes in every direction. Keep this a
/// single authority rather than duplicating the digests per layout: a
/// divergent copy would silently install a second 21.5 GB graph.
fn compact_shared_files(task: Task) -> Vec<ModelFile> {
    let mut files = vec![
        file(
            COMFY_REPO,
            "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            ModelComponent::TextEncoder,
            15_687_142_551,
            "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
        ),
        file(
            COMFY_REPO,
            "vae/minimax_h3_video_vae_fp16.safetensors",
            ModelComponent::Vae,
            5_207_808_496,
            "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
        ),
        file(
            COMFY_REPO,
            "vae/minimax_h3_audio_vae_fp32.safetensors",
            ModelComponent::AudioVae,
            605_254_808,
            "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
        ),
    ];
    files.push(official_task_config(task));
    files.extend(official_runtime_support_files());
    files
}

fn comfy_files(task: Task) -> Vec<ModelFile> {
    let (filename, transformer_sha) = match task {
        Task::Fl2va => (
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
        ),
        Task::Ref2va => (
            "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779",
        ),
    };
    let mut files = vec![file(
        COMFY_REPO,
        filename,
        ModelComponent::Transformer,
        20_970_379_616,
        transformer_sha,
    )];
    files.extend(compact_shared_files(task));
    files
}

/// The pruned NVFP4 compact stack: a third-party transformer on the exact
/// shared graph [`comfy_files`] names.
///
/// The sizes include the appended `L2P_bypass` marker documented on
/// [`NVFP4_REPO`]; the digests are over the published bytes, marker and all.
fn nvfp4_files(task: Task) -> Vec<ModelFile> {
    let (filename, size_bytes, transformer_sha) = match task {
        Task::Fl2va => (
            "MiniMax_H3_FL2VA_pruned_nvfp4.safetensors",
            12_528_636_865,
            "6ab7f0c48141e7919b32f925ca3def22e06a6aebeb9e0b6f5a0be0fe8409976f",
        ),
        Task::Ref2va => (
            "MiniMax_H3_Ref2VA_pruned_nvfp4.safetensors",
            12_528_636_866,
            "3e1be702c95bc057c05a7d1867e8aeea33073dcf5743835f2f27f06a2f34c596",
        ),
    };
    let mut files = vec![file(
        NVFP4_REPO,
        filename,
        ModelComponent::Transformer,
        size_bytes,
        transformer_sha,
    )];
    files.extend(compact_shared_files(task));
    files
}

fn defaults(layout: Layout) -> ManifestDefaults {
    defaults_with_steps(match layout {
        Layout::OfficialBf16 => DEFAULT_STEPS,
        Layout::ComfyPrunedInt8ConvrotNvfp4Awq | Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq => {
            COMFY_DEFAULT_STEPS
        }
    })
}

fn defaults_with_steps(steps: u32) -> ManifestDefaults {
    ManifestDefaults {
        steps,
        guidance: 0.0,
        width: DEFAULT_WIDTH,
        height: DEFAULT_HEIGHT,
        is_schnell: false,
        scheduler: None,
        negative_prompt: None,
        // The shipped default clip length, deliberately the reviewed compact
        // count rather than the family floor: widening the floor to the model
        // card's 4-second minimum must not move what a request renders by
        // default.
        frames: Some(REVIEWED_COMPACT_FRAMES),
        fps: Some(FIXED_FPS),
        // T2VA runs unconditioned; FL2VA's boundary frames ride the
        // dedicated first/last contract, not the generic source well.
        source_image: None,
    }
}

/// Pinned manifests used for exact identity, storage, and runtime work.
/// Every pinned upstream layout is visible for direct download even when the
/// current host cannot execute it. Download authority and runtime authority
/// remain deliberately independent.
pub(crate) fn manifests() -> Vec<ModelManifest> {
    let mut manifests: Vec<ModelManifest> = [
        (FL2VA_OFFICIAL, Task::Fl2va, Layout::OfficialBf16),
        (REF2VA_OFFICIAL, Task::Ref2va, Layout::OfficialBf16),
        (
            FL2VA_COMFY,
            Task::Fl2va,
            Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
        ),
        (
            REF2VA_COMFY,
            Task::Ref2va,
            Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
        ),
        (
            FL2VA_COMFY_NVFP4,
            Task::Fl2va,
            Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq,
        ),
        (
            REF2VA_COMFY_NVFP4,
            Task::Ref2va,
            Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq,
        ),
    ]
    .into_iter()
    .map(|(name, task, layout)| ModelManifest {
        name: name.to_string(),
        family: FAMILY.to_string(),
        description: match (task, layout) {
            (Task::Fl2va, Layout::OfficialBf16) => "MiniMax H3 FL2VA official BF16 transformer/conditioner + FP32 VAEs (downloadable qualification reference; execution unavailable)",
            (Task::Ref2va, Layout::OfficialBf16) => "MiniMax H3 Ref2VA official BF16 transformer/conditioner + FP32 VAEs (downloadable qualification reference; execution unavailable)",
            (Task::Fl2va, Layout::ComfyPrunedInt8ConvrotNvfp4Awq) => "MiniMax H3 FL2VA Comfy pruned INT8-convrot + NVFP4-AWQ (downloadable; CUDA or Apple Metal)",
            (Task::Ref2va, Layout::ComfyPrunedInt8ConvrotNvfp4Awq) => "MiniMax H3 Ref2VA Comfy pruned INT8-convrot + NVFP4-AWQ (downloadable; execution requires a qualified CUDA host)",
            (Task::Fl2va, Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq) => "MiniMax H3 FL2VA pruned NVFP4 transformer + NVFP4-AWQ conditioner (downloadable; execution not implemented in this build)",
            (Task::Ref2va, Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq) => "MiniMax H3 Ref2VA pruned NVFP4 transformer + NVFP4-AWQ conditioner (downloadable; execution not implemented in this build)",
        }
        .to_string(),
        files: match layout {
            Layout::OfficialBf16 => official_files(task),
            Layout::ComfyPrunedInt8ConvrotNvfp4Awq => comfy_files(task),
            Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq => nvfp4_files(task),
        },
        defaults: defaults(layout),
        hidden: false,
    })
    .collect();
    // A reviewed Turbo tier is its own task's compact stack plus one pinned
    // LoRA adapter; the tag selects the tier's reviewed step count and the
    // adapter artifact. The task comes from the tag's own identity, so a
    // `ref2v` adapter never lands beside the FL2VA transformer.
    manifests.extend(REVIEWED_TURBO_MANIFEST_TIERS.iter().map(|tier| {
        let task = task_for_model(tier.model).expect("reviewed turbo tag resolves to a task");
        let mut files = comfy_files(task);
        files.push(file(
            tier.adapter_hf_repo,
            tier.adapter_hf_filename,
            ModelComponent::DistilledLora,
            tier.adapter_size_bytes,
            tier.adapter_sha256,
        ));
        ModelManifest {
            name: tier.model.to_string(),
            family: FAMILY.to_string(),
            // This one sentence is the whole `/api/models` row, the
            // Discover card, and the `mold list` line a user reads before a
            // ~42.8 GB pull, so it discloses the adapter's SHAPE and not
            // only its label: a lossy SVD resize is never described in the
            // same words as the full-rank adapter it approximates.
            description: format!(
                "MiniMax H3 {} Comfy pruned INT8-convrot + NVFP4-AWQ with the reviewed{} {} LoRA (downloadable; CUDA or Apple Metal)",
                match task {
                    Task::Fl2va => "FL2VA",
                    Task::Ref2va => "Ref2VA",
                },
                if tier.adapter_shape_label == RESIZED_TURBO_ADAPTER_SHAPE {
                    ", lossy SVD-resized"
                } else {
                    ""
                },
                tier.display_label
            ),
            files,
            defaults: defaults_with_steps(tier.steps),
            hidden: false,
        }
    }));
    manifests
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{find_manifest, storage_path};

    fn inline(bytes: &[u8]) -> GenerationReferenceAuthority {
        GenerationReferenceAuthority::Inline {
            data: bytes.to_vec(),
        }
    }

    fn image_reference(label: &str, byte: u8) -> GenerationReference {
        GenerationReference::Image {
            media: inline(&[byte; 8]),
            provenance: crate::GenerationReferenceProvenance {
                name: Some(label.to_string()),
                sha256: None,
                crop: None,
            },
            mime_type: "image/png".to_string(),
            width: 1920,
            height: 1080,
        }
    }

    fn video_reference(label: &str, byte: u8, duration_ms: u64) -> GenerationReference {
        let fps = 29.97;
        GenerationReference::Video {
            media: inline(&[byte; 12]),
            provenance: crate::GenerationReferenceProvenance {
                name: Some(label.to_string()),
                sha256: None,
                crop: None,
            },
            mime_type: "video/mp4".to_string(),
            width: 1920,
            height: 1080,
            frame_count: Some((duration_ms as f64 * fps / 1_000.0).round() as u32),
            duration_ms,
            fps,
            has_audio: true,
            audio_duration_ms: Some(duration_ms),
            audio_sample_count: Some(duration_ms.saturating_mul(48)),
            audio_sample_rate: Some(48_000),
            audio_channels: Some(2),
        }
    }

    fn audio_reference(label: &str, byte: u8, duration_ms: u64) -> GenerationReference {
        GenerationReference::Audio {
            media: inline(&[byte; 10]),
            provenance: crate::GenerationReferenceProvenance {
                name: Some(label.to_string()),
                sha256: None,
                crop: None,
            },
            mime_type: "audio/wav".to_string(),
            duration_ms,
            sample_rate: 48_000,
            channels: 1,
            sample_count: Some(duration_ms.saturating_mul(48)),
        }
    }

    /// The scheduler resolves the payload-free durable row, so a FL2VA job
    /// whose first frame sits in the queue-media store reads as
    /// `source_image: None`; the resolver's presence projection is what keeps
    /// it a first-frame render. A hydrated request needs no projection, and
    /// a projection never invents a frame the request itself contradicts.
    #[test]
    fn resolved_media_presence_restores_the_scrubbed_first_frame_mode() {
        let mut hydrated = request();
        hydrated.source_image = Some(vec![0_u8; 16]);
        assert_eq!(
            validate_resolved_request_contract(&hydrated, Task::Fl2va).unwrap(),
            Mode::FirstFrameToAudioVideo
        );
        let mut scrubbed = hydrated.clone();
        crate::request_media::scrub_request_media(&mut scrubbed);
        assert!(scrubbed.source_image.is_none());
        assert_eq!(
            validate_resolved_request_contract(&scrubbed, Task::Fl2va).unwrap(),
            Mode::TextToAudioVideo
        );
        assert_eq!(
            validate_resolved_request_contract_with_media(
                &scrubbed,
                Task::Fl2va,
                ResolvedMediaPresence { source_image: true }
            )
            .unwrap(),
            Mode::FirstFrameToAudioVideo
        );
        assert_eq!(
            validate_resolved_request_contract_with_media(
                &hydrated,
                Task::Fl2va,
                ResolvedMediaPresence {
                    source_image: false
                }
            )
            .unwrap(),
            Mode::FirstFrameToAudioVideo
        );
        assert_eq!(
            ResolvedMediaPresence::from_request(&hydrated),
            ResolvedMediaPresence { source_image: true }
        );
    }

    fn request() -> GenerateRequest {
        GenerateRequest {
            mesh: None,
            video_only: None,
            collection: None,
            tags: None,
            title: None,
            source_fit: None,
            prompt: "a lighthouse in a storm".into(),
            negative_prompt: None,
            model: FL2VA_COMFY.into(),
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            steps: DEFAULT_STEPS,
            guidance: 0.0,
            seed: Some(42),
            batch_size: 1,
            output_format: Some(OutputFormat::Mp4),
            embed_metadata: None,
            scheduler: None,
            cfg_plus: None,
            source_image: None,
            source_image_name: None,
            edit_images: None,
            references: None,
            strength: 1.0,
            mask_image: None,
            control_image: None,
            control_model: None,
            control_scale: 1.0,
            expand: None,
            original_prompt: None,
            prompt_transform: None,
            batch_id: None,
            batch_index: None,
            batch_count: None,
            lora: None,
            frames: Some(REVIEWED_COMPACT_FRAMES),
            fps: Some(FIXED_FPS),
            upscale_model: None,
            gif_preview: false,
            enable_audio: None,
            audio_file: None,
            audio_file_path: None,
            source_video: None,
            source_video_path: None,
            extend_video: None,
            extend_video_path: None,
            extend_overlap_frames: None,
            keyframes: None,
            hdr_exr_dir: None,
            hdr_exr_full_float: false,
            pipeline: None,
            ic_lora_control: None,
            loras: None,
            retake_range: None,
            spatial_upscale: None,
            temporal_upscale: None,
            guidance_overrides: None,
            sample_shift: None,
            distill_strength_high: None,
            distill_strength_low: None,
            placement: None,
            id_image: None,
            id_image_name: None,
            id_weight: None,
            id_start_step: None,
            id_images: None,
            id_image_names: None,
            true_cfg: None,
            cfg_start_step: None,
        }
    }

    /// A source image never bends a compact request off the compact canvas
    /// rule: the official short-edge/area resolver is BF16-reference behavior
    /// only, and a compact fit renders the source's own aspect as large as the
    /// compact area ceiling allows.
    #[test]
    fn source_fit_keeps_the_compact_envelope_and_the_official_aspect_canvas() {
        for model in [
            FL2VA_COMFY,
            REF2VA_COMFY,
            FL2VA_COMFY_TURBO_8STEP,
            FL2VA_COMFY_TURBO_4STEP_768P,
            REF2VA_COMFY_TURBO_4STEP,
            FL2VA_COMFY_TURBO_4STEP_768P_V11,
            FL2VA_COMFY_TURBO_8STEP_768P,
            FL2VA_COMFY_TURBO_4STEP_768P_R21,
            FL2VA_COMFY_TURBO_8STEP_R21,
            REF2VA_COMFY_TURBO_4STEP_R21,
        ] {
            assert_eq!(
                source_fit_dimensions(model, 1024, 1024),
                (992, 992),
                "{model}"
            );
            assert_eq!(
                source_fit_dimensions(model, 1920, 1080),
                (1312, 736),
                "{model}"
            );
        }
        for (width, height) in [(1024, 1024), (1920, 1080), (1080, 1920)] {
            assert_eq!(
                source_fit_dimensions(FL2VA_OFFICIAL, width, height),
                recommended_dimensions(width, height)
            );
        }
        // Unresolvable family-configured models fail toward the stricter
        // compact contract.
        assert_eq!(
            source_fit_dimensions("custom-h3-finetune", 1920, 1080),
            (1312, 736)
        );
    }

    #[test]
    fn aliases_resolve_to_explicit_practical_variants() {
        assert_eq!(resolve_model_name("minimax-h3"), Some(FL2VA_COMFY));
        assert_eq!(resolve_model_name("minimax_h3_ref2va"), Some(REF2VA_COMFY));
        assert_eq!(canonical_family("MiniMax_H3"), Some(FAMILY));
        assert_eq!(canonical_family("h3"), None);
        for alias in FAMILY_ALIASES {
            assert_eq!(
                crate::ExpandTask::for_family(alias),
                crate::ExpandTask::TextToVideo
            );
        }
        for lookalike in [
            "notminimax-h3",
            "minimax-h30",
            "minimax-h3-ref2va-extra",
            "other:minimax-h3-fl2va:official-bf16",
        ] {
            assert_eq!(resolve_model_name(lookalike), None, "{lookalike}");
            assert_eq!(task_for_model(lookalike), None, "{lookalike}");
            assert_eq!(layout_for_model(lookalike), None, "{lookalike}");
        }
    }

    #[test]
    fn request_model_canonicalization_preserves_exact_partition_identity() {
        let mut ref2va = request();
        ref2va.model = "MiniMax_H3_Ref2VA".into();
        assert!(canonicalize_request_model(&mut ref2va));
        assert_eq!(ref2va.model, REF2VA_COMFY);
        assert!(!canonicalize_request_model(&mut ref2va));

        let mut official = request();
        official.model = " MINIMAX_H3_FL2VA:OFFICIAL_BF16 ".into();
        assert!(canonicalize_request_model(&mut official));
        assert_eq!(official.model, FL2VA_OFFICIAL);

        let mut opaque = request();
        opaque.model = "hf:example/custom-checkpoint".into();
        assert!(!canonicalize_request_model(&mut opaque));
        assert_eq!(opaque.model, "hf:example/custom-checkpoint");
    }

    #[test]
    fn timing_grid_accepts_the_three_documented_nominal_durations() {
        for frames in [124, 243, 345] {
            assert!(valid_frame_count(frames), "{frames}");
        }
        for frames in [123, 125, 361, 363] {
            assert!(!valid_frame_count(frames), "{frames}");
        }
        assert_eq!(recommended_frames(125), 124);
        assert_eq!(recommended_frames(350), 345);
        assert_eq!(recommended_frames(u32::MAX), MAX_FRAMES);
    }

    #[test]
    fn fl2va_modes_are_derived_only_from_boundary_frames() {
        let mut req = request();
        assert_eq!(
            validate_request_contract(&req, Task::Fl2va).unwrap(),
            Mode::TextToAudioVideo
        );

        req.source_image = Some(vec![1]);
        assert_eq!(
            validate_request_contract(&req, Task::Fl2va).unwrap(),
            Mode::FirstFrameToAudioVideo
        );
        req.keyframes = Some(vec![crate::KeyframeCondition {
            frame: REVIEWED_COMPACT_FRAMES - 1,
            image: vec![2],
            name: None,
        }]);
        assert_eq!(
            validate_request_contract(&req, Task::Fl2va).unwrap(),
            Mode::FirstAndLastFrameToAudioVideo
        );

        req.source_image = None;
        assert_eq!(
            validate_request_contract(&req, Task::Fl2va).unwrap(),
            Mode::LastFrameToAudioVideo
        );
        req.keyframes.as_mut().unwrap()[0].frame = 17;
        assert_eq!(
            validate_request_contract(&req, Task::Fl2va)
                .unwrap_err()
                .code,
            "MINIMAX_H3_BOUNDARY_FRAME_REQUIRED"
        );
    }

    #[test]
    fn ref2va_and_synchronized_audio_fail_closed() {
        let mut req = request();
        assert_eq!(
            validate_request_contract(&req, Task::Ref2va)
                .unwrap_err()
                .code,
            "MINIMAX_H3_REFERENCE_REQUIRED"
        );
        req.references = Some(vec![image_reference("anchor.png", 1)]);
        assert_eq!(
            validate_request_contract(&req, Task::Ref2va).unwrap(),
            Mode::ReferenceToAudioVideo
        );
        let mut resolved = req.clone();
        if let GenerationReference::Image {
            media, provenance, ..
        } = &mut resolved.references.as_mut().unwrap()[0]
        {
            *media = GenerationReferenceAuthority::Descriptor;
            provenance.sha256 = Some("A".repeat(64));
        }
        assert_eq!(
            validate_request_contract(&resolved, Task::Ref2va)
                .unwrap_err()
                .code,
            "MINIMAX_H3_REFERENCE_DESCRIPTOR_ONLY"
        );
        assert_eq!(
            validate_resolved_request_contract(&resolved, Task::Ref2va).unwrap(),
            Mode::ReferenceToAudioVideo
        );
        assert_eq!(
            validate_resolved_request_contract(&req, Task::Ref2va)
                .unwrap_err()
                .code,
            "MINIMAX_H3_REFERENCE_PREVIEW_MEDIA"
        );
        req.enable_audio = Some(false);
        assert_eq!(
            validate_request_contract(&req, Task::Ref2va)
                .unwrap_err()
                .code,
            "MINIMAX_H3_SYNCHRONIZED_AUDIO_REQUIRED"
        );
    }

    #[test]
    fn generic_sampler_overrides_and_degenerate_grids_fail_closed() {
        let mut req = request();
        req.scheduler = Some(crate::Scheduler::EulerAncestral);
        assert_eq!(
            validate_request_contract(&req, Task::Fl2va)
                .unwrap_err()
                .code,
            "MINIMAX_H3_FIXED_DUAL_SCHEDULE"
        );

        req.scheduler = None;
        req.steps = 1;
        assert_eq!(
            validate_request_contract(&req, Task::Fl2va)
                .unwrap_err()
                .code,
            "MINIMAX_H3_GRID_POINTS"
        );
    }

    #[test]
    fn foreign_pipeline_fields_fail_before_fl2va_planning() {
        let mut req = request();
        req.mask_image = Some(vec![1, 2, 3]);
        assert_eq!(
            validate_request_contract(&req, Task::Fl2va)
                .unwrap_err()
                .code,
            "MINIMAX_H3_FOREIGN_PIPELINE_FIELD"
        );

        req.mask_image = None;
        req.source_image_name = Some("orphan.png".into());
        assert_eq!(
            validate_request_contract(&req, Task::Fl2va)
                .unwrap_err()
                .code,
            "MINIMAX_H3_ORPHAN_SOURCE_NAME"
        );

        req.source_image_name = None;
        req.strength = 0.75;
        assert_eq!(
            validate_request_contract(&req, Task::Fl2va)
                .unwrap_err()
                .code,
            "MINIMAX_H3_FIXED_STRENGTH"
        );

        req.strength = 1.0;
        req.upscale_model = Some("real-esrgan-x4plus:fp16".into());
        assert!(validate_request_contract(&req, Task::Fl2va).is_ok());

        req.control_scale = 0.5;
        assert_eq!(
            validate_request_contract(&req, Task::Fl2va)
                .unwrap_err()
                .code,
            "MINIMAX_H3_FOREIGN_PIPELINE_FIELD"
        );

        req.control_scale = 1.0;
        req.extend_overlap_frames = Some(17);
        assert_eq!(
            validate_request_contract(&req, Task::Fl2va)
                .unwrap_err()
                .code,
            "MINIMAX_H3_FOREIGN_PIPELINE_FIELD"
        );
    }

    #[test]
    fn mixed_reference_order_round_trips_and_metadata_stays_redacted() {
        let references = vec![
            image_reference("  first.png  ", 11),
            video_reference("middle.mp4", 22, 4_000),
            audio_reference("last.wav", 33, 3_000),
        ];
        validate_references(&references).unwrap();

        let wire = serde_json::to_value(&references).unwrap();
        assert_eq!(wire[0]["kind"], "image");
        assert_eq!(wire[1]["kind"], "video");
        assert_eq!(wire[2]["kind"], "audio");
        let parsed: Vec<GenerationReference> = serde_json::from_value(wire).unwrap();
        assert_eq!(
            parsed
                .iter()
                .map(GenerationReference::kind)
                .collect::<Vec<_>>(),
            vec![
                crate::GenerationReferenceKind::Image,
                crate::GenerationReferenceKind::Video,
                crate::GenerationReferenceKind::Audio,
            ]
        );

        let mut req = request();
        req.model = REF2VA_COMFY.to_string();
        req.references = Some(parsed);
        assert_eq!(
            crate::ExpandTask::for_generation(FAMILY, &req),
            crate::ExpandTask::ReferenceToAudioVideo
        );
        let metadata = crate::OutputMetadata::from_generate_request(&req, 7, None, "test");
        let references = metadata.references.unwrap();
        assert_eq!(references.len(), 3);
        assert_eq!(references[0].index, 1);
        assert_eq!(references[0].name.as_deref(), Some("first.png"));
        assert_eq!(references[1].index, 2);
        assert_eq!(references[1].name.as_deref(), Some("middle.mp4"));
        assert!(references[1].has_audio);
        assert_eq!(references[2].index, 3);
        assert_eq!(references[2].name.as_deref(), Some("last.wav"));

        let metadata_wire = serde_json::to_string(&references).unwrap();
        assert!(!metadata_wire.contains("media"));
        assert!(!metadata_wire.contains("authority"));
        assert!(!metadata_wire.contains("handle"));
        assert!(!metadata_wire.contains("path"));
        assert!(!metadata_wire.contains("CwsLCwsL"));
        assert!(references
            .iter()
            .all(|reference| reference.sha256.len() == 64));
    }

    #[test]
    fn prepared_reference_shapes_match_the_official_ref2va_policy() {
        // 1920x1080 sits inside the 2048 short edge, so it keeps its native
        // geometry on the 32 grid instead of being upscaled to 3648x2048
        // (ComfyUI `nodes_minimax_h3.py:298-303`: `min(1.0, 2048/short)`).
        let image = reference_prepared_shape(&image_reference("anchor.png", 1)).unwrap();
        assert_eq!(image.version, REFERENCE_PREPROCESS_VERSION);
        assert_eq!(
            (image.normalized_width, image.normalized_height),
            (Some(1920), Some(1088))
        );
        assert_eq!(image.visual_rows, 60 * 34);
        assert_eq!(image.audio_rows, 0);

        let video = reference_prepared_shape(&video_reference("clip.mp4", 2, 4_000)).unwrap();
        assert_eq!(
            (video.normalized_width, video.normalized_height),
            (Some(1344), Some(768))
        );
        assert_eq!(video.normalized_video_frames, Some(96));
        assert_eq!(video.video_frames, Some(90));
        assert_eq!(video.qwen_video_frames, Some(8));
        assert_eq!(video.audio_samples_per_channel, Some(128_000));
        assert_eq!(video.visual_rows, 27 * 42 * 24);
        assert_eq!(video.audio_rows, 320);

        let audio = reference_prepared_shape(&audio_reference("voice.wav", 3, 3_000)).unwrap();
        assert_eq!(audio.audio_samples_per_channel, Some(96_000));
        assert_eq!(audio.audio_rows, 240);
        assert_eq!(audio.visual_rows, 0);
    }

    #[test]
    fn video_prepared_shape_uses_exact_cfr_frames_and_target_truncation() {
        let mut video = video_reference("long.mp4", 2, 15_000);
        if let GenerationReference::Video {
            frame_count, fps, ..
        } = &mut video
        {
            *frame_count = Some(450);
            *fps = 30.0;
        }
        // The reviewed clip length, which is what actually renders — this
        // exercises CFR truncation, not the family floor.
        let short = reference_prepared_shape_for_target(&video, REVIEWED_COMPACT_FRAMES).unwrap();
        assert_eq!(short.normalized_video_frames, Some(REVIEWED_COMPACT_FRAMES));
        assert_eq!(short.video_frames, Some(REVIEWED_COMPACT_FRAMES));
        assert_eq!(short.qwen_video_frames, Some(11));
        assert_eq!(short.audio_samples_per_channel, Some(165_334));
        assert_eq!(short.audio_rows, 414);

        let long = reference_prepared_shape_for_target(&video, MAX_FRAMES).unwrap();
        assert_eq!(long.normalized_video_frames, Some(MAX_FRAMES));
        assert_eq!(long.video_frames, Some(345));
        assert_eq!(long.qwen_video_frames, Some(29));
    }

    #[test]
    fn legacy_video_without_exact_frame_count_deserializes_then_fails_closed() {
        let reference: GenerationReference = serde_json::from_value(serde_json::json!({
            "kind": "video",
            "media": { "authority": "inline", "data": "AQID" },
            "mime_type": "video/mp4",
            "width": 640,
            "height": 480,
            "duration_ms": 2000,
            "fps": 24.0
        }))
        .unwrap();
        let error =
            reference_prepared_shape_for_target(&reference, REVIEWED_COMPACT_FRAMES).unwrap_err();
        assert_eq!(error.code, "MINIMAX_H3_REFERENCE_EXACT_VIDEO_SHAPE");
        assert_eq!(error.field, Some("frame_count"));
    }

    #[test]
    fn legacy_audio_without_exact_sample_count_deserializes_then_fails_closed() {
        let audio: GenerationReference = serde_json::from_value(serde_json::json!({
            "kind": "audio",
            "media": { "authority": "inline", "data": "AQID" },
            "mime_type": "audio/wav",
            "duration_ms": 2000,
            "sample_rate": 48000,
            "channels": 2
        }))
        .unwrap();
        let error =
            reference_prepared_shape_for_target(&audio, REVIEWED_COMPACT_FRAMES).unwrap_err();
        assert_eq!(error.code, "MINIMAX_H3_REFERENCE_EXACT_AUDIO_SHAPE");
        assert_eq!(error.field, Some("sample_count"));

        let mut video = video_reference("soundtrack.mp4", 4, 2_000);
        if let GenerationReference::Video {
            audio_sample_count, ..
        } = &mut video
        {
            *audio_sample_count = None;
        }
        let error =
            reference_prepared_shape_for_target(&video, REVIEWED_COMPACT_FRAMES).unwrap_err();
        assert_eq!(error.code, "MINIMAX_H3_REFERENCE_EXACT_AUDIO_SHAPE");
        assert_eq!(error.field, Some("sample_count"));
    }

    #[test]
    fn reference_geometry_never_upscales_small_images_and_videos() {
        let image = GenerationReference::Image {
            media: inline(&[1; 8]),
            provenance: crate::GenerationReferenceProvenance::default(),
            mime_type: "image/png".to_string(),
            width: 80,
            height: 48,
        };
        // ComfyUI's reference node never upscales an image
        // (`nodes_minimax_h3.py:267`, `:298-303`): a tiny source keeps its
        // native geometry, floored at the 32 grid.
        let image = reference_prepared_shape(&image).unwrap();
        assert_eq!(
            (image.normalized_width, image.normalized_height),
            (Some(64), Some(64))
        );

        let video = GenerationReference::Video {
            media: inline(&[2; 12]),
            provenance: crate::GenerationReferenceProvenance::default(),
            mime_type: "video/mp4".to_string(),
            width: 320,
            height: 240,
            frame_count: Some(48),
            duration_ms: 2_000,
            fps: 24.0,
            has_audio: false,
            audio_duration_ms: None,
            audio_sample_count: None,
            audio_sample_rate: None,
            audio_channels: None,
        };
        // A video smaller than its 768-short-edge canvas keeps its native
        // geometry on the 32 grid (`nodes_minimax_h3.py:318-323`).
        let video = reference_prepared_shape(&video).unwrap();
        assert_eq!(
            (video.normalized_width, video.normalized_height),
            (Some(320), Some(256))
        );
    }

    #[test]
    fn invalid_reference_cannot_silently_erase_ordered_metadata() {
        let mut invalid = image_reference("missing-digest.png", 9);
        if let GenerationReference::Image {
            media, provenance, ..
        } = &mut invalid
        {
            *media = GenerationReferenceAuthority::ServerPath {
                path: "/srv/mold-media/missing-digest.png".to_string(),
            };
            provenance.name = Some("/private/secret.png".to_string());
        }
        let mut req = request();
        req.references = Some(vec![image_reference("valid.png", 1), invalid]);
        let metadata = crate::OutputMetadata::from_generate_request(&req, 7, None, "test");
        let references = metadata.references.expect("reference order is retained");
        assert_eq!(references.len(), 2);
        assert_eq!(references[0].index, 1);
        assert_eq!(references[1].index, 2);
        assert!(references[1].sha256.is_empty());
        assert!(references[1].name.is_none());
        assert!(!serde_json::to_string(&references)
            .unwrap()
            .contains("/private/secret.png"));
    }

    #[test]
    fn reference_debug_redacts_bytes_handles_and_paths() {
        let cases = [
            image_reference("inline.png", 99),
            GenerationReference::Image {
                media: GenerationReferenceAuthority::Upload {
                    handle: "secret-upload-handle".to_string(),
                },
                provenance: crate::GenerationReferenceProvenance {
                    name: Some("upload.png".to_string()),
                    sha256: Some("a".repeat(64)),
                    crop: None,
                },
                mime_type: "image/png".to_string(),
                width: 1,
                height: 1,
            },
            GenerationReference::Image {
                media: GenerationReferenceAuthority::ServerPath {
                    path: "/private/reference.png".to_string(),
                },
                provenance: crate::GenerationReferenceProvenance {
                    name: Some("path.png".to_string()),
                    sha256: Some("b".repeat(64)),
                    crop: None,
                },
                mime_type: "image/png".to_string(),
                width: 1,
                height: 1,
            },
        ];
        let debug = format!("{cases:?}");
        assert!(debug.contains("<redacted"));
        assert!(!debug.contains("secret-upload-handle"));
        assert!(!debug.contains("/private/reference.png"));
        assert!(!debug.contains("99, 99"));
    }

    #[test]
    fn reference_limits_cover_kind_counts_durations_and_audio_only_sets() {
        let mut images = (0..=MAX_REFERENCE_IMAGES)
            .map(|index| image_reference(&format!("image-{index}.png"), index as u8))
            .collect::<Vec<_>>();
        let error = validate_references(&images).unwrap_err();
        assert_eq!(error.code, "MINIMAX_H3_REFERENCE_KIND_COUNT");
        images.pop();
        validate_references(&images).unwrap();

        let audio_only = vec![audio_reference("voice.wav", 1, 2_000)];
        assert_eq!(
            validate_references(&audio_only).unwrap_err().code,
            "MINIMAX_H3_REFERENCE_AUDIO_ONLY"
        );

        let too_short = vec![video_reference("short.mp4", 1, 1_999)];
        let error = validate_references(&too_short).unwrap_err();
        assert_eq!(error.reference, Some(1));
        assert_eq!(error.field, Some("duration_ms"));

        let too_much_video = vec![
            video_reference("one.mp4", 1, 8_000),
            video_reference("two.mp4", 2, 8_000),
        ];
        assert_eq!(
            validate_references(&too_much_video).unwrap_err().code,
            "MINIMAX_H3_REFERENCE_VIDEO_DURATION_TOTAL"
        );

        let too_much_audio = vec![
            image_reference("anchor.png", 1),
            audio_reference("one.wav", 2, 8_000),
            audio_reference("two.wav", 3, 8_000),
        ];
        assert_eq!(
            validate_references(&too_much_audio).unwrap_err().code,
            "MINIMAX_H3_REFERENCE_AUDIO_DURATION_TOTAL"
        );
    }

    #[test]
    fn reference_authorities_and_declared_identity_fail_closed() {
        let bytes = [7u8; 8];
        let mut bad_digest = image_reference("bad.png", 7);
        if let GenerationReference::Image { provenance, .. } = &mut bad_digest {
            provenance.sha256 = Some("0".repeat(64));
        }
        assert_eq!(
            validate_references(&[bad_digest]).unwrap_err().code,
            "MINIMAX_H3_REFERENCE_DIGEST_MISMATCH"
        );

        let path_without_digest = GenerationReference::Image {
            media: GenerationReferenceAuthority::ServerPath {
                path: "/srv/mold-media/anchor.png".to_string(),
            },
            provenance: crate::GenerationReferenceProvenance {
                name: Some("anchor.png".to_string()),
                sha256: None,
                crop: None,
            },
            mime_type: "image/png".to_string(),
            width: 100,
            height: 100,
        };
        assert_eq!(
            validate_references(&[path_without_digest])
                .unwrap_err()
                .code,
            "MINIMAX_H3_REFERENCE_DIGEST_REQUIRED"
        );

        let mut correct = image_reference("correct.png", 7);
        if let GenerationReference::Image { provenance, .. } = &mut correct {
            use sha2::{Digest, Sha256};
            provenance.sha256 = Some(format!("{:x}", Sha256::digest(bytes)));
        }
        validate_references(&[correct]).unwrap();

        let descriptor = GenerationReference::Image {
            media: GenerationReferenceAuthority::Descriptor,
            provenance: crate::GenerationReferenceProvenance {
                name: Some("preview.png".to_string()),
                sha256: Some("A".repeat(64)),
                crop: None,
            },
            mime_type: "image/png".to_string(),
            width: 100,
            height: 100,
        };
        assert_eq!(
            validate_references(std::slice::from_ref(&descriptor))
                .unwrap_err()
                .code,
            "MINIMAX_H3_REFERENCE_DESCRIPTOR_ONLY"
        );
        validate_reference_descriptors(&[descriptor]).unwrap();
        assert_eq!(
            validate_reference_descriptors(&[image_reference("raw.png", 1)])
                .unwrap_err()
                .code,
            "MINIMAX_H3_REFERENCE_PREVIEW_MEDIA"
        );
    }

    #[test]
    fn reference_reordering_changes_serialized_authority() {
        use sha2::{Digest, Sha256};
        let first = image_reference("first.png", 1);
        let second = image_reference("second.png", 2);
        let digest = |references: &[GenerationReference]| {
            format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(references).unwrap())
            )
        };
        assert_ne!(
            digest(&[first.clone(), second.clone()]),
            digest(&[second, first])
        );
    }

    #[test]
    fn invalid_timing_and_canvas_errors_include_recommendations() {
        let mut req = request();
        req.frames = Some(125);
        let error = validate_request_contract(&req, Task::Fl2va).unwrap_err();
        assert_eq!(error.code, "MINIMAX_H3_FRAME_GRID");
        assert_eq!(error.recommended_frames, Some(124));

        req.frames = Some(REVIEWED_COMPACT_FRAMES);
        req.width = 1056;
        req.height = 1056;
        let error = validate_request_contract(&req, Task::Fl2va).unwrap_err();
        assert_eq!(error.code, "MINIMAX_H3_DIMENSIONS");
        let (width, height) = error.recommended_dimensions.unwrap();
        assert!(width.is_multiple_of(DIMENSION_ALIGNMENT));
        assert!(height.is_multiple_of(DIMENSION_ALIGNMENT));
        assert!(u64::from(width) * u64::from(height) <= MAX_PIXELS);
        assert_eq!((width, height), (768, 768));
    }

    #[test]
    fn fixed_fps_frame_bounds_and_every_canvas_rule_fail_explicitly() {
        for fps in [1, 23, 25, 120] {
            let mut req = request();
            req.fps = Some(fps);
            assert_eq!(
                validate_request_contract(&req, Task::Fl2va)
                    .unwrap_err()
                    .code,
                "MINIMAX_H3_FIXED_FPS",
                "fps={fps}"
            );
        }

        // 90 is on the `17n+5` grid but only 3.75 s, below the model card's
        // 4-second floor; 125 is off the grid; 379 is past 15 seconds.
        for frames in [90, 125, 379] {
            let mut req = request();
            req.frames = Some(frames);
            assert_eq!(
                validate_request_contract(&req, Task::Fl2va)
                    .unwrap_err()
                    .code,
                "MINIMAX_H3_FRAME_GRID",
                "frames={frames}"
            );
        }

        for (width, height, rule) in [
            (0, DEFAULT_HEIGHT, "positive"),
            (DEFAULT_WIDTH - 1, DEFAULT_HEIGHT, "alignment"),
            (1056, 1056, "area"),
            (160, 768, "minimum aspect"),
            (1344, 256, "maximum aspect"),
        ] {
            let mut req = request();
            req.width = width;
            req.height = height;
            assert_eq!(
                validate_request_contract(&req, Task::Fl2va)
                    .unwrap_err()
                    .code,
                "MINIMAX_H3_DIMENSIONS",
                "{rule}: {width}x{height}"
            );
        }

        // The family envelope, on an identity that keeps it.
        for (width, height) in [
            (DEFAULT_WIDTH, DEFAULT_HEIGHT),
            (DEFAULT_HEIGHT, DEFAULT_WIDTH),
            (1024, 1024),
            (256, 1024),
            (1024, 256),
        ] {
            let mut req = request();
            req.model = FL2VA_OFFICIAL.into();
            req.steps = DEFAULT_STEPS;
            req.width = width;
            req.height = height;
            assert!(
                validate_request_contract(&req, Task::Fl2va).is_ok(),
                "valid canvas {width}x{height}"
            );
        }
    }

    /// The reviewed-canvas gate is a DOOR check, never part of the family
    /// contract: the engine's own `prepare_request` runs the contract on the
    /// tiny synthetic canvases its pipeline tests use, and a compact identity
    /// is a perfectly valid H3 model there.
    #[test]
    fn the_reviewed_canvas_gate_is_a_door_check_and_not_the_family_contract() {
        for &(width, height) in REVIEWED_COMPACT_CANVASES {
            let mut req = request();
            req.width = width;
            req.height = height;
            validate_reviewed_canvas(&req)
                .unwrap_or_else(|error| panic!("{width}x{height}: {}", error.message));
            assert!(validate_request_contract(&req, Task::Fl2va).is_ok());
        }

        // Family-legal but outside the compact rule: over the area
        // ceiling (the first two) and under the compact axis floor.
        for (width, height) in [(1056, 992), (576, 1856), (224, 896)] {
            let mut req = request();
            req.width = width;
            req.height = height;
            let error = validate_reviewed_canvas(&req).unwrap_err();
            assert_eq!(
                error.code, "MINIMAX_H3_DIMENSIONS",
                "off-envelope compact canvas {width}x{height}"
            );
            let repair = error.recommended_dimensions.unwrap();
            assert!(
                is_admitted_compact_canvas(repair.0, repair.1),
                "the repair must land on a canvas that runs: {width}x{height}"
            );
            // The family contract still admits it — the engine's synthetic
            // pipeline tests depend on exactly that.
            assert!(
                validate_request_contract(&req, Task::Fl2va).is_ok(),
                "{width}x{height} is inside the family envelope"
            );
        }

        // A hidden official reference has no reviewed canvas set at all.
        let mut official = request();
        official.model = FL2VA_OFFICIAL.into();
        official.steps = DEFAULT_STEPS;
        official.width = 1024;
        official.height = 768;
        validate_reviewed_canvas(&official).unwrap();
    }

    /// The frame bounds are DERIVED from the published duration contract, not
    /// transcribed beside it. `MiniMax-AI/MiniMax-H3` `README.md:73` states an
    /// output duration of 4-15 seconds; at the fixed 24 fps that is the first
    /// and last `17n+5` grid point inside `[4, 15]` seconds. `MIN_FRAMES` was
    /// 124 for exactly as long as `MIN_DURATION_SECONDS` said 5, which refused
    /// the 107-frame clip the model card permits.
    #[test]
    fn frame_bounds_are_derived_from_the_published_duration_contract() {
        let grid: Vec<u32> = (0..40)
            .map(|n| FRAME_OFFSET + n * FRAME_STEP)
            .take_while(|frames| *frames < 1_000)
            .collect();

        let lowest = grid
            .iter()
            .copied()
            .find(|frames| {
                f64::from(*frames) / f64::from(FIXED_FPS) >= f64::from(MIN_DURATION_SECONDS)
            })
            .expect("a grid point reaches the minimum duration");
        let highest = grid
            .iter()
            .copied()
            .rfind(|frames| {
                f64::from(*frames) / f64::from(FIXED_FPS) <= f64::from(MAX_DURATION_SECONDS)
            })
            .expect("a grid point sits inside the maximum duration");

        assert_eq!(MIN_FRAMES, lowest, "minimum frames");
        assert_eq!(MAX_FRAMES, highest, "maximum frames");
        assert_eq!(MIN_FRAMES, 107);
        assert_eq!(MAX_FRAMES, 345);
        // The grid point below the floor really is short of the contract, so
        // 107 is a floor rather than a rounding artefact.
        assert!(
            f64::from(MIN_FRAMES - FRAME_STEP) / f64::from(FIXED_FPS)
                < f64::from(MIN_DURATION_SECONDS)
        );
        assert!(valid_frame_count(MIN_FRAMES));
        assert!(valid_frame_count(MAX_FRAMES));
    }

    /// The reviewed compact runtime's clip length is its own authority. It sat
    /// on `MIN_FRAMES` while the two happened to be equal, so widening the
    /// family floor would silently have widened what the runtime admits.
    #[test]
    fn the_reviewed_compact_clip_length_is_independent_of_the_family_floor() {
        assert_eq!(REVIEWED_COMPACT_FRAMES, 124);
        assert!(valid_frame_count(REVIEWED_COMPACT_FRAMES));
        // Strictly greater: the reviewed clip is longer than the family
        // floor, so collapsing the two constants back into one fails here.
        assert_eq!(
            REVIEWED_COMPACT_FRAMES.cmp(&MIN_FRAMES),
            std::cmp::Ordering::Greater
        );
        assert_ne!(
            REVIEWED_COMPACT_FRAMES.cmp(&MAX_FRAMES),
            std::cmp::Ordering::Greater
        );
    }

    /// `is_admitted_compact_canvas` is the one canvas authority, and
    /// `REVIEWED_COMPACT_CANVASES` is a recommendation list that must satisfy
    /// it. The rule replaced set membership so a compact tag can render any
    /// aligned canvas inside the campaign's own area ceiling; the memory
    /// estimate, not a pinned list, decides what actually fits.
    #[test]
    fn the_compact_canvas_rule_is_the_one_canvas_authority() {
        // The default is first, which is what a client offering
        // `recommended_dimensions` shows first.
        assert_eq!(
            REVIEWED_COMPACT_CANVASES.first().copied(),
            Some((DEFAULT_WIDTH, DEFAULT_HEIGHT))
        );
        for &(width, height) in REVIEWED_COMPACT_CANVASES {
            assert!(
                is_admitted_compact_canvas(width, height),
                "recommended preset {width}x{height} must satisfy the rule"
            );
            // Every preset must be inside the family contract too.
            assert!(width.is_multiple_of(DIMENSION_ALIGNMENT));
            assert!(height.is_multiple_of(DIMENSION_ALIGNMENT));
            assert!(u64::from(width) * u64::from(height) <= MAX_PIXELS);
        }
        // Both campaign canvases are still offered.
        assert!(REVIEWED_COMPACT_CANVASES.contains(&(1344, 768)));
        assert!(REVIEWED_COMPACT_CANVASES.contains(&(768, 768)));

        // Admitted: aligned, inside the ceiling, sane aspect. 1024x576 and
        // the transpose of the default were both refused before the rule.
        for (width, height) in [
            (1024, 576),
            (768, 1344),
            (1024, 768),
            (992, 992),
            (256, 256),
            (2016, 512),
        ] {
            assert!(
                is_admitted_compact_canvas(width, height),
                "{width}x{height} should be admitted"
            );
        }
        // Over the area ceiling by one stride on one axis.
        assert!(!is_admitted_compact_canvas(1344, 800));
        // Off the 32 stride.
        assert!(!is_admitted_compact_canvas(1000, 600));
        assert!(!is_admitted_compact_canvas(1024, 570));
        // Under the short-axis floor.
        assert!(!is_admitted_compact_canvas(224, 224));
        // Outside the family aspect bounds (5:1).
        assert!(!is_admitted_compact_canvas(1600, 288));
        // Zero is never a canvas.
        assert!(!is_admitted_compact_canvas(0, 768));
        assert!(!is_admitted_compact_canvas(768, 0));

        assert_eq!(reviewed_compact_max_pixels(), 1344 * 768);
        assert_eq!(COMPACT_MAX_PIXELS, 1_032_192);
        // The rule's own widest canvas is 2016x512 — aspect exactly 4:1 and
        // area exactly on the ceiling — not the widest preset.
        assert_eq!(reviewed_compact_max_axis_pixels(), 2016);
        assert_eq!(reviewed_compact_min_axis_pixels(), MIN_COMPACT_AXIS_PIXELS);
        let (min_aspect, max_aspect) = reviewed_compact_aspect_bounds();
        assert!((min_aspect - MIN_ASPECT_RATIO).abs() < 1e-9);
        assert!((max_aspect - MAX_ASPECT_RATIO).abs() < 1e-9);
    }

    /// The rows-per-latent ceiling is a CONSEQUENCE of the area ceiling, and
    /// the conditioning row caps in `private_server.rs` depend on it: one
    /// packed row is a 32x32 cell, so no admitted canvas can exceed
    /// `COMPACT_MAX_PIXELS / 1024` rows, which is the default canvas's own
    /// 1,008. If this ever fails, the reviewed vision/condition ceilings stop
    /// being ceilings.
    #[test]
    fn no_admitted_canvas_packs_more_rows_per_latent_than_the_default() {
        let ceiling = COMPACT_MAX_PIXELS / u64::from(VIDEO_ROW_STRIDE * VIDEO_ROW_STRIDE);
        assert_eq!(ceiling, 1_008);
        assert_eq!(
            rows_per_video_latent(DEFAULT_WIDTH, DEFAULT_HEIGHT),
            Some(ceiling)
        );
        let mut width = MIN_COMPACT_AXIS_PIXELS;
        while width <= reviewed_compact_max_axis_pixels() {
            let mut height = MIN_COMPACT_AXIS_PIXELS;
            while height <= reviewed_compact_max_axis_pixels() {
                if is_admitted_compact_canvas(width, height) {
                    assert!(
                        rows_per_video_latent(width, height).unwrap() <= ceiling,
                        "{width}x{height}"
                    );
                }
                height += VIDEO_ROW_STRIDE;
            }
            width += VIDEO_ROW_STRIDE;
        }
    }

    /// `qwen_vision_rows` is the pre-merge patch count on the 16-px grid, and
    /// it is exactly four merged pads on every 32-aligned canvas. FL2VA's
    /// reviewed 4,032 vision rows and 1,008 pads at 1344x768 are these two
    /// functions, and a 2048-square image reference is 16,384 patches — the
    /// count the runtime prepares, which #1418's admission charged as 4,096.
    #[test]
    fn qwen_vision_patch_rows_are_four_per_merged_pad() {
        assert_eq!(
            qwen_vision_pad_rows_per_block(DEFAULT_WIDTH, DEFAULT_HEIGHT),
            1_008
        );
        assert_eq!(
            qwen_vision_patch_rows_per_block(DEFAULT_WIDTH, DEFAULT_HEIGHT),
            4_032
        );
        assert_eq!(qwen_vision_pad_rows_per_block(2_048, 2_048), 4_096);
        assert_eq!(qwen_vision_patch_rows_per_block(2_048, 2_048), 16_384);
        let per_pad = u64::from(QWEN_VISION_SPATIAL_MERGE * QWEN_VISION_SPATIAL_MERGE);
        assert_eq!(per_pad, 4);
        let mut width = VIDEO_ROW_STRIDE;
        while width <= 4_096 {
            let mut height = VIDEO_ROW_STRIDE;
            while height <= 4_096 {
                assert_eq!(
                    qwen_vision_patch_rows_per_block(width, height),
                    per_pad * qwen_vision_pad_rows_per_block(width, height),
                    "{width}x{height}"
                );
                height += VIDEO_ROW_STRIDE;
            }
            width += VIDEO_ROW_STRIDE;
        }
    }

    /// The packed-row arithmetic is one authority. These are the numbers the
    /// #827 campaign's envelope was transcribed from, so a change here is a
    /// change to what admission charges.
    #[test]
    fn packed_row_geometry_reproduces_the_measured_envelope() {
        assert_eq!(video_latent_frames(REVIEWED_COMPACT_FRAMES), Some(37));
        assert_eq!(
            rows_per_video_latent(DEFAULT_WIDTH, DEFAULT_HEIGHT),
            Some(1_008)
        );
        assert_eq!(
            target_video_rows(DEFAULT_WIDTH, DEFAULT_HEIGHT, REVIEWED_COMPACT_FRAMES),
            Some(37_296)
        );
        assert_eq!(target_audio_rows(REVIEWED_COMPACT_FRAMES), Some(414));
        assert_eq!(
            audio_latents_per_channel(REVIEWED_COMPACT_FRAMES),
            Some(207)
        );
        // The smaller campaign canvas.
        assert_eq!(rows_per_video_latent(768, 768), Some(576));
        assert_eq!(
            target_video_rows(768, 768, REVIEWED_COMPACT_FRAMES),
            Some(21_312)
        );
        // Grid endpoints.
        assert_eq!(video_latent_frames(MIN_FRAMES), Some(32));
        assert_eq!(video_latent_frames(MAX_FRAMES), Some(102));
        // The two audio sample counts are deliberately different quantities.
        assert_eq!(
            audio_samples_per_channel(REVIEWED_COMPACT_FRAMES),
            Some(165_333)
        );
        assert_eq!(
            vocoder_audio_samples_per_channel(REVIEWED_COMPACT_FRAMES),
            Some(165_600)
        );
    }

    /// The compact steps axis is a RANGE for the base tier. The Turbo tiers
    /// keep their exact distilled counts, which is a property of the adapter.
    #[test]
    fn the_compact_steps_range_is_derived_and_contains_every_reviewed_count() {
        assert_eq!(COMPACT_MIN_STEPS, 2);
        assert_eq!(COMPACT_MAX_STEPS, DEFAULT_STEPS);
        assert_eq!(COMPACT_MAX_STEPS, 50);
        assert!((COMPACT_MIN_STEPS..=COMPACT_MAX_STEPS).contains(&COMFY_DEFAULT_STEPS));
        for tier in REVIEWED_TURBO_MANIFEST_TIERS {
            assert!(
                (COMPACT_MIN_STEPS..=COMPACT_MAX_STEPS).contains(&tier.steps),
                "{} steps {}",
                tier.model,
                tier.steps
            );
        }
    }

    /// A compact tag renders one of the reviewed canvases and nothing else;
    /// the hidden official BF16 references keep the family envelope.
    #[test]
    fn compact_dimensions_are_model_aware() {
        for model in [
            FL2VA_COMFY,
            REF2VA_COMFY,
            FL2VA_COMFY_TURBO_8STEP,
            FL2VA_COMFY_TURBO_4STEP_768P,
            REF2VA_COMFY_TURBO_4STEP,
            FL2VA_COMFY_TURBO_4STEP_768P_V11,
            FL2VA_COMFY_TURBO_8STEP_768P,
            FL2VA_COMFY_TURBO_4STEP_768P_R21,
            FL2VA_COMFY_TURBO_8STEP_R21,
            REF2VA_COMFY_TURBO_4STEP_R21,
        ] {
            for &(width, height) in REVIEWED_COMPACT_CANVASES {
                assert!(
                    valid_dimensions_for_model(FAMILY, model, width, height),
                    "{model} {width}x{height}"
                );
            }
            // The rule admits far more than the preset list: 4:3 and the
            // default's own transpose were both refused by set membership.
            for (width, height) in [(1024, 768), (768, 1344), (1024, 576)] {
                assert!(
                    valid_dimensions_for_model(FAMILY, model, width, height),
                    "{model} {width}x{height}"
                );
            }
            // Off the stride, over the area ceiling, and under the axis floor.
            for (width, height) in [(1000, 600), (1056, 992), (224, 896)] {
                assert!(
                    !valid_dimensions_for_model(FAMILY, model, width, height),
                    "{model} {width}x{height}"
                );
                // Repair lands on a canvas that runs, never on the free-form
                // family resolver's answer.
                let repair = recommended_dimensions_for_model(FAMILY, model, width, height);
                assert!(
                    is_admitted_compact_canvas(repair.0, repair.1),
                    "{model} {width}x{height} -> {repair:?}"
                );
            }
            assert_eq!(
                recommended_dimensions_for_model(FAMILY, model, 1000, 600),
                (1280, 768),
                "{model}"
            );
        }

        for model in [FL2VA_OFFICIAL, REF2VA_OFFICIAL] {
            assert!(
                qualified_canvases_for_model(FAMILY, model).is_none(),
                "{model}"
            );
            assert!(
                valid_dimensions_for_model(FAMILY, model, 1024, 768),
                "{model}"
            );
            assert_eq!(
                recommended_dimensions_for_model(FAMILY, model, 1024, 1024),
                recommended_dimensions(1024, 1024),
                "{model}"
            );
        }
    }

    /// Source fitting renders the SOURCE'S own aspect at the largest size the
    /// compact area ceiling admits. It used to pick the nearest of two fixed
    /// canvases, which letterboxed every source that was neither 7:4 nor
    /// square.
    #[test]
    fn compact_source_fit_renders_the_source_aspect_at_the_ceiling() {
        for model in [FL2VA_COMFY, FL2VA_COMFY_TURBO_8STEP] {
            // 16:9 is no longer flattened into the 7:4 default.
            assert_eq!(
                source_fit_dimensions(model, 1920, 1080),
                (1312, 736),
                "{model}"
            );
            // A source already on a preset keeps it exactly.
            assert_eq!(
                source_fit_dimensions(model, 1344, 768),
                (1344, 768),
                "{model}"
            );
            // A square source stays square, and larger than 768x768.
            assert_eq!(
                source_fit_dimensions(model, 1024, 1024),
                (992, 992),
                "{model}"
            );
            // 3:4 stays 3:4 rather than collapsing to 1:1.
            assert_eq!(
                source_fit_dimensions(model, 768, 1024),
                (864, 1152),
                "{model}"
            );
            // Past the family aspect bound the aspect is clamped to 1:4.
            assert_eq!(
                source_fit_dimensions(model, 512, 2048),
                (480, 1920),
                "{model}"
            );
            assert_eq!(
                source_fit_dimensions(model, 100, 2048),
                (480, 1920),
                "{model}"
            );
            // A degenerate source keeps the default.
            assert_eq!(
                source_fit_dimensions(model, 0, 0),
                (DEFAULT_WIDTH, DEFAULT_HEIGHT),
                "{model}"
            );
            // Every answer is itself admitted, and never letterboxed by more
            // than one stride of aspect drift.
            for source in [
                (1920, 1080),
                (1024, 1024),
                (768, 1024),
                (3, 1),
                (1, 3),
                (2560, 1080),
                (640, 480),
            ] {
                let (width, height) = source_fit_dimensions(model, source.0, source.1);
                assert!(
                    is_admitted_compact_canvas(width, height),
                    "{model} {source:?} -> {width}x{height}"
                );
                let wanted = (f64::from(source.0) / f64::from(source.1))
                    .clamp(MIN_ASPECT_RATIO, MAX_ASPECT_RATIO);
                let got = f64::from(width) / f64::from(height);
                assert!(
                    (got / wanted).ln().abs() < 0.05,
                    "{model} {source:?} -> {width}x{height} drifted from {wanted}"
                );
            }
        }
        // The hidden official reference keeps its flexible resolver.
        assert_eq!(
            source_fit_dimensions(FL2VA_OFFICIAL, 1024, 1024),
            recommended_dimensions(1024, 1024)
        );
    }

    /// Mold's canvas resolver is a port of ComfyUI's `adapt_canvas`
    /// (`comfy_extras/nodes_minimax_h3.py:50-61`), which is the reference for
    /// the compact checkpoints. Its constants are pinned here so a drift in
    /// either is a failure rather than a silently different canvas.
    #[test]
    fn canvas_constants_match_the_comfyui_reference() {
        // CANVAS_MULTIPLE = 32, BASE_SHORT_EDGE = 768, MAX_PIXELS = 768 * 1344
        assert_eq!(DIMENSION_ALIGNMENT, 32);
        assert_eq!(CANVAS_SHORT_EDGE, 768);
        assert_eq!(CANVAS_MAX_PIXELS, 768 * 1344);
        assert_eq!(FIXED_FPS, 24);
        // `align_frame_count`: `while n % 17 != 5: n += 1`.
        for frames in [MIN_FRAMES, REVIEWED_COMPACT_FRAMES, MAX_FRAMES] {
            assert_eq!(frames % FRAME_STEP, FRAME_OFFSET, "{frames} on the grid");
        }
    }

    #[test]
    fn canvas_recommendations_match_the_official_short_edge_area_then_round_order() {
        for ((aspect_width, aspect_height), expected) in [
            ((21, 9), (1536, 672)),
            ((16, 9), (1344, 768)),
            ((4, 3), (1024, 768)),
            ((1, 1), (768, 768)),
            ((3, 4), (768, 1024)),
            ((9, 16), (768, 1344)),
            ((4, 1), (2016, 512)),
            ((1, 4), (512, 2016)),
        ] {
            assert_eq!(
                recommended_dimensions(aspect_width, aspect_height),
                expected,
                "aspect {aspect_width}:{aspect_height}"
            );
        }

        // The oracle's independent post-budget rounding may exceed the
        // pre-rounding area. 7:23 is one ratio that reaches the exact envelope.
        let rounded = recommended_dimensions(7, 23);
        assert_eq!(rounded, (576, 1856));
        assert_eq!(u64::from(rounded.0) * u64::from(rounded.1), MAX_PIXELS);
    }

    #[test]
    fn capabilities_are_truthful_for_the_public_fl2va_runtime() {
        for task in [Task::Fl2va, Task::Ref2va] {
            let caps = capabilities(task);
            let fl2va_runtime = cfg!(feature = "h3") && task == Task::Fl2va;
            assert_eq!(caps.runtime_available, fl2va_runtime);
            assert_eq!(caps.native_batch_sizes, &[1]);
            assert_eq!(
                caps.backends.cuda,
                if fl2va_runtime {
                    BackendQualification::Supported
                } else {
                    BackendQualification::ContractTarget
                }
            );
            // #1164: Metal is a real execution path, qualified for
            // correctness only. CPU stays unsupported — a real capability
            // limit, not a licence gate.
            assert_eq!(caps.backends.metal, BackendQualification::CorrectnessOnly);
            assert_eq!(caps.backends.cpu, BackendQualification::Unsupported);
            assert!(caps.synchronized_audio);
            assert!(!caps.audio_disable_supported);
            assert_eq!(caps.audio_sample_rate_hz, 32_000);
            assert_eq!(caps.audio_channels, 2);
            // Derived, not transcribed: the advertised window is the model
            // card's own 4-15 seconds (`MiniMax-AI/MiniMax-H3`
            // `README.md:73`).
            assert_eq!(
                (caps.min_duration_seconds, caps.max_duration_seconds),
                (MIN_DURATION_SECONDS, MAX_DURATION_SECONDS)
            );
            assert_eq!(
                (caps.min_duration_seconds, caps.max_duration_seconds),
                (4, 15)
            );
            assert_eq!(caps.default_dimensions, (1344, 768));
            assert_eq!(caps.min_aspect_ratio, (1, 4));
            assert_eq!(caps.max_aspect_ratio, (4, 1));
            assert_eq!(caps.noise_domain_version, NOISE_DOMAIN_VERSION);
            assert_eq!(caps.noise_streams, NOISE_STREAMS);
            assert_eq!(
                caps.noise_draws,
                &[
                    NoiseDrawContract {
                        name: NOISE_STREAMS[0],
                        seed_source: NoiseSeedSource::FixedFreshPerVisualCondition(42),
                        cardinality: NoiseDrawCardinality::PerVisualCondition,
                    },
                    NoiseDrawContract {
                        name: NOISE_STREAMS[1],
                        seed_source: NoiseSeedSource::RequestSeed,
                        cardinality: NoiseDrawCardinality::PerVisualCondition,
                    },
                    NoiseDrawContract {
                        name: NOISE_STREAMS[2],
                        seed_source: NoiseSeedSource::RequestSeed,
                        cardinality: NoiseDrawCardinality::Once,
                    },
                    NoiseDrawContract {
                        name: NOISE_STREAMS[3],
                        seed_source: NoiseSeedSource::RequestSeed,
                        cardinality: NoiseDrawCardinality::Once,
                    },
                ]
            );
        }
    }

    #[test]
    fn per_model_capability_authority_covers_five_modes_without_advertising_runtime() {
        let cases = [
            (
                FL2VA_OFFICIAL,
                Task::Fl2va,
                Layout::OfficialBf16,
                FL2VA_MODES,
            ),
            (
                REF2VA_OFFICIAL,
                Task::Ref2va,
                Layout::OfficialBf16,
                REF2VA_MODES,
            ),
            (
                FL2VA_COMFY,
                Task::Fl2va,
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
                FL2VA_MODES,
            ),
            (
                REF2VA_COMFY,
                Task::Ref2va,
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
                REF2VA_MODES,
            ),
            (
                FL2VA_COMFY_TURBO_8STEP,
                Task::Fl2va,
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
                FL2VA_MODES,
            ),
            (
                FL2VA_COMFY_TURBO_4STEP_768P,
                Task::Fl2va,
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
                FL2VA_MODES,
            ),
            (
                REF2VA_COMFY_TURBO_4STEP,
                Task::Ref2va,
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
                REF2VA_MODES,
            ),
            (
                FL2VA_COMFY_TURBO_4STEP_768P_V11,
                Task::Fl2va,
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
                FL2VA_MODES,
            ),
            (
                FL2VA_COMFY_TURBO_8STEP_768P,
                Task::Fl2va,
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
                FL2VA_MODES,
            ),
            (
                FL2VA_COMFY_TURBO_4STEP_768P_R21,
                Task::Fl2va,
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
                FL2VA_MODES,
            ),
            (
                FL2VA_COMFY_TURBO_8STEP_R21,
                Task::Fl2va,
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
                FL2VA_MODES,
            ),
            (
                REF2VA_COMFY_TURBO_4STEP_R21,
                Task::Ref2va,
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
                REF2VA_MODES,
            ),
        ];
        let mut observed_modes = Vec::new();
        for (model, task, layout, modes) in cases {
            let contract = capability_contract_for_model(model).unwrap();
            assert_eq!(contract.canonical_model, model);
            assert_eq!(contract.task, task);
            assert_eq!(contract.layout, layout);
            assert_eq!(contract.generation.modes, modes);
            // `backends` is keyed on the task partition mold implements;
            // `runtime_available` narrows that by the layout this build can
            // actually load, so a qualification reference reports a CUDA
            // contract target and no runtime at once.
            let task_runnable = engine_is_built() && task_runtime_available(task);
            let runnable = task_runnable && layout_runtime_available(layout);
            assert_eq!(contract.generation.runtime_available, runnable);
            assert_eq!(contract.generation.native_batch_sizes, &[1]);
            assert_eq!(
                contract.generation.backends,
                BackendApplicability {
                    cuda: if task_runnable {
                        BackendQualification::Supported
                    } else {
                        BackendQualification::ContractTarget
                    },
                    metal: BackendQualification::CorrectnessOnly,
                    cpu: BackendQualification::Unsupported,
                }
            );
            assert!(contract.generation.synchronized_audio);
            assert_eq!(
                runnable_capability_contract_for_model(model).is_some(),
                runnable
            );
            observed_modes.extend_from_slice(modes);
        }
        observed_modes.sort_by_key(|mode| *mode as u8);
        observed_modes.dedup();
        assert_eq!(observed_modes, ALL_MODES);

        for alias in FAMILY_ALIASES {
            assert!(capability_contract_for_model(alias).is_some());
            assert_eq!(
                runnable_capability_contract_for_model(alias).is_some(),
                cfg!(feature = "h3")
            );
        }

        let advertised = crate::build_model_catalog(&crate::Config::default(), None, false);
        let advertised_h3 = advertised
            .iter()
            .filter(|model| model.family == FAMILY)
            .map(|model| model.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            advertised_h3,
            std::collections::BTreeSet::from([
                FL2VA_OFFICIAL,
                REF2VA_OFFICIAL,
                FL2VA_COMFY,
                REF2VA_COMFY,
                FL2VA_COMFY_NVFP4,
                REF2VA_COMFY_NVFP4,
                FL2VA_COMFY_TURBO_8STEP,
                FL2VA_COMFY_TURBO_4STEP_768P,
                REF2VA_COMFY_TURBO_4STEP,
                FL2VA_COMFY_TURBO_4STEP_768P_V11,
                FL2VA_COMFY_TURBO_8STEP_768P,
                FL2VA_COMFY_TURBO_4STEP_768P_R21,
                FL2VA_COMFY_TURBO_8STEP_R21,
                REF2VA_COMFY_TURBO_4STEP_R21,
            ])
        );
        assert!(advertised
            .iter()
            .filter(|model| model.family == FAMILY)
            .all(|model| model.defaults.description.contains("downloadable")));
    }

    #[test]
    fn manifests_expose_every_pinned_download_and_cannot_mix_task_transformers() {
        let fl_official = find_manifest(FL2VA_OFFICIAL).unwrap();
        let ref_official = find_manifest(REF2VA_OFFICIAL).unwrap();
        let fl_comfy = find_manifest(FL2VA_COMFY).unwrap();
        let ref_comfy = find_manifest(REF2VA_COMFY).unwrap();
        let fl_turbo_8 = find_manifest(FL2VA_COMFY_TURBO_8STEP).unwrap();
        let fl_turbo_4 = find_manifest(FL2VA_COMFY_TURBO_4STEP_768P).unwrap();
        let fl_turbo_4_v11 = find_manifest(FL2VA_COMFY_TURBO_4STEP_768P_V11).unwrap();
        let fl_turbo_8_768p = find_manifest(FL2VA_COMFY_TURBO_8STEP_768P).unwrap();
        let fl_turbo_4_768p_r21 = find_manifest(FL2VA_COMFY_TURBO_4STEP_768P_R21).unwrap();
        let fl_turbo_8_r21 = find_manifest(FL2VA_COMFY_TURBO_8STEP_R21).unwrap();
        let ref_turbo_4_r21 = find_manifest(REF2VA_COMFY_TURBO_4STEP_R21).unwrap();
        let fl_nvfp4 = find_manifest(FL2VA_COMFY_NVFP4).unwrap();
        let ref_nvfp4 = find_manifest(REF2VA_COMFY_NVFP4).unwrap();
        assert!(!fl_nvfp4.hidden);
        assert!(!ref_nvfp4.hidden);
        assert!(!fl_official.hidden);
        assert!(!ref_official.hidden);
        assert!(!fl_comfy.hidden);
        assert!(!ref_comfy.hidden);
        assert!(!fl_turbo_8.hidden);
        assert!(!fl_turbo_4.hidden);
        assert!(!fl_turbo_4_v11.hidden);
        assert!(!fl_turbo_8_768p.hidden);
        assert!(!fl_turbo_4_768p_r21.hidden);
        assert!(!fl_turbo_8_r21.hidden);
        assert!(!ref_turbo_4_r21.hidden);
        for manifest in [
            fl_official,
            ref_official,
            fl_comfy,
            ref_comfy,
            fl_turbo_8,
            fl_turbo_4,
            fl_turbo_4_v11,
            fl_turbo_8_768p,
            fl_turbo_4_768p_r21,
            fl_turbo_8_r21,
            ref_turbo_4_r21,
            fl_nvfp4,
            ref_nvfp4,
        ] {
            let contract = manifest_contract(manifest).unwrap();
            // A layout with no engine arm never advertises a runtime, however
            // runnable its task is.
            assert_eq!(
                contract.runtime_available,
                cfg!(feature = "h3")
                    && contract.task == Task::Fl2va
                    && layout_runtime_available(contract.layout)
            );
            assert_eq!(contract.license_url, MINIMAX_H3_LICENSE_URL);
            assert_eq!(contract.license_sha256, LICENSE_SHA256);
            assert_eq!(
                contract.source_revision,
                repo_revision(contract.source_repo).unwrap()
            );
            assert_eq!(
                contract.shared_identity_scheme,
                "hf-repo+revision+path+sha256"
            );
            assert!(!contract.implementation_revision.is_empty());
            assert_eq!(contract.diffusers_reference_repo, DIFFUSERS_REFERENCE_REPO);
            assert_eq!(
                contract.diffusers_reference_revision,
                DIFFUSERS_REFERENCE_REVISION
            );
            for artifact in &manifest.files {
                let metadata = artifact_contract(manifest, artifact).unwrap();
                assert_eq!(
                    Some(metadata.identity.source_revision),
                    file_revision(metadata.identity.source_repo, metadata.identity.source_path)
                );
                assert_eq!(metadata.identity.source_path, artifact.hf_filename);
                assert_eq!(Some(metadata.identity.sha256), artifact.sha256);
                assert_eq!(metadata.license_url, MINIMAX_H3_LICENSE_URL);
                assert_eq!(metadata.license_sha256, LICENSE_SHA256);
            }
            let task_transformer = manifest
                .files
                .iter()
                .find(|artifact| {
                    matches!(
                        artifact.component,
                        ModelComponent::Transformer | ModelComponent::TransformerShard
                    )
                })
                .unwrap();
            assert_eq!(
                artifact_contract(manifest, task_transformer)
                    .unwrap()
                    .identity
                    .source_repo,
                contract.source_repo
            );
        }

        let task_files = |manifest: &ModelManifest| {
            manifest
                .files
                .iter()
                .filter(|file| {
                    matches!(
                        artifact_contract(manifest, file).map(|metadata| metadata.role),
                        Some(ArtifactRole::TaskTransformer | ArtifactRole::TaskConfig)
                    )
                })
                .map(|file| file.hf_filename.clone())
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert!(task_files(fl_official).is_disjoint(&task_files(ref_official)));
        assert!(task_files(fl_comfy).is_disjoint(&task_files(ref_comfy)));

        let visible = crate::manifest::visible_manifests()
            .filter(|manifest| manifest.family == FAMILY)
            .map(|manifest| manifest.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            visible,
            std::collections::BTreeSet::from([
                FL2VA_OFFICIAL,
                REF2VA_OFFICIAL,
                FL2VA_COMFY,
                REF2VA_COMFY,
                FL2VA_COMFY_NVFP4,
                REF2VA_COMFY_NVFP4,
                FL2VA_COMFY_TURBO_8STEP,
                FL2VA_COMFY_TURBO_4STEP_768P,
                REF2VA_COMFY_TURBO_4STEP,
                FL2VA_COMFY_TURBO_4STEP_768P_V11,
                FL2VA_COMFY_TURBO_8STEP_768P,
                FL2VA_COMFY_TURBO_4STEP_768P_R21,
                FL2VA_COMFY_TURBO_8STEP_R21,
                REF2VA_COMFY_TURBO_4STEP_R21,
            ])
        );

        let ref_transformer = ref_comfy
            .files
            .iter()
            .find(|file| file.component == ModelComponent::Transformer)
            .unwrap();
        assert!(artifact_contract(fl_comfy, ref_transformer).is_none());
    }

    #[test]
    fn turbo_tags_are_first_class_compact_identities_of_their_own_task() {
        for tier in REVIEWED_TURBO_MANIFEST_TIERS {
            assert_eq!(resolve_model_name(tier.model), Some(tier.model));
            assert_eq!(
                resolve_model_name(&tier.model.to_ascii_uppercase()),
                Some(tier.model)
            );
            let expected_task = if tier.model.starts_with("minimax-h3-ref2va:") {
                Task::Ref2va
            } else {
                Task::Fl2va
            };
            assert_eq!(task_for_model(tier.model), Some(expected_task));
            assert_eq!(
                layout_for_model(tier.model),
                Some(Layout::ComfyPrunedInt8ConvrotNvfp4Awq)
            );
            assert!(is_reviewed_compact_model(tier.model));
            assert_eq!(turbo_tier_for_model(tier.model), Some(tier));
        }
        // The base compact identities select no Turbo tier, and lookalike
        // tags resolve to nothing at all.
        assert_eq!(turbo_tier_for_model(FL2VA_COMFY), None);
        assert_eq!(turbo_tier_for_model(REF2VA_COMFY), None);
        for lookalike in [
            "minimax-h3-fl2va:comfy-pruned-int8-turbo-2step",
            "minimax-h3-ref2va:comfy-pruned-int8-turbo-8step",
            "minimax-h3",
        ] {
            assert_eq!(turbo_tier_for_model(lookalike), None, "{lookalike}");
            assert!(!is_reviewed_compact_model(lookalike), "{lookalike}");
        }
        assert!(is_reviewed_compact_model(FL2VA_COMFY));
        assert!(is_reviewed_compact_model(REF2VA_COMFY));
        // The compact engine partition exists only for compact-layout
        // identities: Turbo tags map to their task's base, the bases map to
        // themselves, and the official BF16 references map to nothing.
        for tier in REVIEWED_TURBO_MANIFEST_TIERS {
            assert_eq!(
                base_compact_model(tier.model),
                Some(base_compact_model_for_task(
                    task_for_model(tier.model).unwrap()
                )),
                "{}",
                tier.model
            );
        }
        assert_eq!(base_compact_model(FL2VA_COMFY), Some(FL2VA_COMFY));
        assert_eq!(base_compact_model(REF2VA_COMFY), Some(REF2VA_COMFY));
        assert_eq!(base_compact_model(FL2VA_OFFICIAL), None);
        assert_eq!(base_compact_model(REF2VA_OFFICIAL), None);
    }

    #[test]
    fn turbo_manifests_pin_the_adapter_beside_the_exact_base_stack() {
        for tier in REVIEWED_TURBO_MANIFEST_TIERS {
            let manifest = find_manifest(tier.model).unwrap();
            let expected_task = task_for_model(tier.model).unwrap();
            // A Turbo tag is its OWN task's base stack plus one adapter, so
            // the base it must match byte-for-byte is that task's partition.
            let base = find_manifest(base_compact_model_for_task(expected_task)).unwrap();
            assert_eq!(manifest.family, FAMILY);
            assert!(!manifest.hidden);
            // The tier moves exactly one manifest default: the reviewed
            // terminal-inclusive step count.
            assert_eq!(manifest.defaults.steps, tier.steps);
            assert_eq!(manifest.defaults.width, DEFAULT_WIDTH);
            assert_eq!(manifest.defaults.height, DEFAULT_HEIGHT);
            assert_eq!(manifest.defaults.frames, Some(REVIEWED_COMPACT_FRAMES));
            assert_eq!(manifest.defaults.fps, Some(FIXED_FPS));

            // Every base compact file is present byte-for-byte, plus exactly
            // one pinned adapter.
            for file in &base.files {
                assert!(
                    manifest.files.contains(file),
                    "{} missing base file {}",
                    tier.model,
                    file.hf_filename
                );
            }
            let adapters: Vec<_> = manifest
                .files
                .iter()
                .filter(|file| file.component == ModelComponent::DistilledLora)
                .collect();
            let [adapter] = adapters.as_slice() else {
                panic!("{} must pin exactly one Turbo adapter", tier.model);
            };
            assert_eq!(adapter.hf_repo, tier.adapter_hf_repo);
            // An adapter repository with no pinned revision would reach
            // `download`'s unpinned `main` fetch, so the source table and the
            // revision table can never disagree.
            assert!(
                repo_revision(tier.adapter_hf_repo).is_some(),
                "{} adapter repo has no pinned revision",
                tier.model
            );
            assert_eq!(adapter.hf_filename, tier.adapter_hf_filename);
            assert_eq!(adapter.size_bytes, tier.adapter_size_bytes);
            assert_eq!(adapter.sha256, Some(tier.adapter_sha256));
            assert_eq!(manifest.files.len(), base.files.len() + 1);

            // The adapter downloads from its OWN pinned source revision —
            // Comfy-Org's later `loras/` publication, or lightx2v's or
            // drbaph's repository root — while the base stack stays on the
            // reviewed pin.
            assert_eq!(
                file_revision(&adapter.hf_repo, &adapter.hf_filename),
                Some(tier.adapter_hf_revision)
            );
            let transformer = manifest
                .files
                .iter()
                .find(|file| file.component == ModelComponent::Transformer)
                .unwrap();
            assert_eq!(
                file_revision(&transformer.hf_repo, &transformer.hf_filename),
                Some(COMFY_REVISION)
            );

            // Every Turbo tag shares one on-disk adapter copy under the
            // family `loras/` bucket, keyed by the adapter's own basename
            // whatever directory its upstream repository publishes it in.
            assert_eq!(
                storage_path(manifest, adapter),
                std::path::PathBuf::from("shared")
                    .join(FAMILY)
                    .join("loras")
                    .join(
                        std::path::Path::new(tier.adapter_hf_filename)
                            .file_name()
                            .unwrap()
                    )
            );

            // The adapter is reviewed for exactly its own task partition, so
            // a `ref2v` adapter can never mint an FL2VA qualification.
            let contract = artifact_contract(manifest, adapter).unwrap();
            assert_eq!(contract.role, ArtifactRole::TurboLoraAdapter);
            assert_eq!(
                contract.compatible_tasks,
                if expected_task == Task::Ref2va {
                    REF2VA_ONLY
                } else {
                    FL2VA_ONLY
                }
            );
            assert_eq!(contract.identity.source_repo, tier.adapter_hf_repo);
            assert_eq!(contract.identity.source_revision, tier.adapter_hf_revision);
            assert_eq!(contract.identity.sha256, tier.adapter_sha256);
        }
    }

    /// Every reviewed adapter comes from one of exactly three reviewed
    /// repositories, and they publish at different depths: Comfy-Org under
    /// `loras/`, lightx2v and drbaph at their repository roots. Nothing else
    /// may appear here without a `repo_revision` arm, and no two tiers may
    /// collide on a repository path, an on-disk basename, a digest, a stable
    /// id, or a tag.
    #[test]
    fn turbo_adapter_sources_are_exactly_the_three_reviewed_repositories() {
        let mut repo_paths = std::collections::BTreeSet::new();
        let mut basenames = std::collections::BTreeSet::new();
        let mut digests = std::collections::BTreeSet::new();
        let mut stable_ids = std::collections::BTreeSet::new();
        let mut models = std::collections::BTreeSet::new();
        let mut labels = std::collections::BTreeSet::new();
        for tier in REVIEWED_TURBO_MANIFEST_TIERS {
            match tier.adapter_hf_repo {
                COMFY_REPO => {
                    assert_eq!(
                        tier.adapter_hf_revision, COMFY_TURBO_LORA_REVISION,
                        "{}",
                        tier.model
                    );
                    assert!(
                        tier.adapter_hf_filename.starts_with("loras/"),
                        "{} Comfy-Org adapter is not under loras/",
                        tier.model
                    );
                }
                LIGHTX2V_REPO => {
                    assert_eq!(
                        tier.adapter_hf_revision, LIGHTX2V_REVISION,
                        "{}",
                        tier.model
                    );
                    assert!(
                        !tier.adapter_hf_filename.contains('/'),
                        "{} lightx2v adapter is not at the repository root",
                        tier.model
                    );
                }
                DRBAPH_TURBO_LORA_REPO => {
                    assert_eq!(
                        tier.adapter_hf_revision, DRBAPH_TURBO_LORA_REVISION,
                        "{}",
                        tier.model
                    );
                    assert!(
                        !tier.adapter_hf_filename.contains('/'),
                        "{} drbaph adapter is not at the repository root",
                        tier.model
                    );
                    // Only the SVD-resized derivatives come from drbaph, and
                    // the shape label is what the acquisition contract shows a
                    // user before a pull: a lossy approximation is never
                    // described as the rank-128 adapter it approximates.
                    assert_eq!(
                        tier.adapter_shape_label, RESIZED_TURBO_ADAPTER_SHAPE,
                        "{}",
                        tier.model
                    );
                }
                other => panic!(
                    "{} pins an unreviewed adapter repository {other}",
                    tier.model
                ),
            }
            assert!(
                repo_revision(tier.adapter_hf_repo).is_some(),
                "{} adapter repo has no pinned revision",
                tier.model
            );
            let basename = std::path::Path::new(tier.adapter_hf_filename)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap();
            assert!(
                repo_paths.insert((tier.adapter_hf_repo, tier.adapter_hf_filename)),
                "{} repeats a repository path",
                tier.model
            );
            // The storage rule flattens every adapter to its basename, so two
            // tiers sharing one basename would share one on-disk file with two
            // different digests.
            assert!(
                basenames.insert(basename),
                "{} repeats a basename",
                tier.model
            );
            assert!(
                digests.insert(tier.adapter_sha256),
                "{} repeats a digest",
                tier.model
            );
            assert!(
                stable_ids.insert(tier.tier_stable_id),
                "{} repeats a stable id",
                tier.model
            );
            assert!(
                models.insert(tier.model),
                "{} repeats a model tag",
                tier.model
            );
            // The label is the only thing a Discover row shows to tell two
            // tiers apart, so a copy-pasted one would make a lossy tier
            // indistinguishable from the adapter it approximates.
            assert!(
                labels.insert(tier.display_label),
                "{} repeats a display label",
                tier.model
            );
        }
        assert_eq!(LIGHTX2V_REPO, "lightx2v/Minimax-h3-Turbo");
        assert_eq!(
            LIGHTX2V_REVISION,
            "05ef678438e84933c406131b59abbf86919b3aac"
        );
        assert_eq!(repo_revision(LIGHTX2V_REPO), Some(LIGHTX2V_REVISION));
        assert_eq!(
            DRBAPH_TURBO_LORA_REPO,
            "drbaph/MiniMax-H3-Turbo-Lora-ComfyUI"
        );
        assert_eq!(
            DRBAPH_TURBO_LORA_REVISION,
            "be8eb3ea3466cbb7def202ffec0d2fdc054256ac"
        );
        assert_eq!(
            repo_revision(DRBAPH_TURBO_LORA_REPO),
            Some(DRBAPH_TURBO_LORA_REVISION)
        );
        // All three repositories are actually exercised, so the match arms
        // above cannot pass vacuously if a source is dropped from the table.
        assert_eq!(
            REVIEWED_TURBO_MANIFEST_TIERS
                .iter()
                .map(|tier| tier.adapter_hf_repo)
                .collect::<std::collections::BTreeSet<_>>(),
            std::collections::BTreeSet::from([COMFY_REPO, LIGHTX2V_REPO, DRBAPH_TURBO_LORA_REPO])
        );
    }

    /// A user reading an acquisition contract must be able to tell a lossy
    /// SVD-resized adapter from the rank-128 adapter it approximates, so the
    /// `DistilledLora` shape sentence — and the manifest description the
    /// Discover row, `/api/models`, and `mold list` actually show — are both
    /// derived from the tier row rather than written once for every adapter.
    #[test]
    fn the_distilled_lora_shape_names_the_tier_it_actually_describes() {
        let mut seen_uniform = false;
        let mut seen_resized = false;
        for tier in REVIEWED_TURBO_MANIFEST_TIERS {
            let manifest = find_manifest(tier.model).unwrap();
            let adapter = manifest
                .files
                .iter()
                .find(|file| file.component == ModelComponent::DistilledLora)
                .unwrap();
            let contract = artifact_contract(manifest, adapter).unwrap();
            assert_eq!(
                contract.role,
                ArtifactRole::TurboLoraAdapter,
                "{}",
                tier.model
            );
            assert_eq!(contract.dtype, "bf16", "{}", tier.model);
            assert_eq!(contract.shape, tier.adapter_shape_label, "{}", tier.model);
            // The description is the only sentence most clients ever render
            // for this row, so it carries the same disclosure the contract
            // shape does.
            assert!(
                manifest.description.contains(tier.display_label),
                "{}",
                tier.model
            );
            match tier.adapter_shape_label {
                UNIFORM_TURBO_ADAPTER_SHAPE => {
                    seen_uniform = true;
                    assert_ne!(
                        tier.adapter_hf_repo, DRBAPH_TURBO_LORA_REPO,
                        "{}",
                        tier.model
                    );
                    assert!(contract.shape.contains("rank 128"), "{}", tier.model);
                    assert!(
                        !manifest.description.contains("lossy"),
                        "{} is full rank but its description calls it lossy",
                        tier.model
                    );
                    assert!(
                        manifest.description.contains("with the reviewed "),
                        "{}",
                        tier.model
                    );
                }
                RESIZED_TURBO_ADAPTER_SHAPE => {
                    seen_resized = true;
                    assert!(contract.shape.contains("avg 21"), "{}", tier.model);
                    assert!(contract.shape.contains("baked scale"), "{}", tier.model);
                    assert!(!contract.shape.contains("rank 128"), "{}", tier.model);
                    assert!(
                        manifest
                            .description
                            .contains("with the reviewed, lossy SVD-resized "),
                        "{} description does not disclose the resize: {}",
                        tier.model,
                        manifest.description
                    );
                }
                other => panic!("{} declares an unreviewed shape {other}", tier.model),
            }
            // Both shapes describe the same 208-module target set; only the
            // per-module rank and the baked scale differ.
            assert!(contract.shape.contains("208 modules"), "{}", tier.model);
        }
        assert!(seen_uniform && seen_resized);
    }

    /// `file_revision` is the only thing standing between an adapter path and
    /// an unpinned `main` fetch, so it resolves through the tier table on the
    /// exact `(repo, path)` pair rather than on a repository special case.
    #[test]
    fn file_revision_pins_each_adapter_to_its_own_source() {
        let lightx2v = REVIEWED_TURBO_MANIFEST_TIERS
            .iter()
            .find(|tier| tier.model == FL2VA_COMFY_TURBO_4STEP_768P_V11)
            .unwrap();
        let comfy = REVIEWED_TURBO_MANIFEST_TIERS
            .iter()
            .find(|tier| tier.model == FL2VA_COMFY_TURBO_8STEP)
            .unwrap();
        assert_eq!(
            file_revision(LIGHTX2V_REPO, lightx2v.adapter_hf_filename),
            Some(LIGHTX2V_REVISION)
        );
        assert_eq!(
            file_revision(COMFY_REPO, comfy.adapter_hf_filename),
            Some(COMFY_TURBO_LORA_REVISION)
        );
        let drbaph = REVIEWED_TURBO_MANIFEST_TIERS
            .iter()
            .find(|tier| tier.model == REF2VA_COMFY_TURBO_4STEP_R21)
            .unwrap();
        assert_eq!(
            file_revision(DRBAPH_TURBO_LORA_REPO, drbaph.adapter_hf_filename),
            Some(DRBAPH_TURBO_LORA_REVISION)
        );
        // A Comfy-Org path that merely LOOKS like a lightx2v adapter is not
        // one: the tier lookup keys on the pair, so this falls back to the
        // repository-wide pin.
        assert_eq!(
            file_revision(COMFY_REPO, lightx2v.adapter_hf_filename),
            Some(COMFY_REVISION)
        );
        assert_eq!(
            file_revision("someone/else", lightx2v.adapter_hf_filename),
            None
        );

        // Every file of every reviewed manifest resolves a revision; the only
        // caller is the private `hf_file_repo`, which silently fetches `main`
        // when this answers `None`.
        for tier in REVIEWED_TURBO_MANIFEST_TIERS {
            let manifest = find_manifest(tier.model).unwrap();
            for file in &manifest.files {
                assert!(
                    file_revision(&file.hf_repo, &file.hf_filename).is_some(),
                    "{} file {} has no pinned revision",
                    tier.model,
                    file.hf_filename
                );
            }
        }
    }

    /// A Turbo tag is the base checkpoint plus one shared adapter, so every
    /// non-adapter file must resolve to the exact storage path the base
    /// manifest owns. Anything else re-downloads the ~41 GB base stack into a
    /// tag-named directory beside an identical installed copy, and removal
    /// ref-counting stops protecting the shared bytes.
    #[test]
    fn turbo_manifests_store_the_base_stack_in_the_base_models_directory() {
        for tier in REVIEWED_TURBO_MANIFEST_TIERS {
            let manifest = find_manifest(tier.model).unwrap();
            // The base a Turbo tag collapses onto is its OWN task's stack.
            let base = find_manifest(base_compact_model(tier.model).unwrap()).unwrap();
            for file in &manifest.files {
                if file.component == ModelComponent::DistilledLora {
                    continue;
                }
                let same = base
                    .files
                    .iter()
                    .find(|candidate| {
                        candidate.hf_repo == file.hf_repo
                            && candidate.hf_filename == file.hf_filename
                    })
                    .unwrap_or_else(|| {
                        panic!(
                            "{} carries {} which the base manifest does not own",
                            tier.model, file.hf_filename
                        )
                    });
                assert_eq!(
                    storage_path(manifest, file),
                    storage_path(base, same),
                    "{} must reuse the base install for {}",
                    tier.model,
                    file.hf_filename
                );
            }
        }
    }

    /// Removal ref-counting must protect the shared base stack in both
    /// directions: removing a Turbo tag deletes only its adapter, and
    /// removing the base while a Turbo tag is installed keeps every shared
    /// file, naming the tag that still uses it. A half-installed Turbo tag
    /// (its adapter deleted) is not a complete install and owns nothing.
    #[test]
    fn removing_one_compact_layout_keeps_the_other_layouts_shared_graph() {
        use crate::removal::plan_removal;
        use crate::{Config, ModelConfig};

        // Ownership is read from the manifest and requires a COMPLETE install,
        // so this must be pinned to a temp root and materialize both stacks
        // whole: the audio VAE and the runtime support configs count even
        // though no `ModelConfig` field can name them, and without the pin the
        // test consults whatever the host's real models dir happens to hold.
        let _lock = crate::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let root =
            std::env::temp_dir().join(format!("mold-h3-nvfp4-removal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::env::set_var("MOLD_MODELS_DIR", &root);
        let int8 = find_manifest(FL2VA_COMFY).unwrap();
        let nvfp4 = find_manifest(FL2VA_COMFY_NVFP4).unwrap();
        let path_of = |manifest: &ModelManifest, component: ModelComponent| {
            let file = manifest
                .files
                .iter()
                .find(|file| file.component == component)
                .unwrap();
            root.join(storage_path(manifest, file))
                .to_string_lossy()
                .to_string()
        };
        for manifest in [int8, nvfp4] {
            for file in &manifest.files {
                let path = root.join(storage_path(manifest, file));
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(&path, b"weights").unwrap();
            }
        }

        let int8_transformer = path_of(int8, ModelComponent::Transformer);
        let nvfp4_transformer = path_of(nvfp4, ModelComponent::Transformer);
        let shared_encoder = path_of(int8, ModelComponent::TextEncoder);
        assert_eq!(shared_encoder, path_of(nvfp4, ModelComponent::TextEncoder));
        assert_ne!(int8_transformer, nvfp4_transformer);

        let mut config = Config::default();
        for (name, transformer) in [
            (FL2VA_COMFY, &int8_transformer),
            (FL2VA_COMFY_NVFP4, &nvfp4_transformer),
        ] {
            config.models.insert(
                name.to_string(),
                ModelConfig {
                    transformer: Some(transformer.clone()),
                    text_encoder_files: Some(vec![shared_encoder.clone()]),
                    ..ModelConfig::default()
                },
            );
        }

        // Removing the NVFP4 layout frees its own transformer and nothing
        // else: the conditioner is still owned by the INT8 stack.
        let plan = plan_removal(&config, FL2VA_COMFY_NVFP4);
        let unique: Vec<&str> = plan
            .unique_files
            .iter()
            .map(|(path, _)| path.as_str())
            .collect();
        assert!(unique.contains(&nvfp4_transformer.as_str()));
        assert!(!unique.contains(&shared_encoder.as_str()));
        assert!(!unique.contains(&int8_transformer.as_str()));
        assert!(plan
            .shared_files
            .iter()
            .any(|(path, owners)| path == &shared_encoder
                && owners.iter().any(|owner| owner == FL2VA_COMFY)));

        // ...and the same holds in the other direction.
        let reverse = plan_removal(&config, FL2VA_COMFY);
        let reverse_unique: Vec<&str> = reverse
            .unique_files
            .iter()
            .map(|(path, _)| path.as_str())
            .collect();
        assert!(reverse_unique.contains(&int8_transformer.as_str()));
        assert!(!reverse_unique.contains(&shared_encoder.as_str()));
        assert!(reverse
            .shared_files
            .iter()
            .any(|(path, owners)| path == &shared_encoder
                && owners.iter().any(|owner| owner == FL2VA_COMFY_NVFP4)));

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn turbo_removal_refcounts_protect_the_shared_base_stack() {
        use crate::removal::plan_removal;
        use crate::{Config, ModelConfig};

        // Ownership is read from the manifest, so this must be pinned to the
        // temp root: without it the test consults — and can be satisfied by —
        // whatever the host's real models dir happens to hold.
        let _lock = crate::test_support::ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let root =
            std::env::temp_dir().join(format!("mold-h3-turbo-removal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::env::set_var("MOLD_MODELS_DIR", &root);
        let base_manifest = find_manifest(FL2VA_COMFY).unwrap();
        let tier = &REVIEWED_TURBO_MANIFEST_TIERS[0];
        let turbo_manifest = find_manifest(tier.model).unwrap();
        let path_of = |manifest: &ModelManifest, component: ModelComponent| {
            let file = manifest
                .files
                .iter()
                .find(|file| file.component == component)
                .unwrap();
            root.join(storage_path(manifest, file))
                .to_string_lossy()
                .to_string()
        };

        let adapter_path = path_of(turbo_manifest, ModelComponent::DistilledLora);
        let transformer_path = path_of(base_manifest, ModelComponent::Transformer);
        // Both stacks are materialized WHOLE. Ownership requires a complete
        // install, and completeness is a manifest question — the audio VAE
        // and the runtime support configs count even though no `ModelConfig`
        // field can name them.
        for manifest in [base_manifest, turbo_manifest] {
            for file in &manifest.files {
                let path = root.join(storage_path(manifest, file));
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::write(&path, b"weights").unwrap();
            }
        }

        let mut config = Config::default();
        config.models.insert(
            FL2VA_COMFY.to_string(),
            ModelConfig {
                transformer: Some(transformer_path.clone()),
                ..ModelConfig::default()
            },
        );
        config.models.insert(
            tier.model.to_string(),
            ModelConfig {
                transformer: Some(path_of(turbo_manifest, ModelComponent::Transformer)),
                distilled_lora: Some(adapter_path.clone()),
                ..ModelConfig::default()
            },
        );

        let plan = plan_removal(&config, tier.model);
        let unique: Vec<&str> = plan
            .unique_files
            .iter()
            .map(|(path, _)| path.as_str())
            .collect();
        assert_eq!(unique, vec![adapter_path.as_str()]);
        assert!(plan
            .shared_files
            .iter()
            .any(|(path, used_by)| path == &transformer_path
                && used_by == &vec![FL2VA_COMFY.to_string()]));

        let plan = plan_removal(&config, FL2VA_COMFY);
        assert!(
            plan.unique_files.is_empty(),
            "the base owns nothing exclusively while a Turbo tag is installed"
        );
        assert!(plan
            .shared_files
            .iter()
            .any(|(path, used_by)| path == &transformer_path
                && used_by == &vec![tier.model.to_string()]));

        // Delete the adapter: the Turbo tag is now half-installed and must
        // stop owning the base stack — removing the base frees it.
        std::fs::remove_file(&adapter_path).unwrap();
        let plan = plan_removal(&config, FL2VA_COMFY);
        assert!(
            plan.shared_files.is_empty(),
            "a Turbo tag without its adapter is not an owner: {:?}",
            plan.shared_files
        );
        assert!(plan
            .unique_files
            .iter()
            .any(|(path, _)| path == &transformer_path));

        std::env::remove_var("MOLD_MODELS_DIR");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    #[cfg(any(feature = "h3", feature = "h3-private-uat"))]
    fn turbo_requests_validate_at_their_reviewed_step_counts() {
        for tier in REVIEWED_TURBO_MANIFEST_TIERS {
            let mut req = request();
            req.model = tier.model.into();
            req.steps = tier.steps;
            assert!(crate::validation::validate_h3_private_uat_request(&req).is_ok());
        }
        let mut unreviewed = request();
        unreviewed.model = "minimax-h3-fl2va:comfy-pruned-int8-turbo-2step".into();
        assert!(crate::validation::validate_h3_private_uat_request(&unreviewed).is_err());
    }

    #[test]
    fn every_variant_has_a_complete_standalone_component_graph() {
        let required_roles = [
            ArtifactRole::TaskTransformer,
            ArtifactRole::Qwen3VlConditioner,
            ArtifactRole::VideoVae,
            ArtifactRole::AudioVae,
            ArtifactRole::Processor,
            ArtifactRole::VideoScheduler,
            ArtifactRole::AudioScheduler,
            ArtifactRole::SharedConfig,
            ArtifactRole::TaskConfig,
        ];
        for manifest_name in [
            FL2VA_OFFICIAL,
            REF2VA_OFFICIAL,
            FL2VA_COMFY,
            REF2VA_COMFY,
            FL2VA_COMFY_NVFP4,
            REF2VA_COMFY_NVFP4,
        ] {
            let manifest = find_manifest(manifest_name).unwrap();
            for role in required_roles {
                assert!(
                    manifest.files.iter().any(|file| {
                        artifact_contract(manifest, file)
                            .is_some_and(|contract| contract.role == role)
                    }),
                    "{manifest_name} is missing {role:?}"
                );
            }
        }
    }

    #[test]
    fn manifest_defaults_match_each_layouts_released_sampler_count() {
        for name in [FL2VA_OFFICIAL, REF2VA_OFFICIAL] {
            assert_eq!(find_manifest(name).unwrap().defaults.steps, DEFAULT_STEPS);
        }
        for name in [
            FL2VA_COMFY,
            REF2VA_COMFY,
            FL2VA_COMFY_NVFP4,
            REF2VA_COMFY_NVFP4,
        ] {
            assert_eq!(
                find_manifest(name).unwrap().defaults.steps,
                COMFY_DEFAULT_STEPS
            );
        }
    }

    #[test]
    fn comfy_reuses_exact_official_runtime_assets_without_official_weight_indexes() {
        for (official_name, comfy_name) in [
            (FL2VA_OFFICIAL, FL2VA_COMFY),
            (REF2VA_OFFICIAL, REF2VA_COMFY),
            (FL2VA_OFFICIAL, FL2VA_COMFY_NVFP4),
            (REF2VA_OFFICIAL, REF2VA_COMFY_NVFP4),
        ] {
            let official = find_manifest(official_name).unwrap();
            let comfy = find_manifest(comfy_name).unwrap();
            let reused = comfy
                .files
                .iter()
                .filter(|file| file.hf_repo == OFFICIAL_REPO)
                .collect::<Vec<_>>();
            assert_eq!(reused.len(), 13);
            for file in reused {
                assert!(!matches!(
                    file.hf_filename.as_str(),
                    "model_index.json" | "modular_model_index.json"
                ));
                assert!(!file.hf_filename.ends_with("safetensors.index.json"));
                let same = official
                    .files
                    .iter()
                    .find(|candidate| {
                        candidate.hf_repo == file.hf_repo
                            && candidate.hf_filename == file.hf_filename
                            && candidate.component == file.component
                            && candidate.size_bytes == file.size_bytes
                            && candidate.gated == file.gated
                            && candidate.sha256 == file.sha256
                    })
                    .unwrap_or_else(|| {
                        panic!(
                            "{comfy_name} reused asset is not exact in {official_name}: {}",
                            file.hf_filename
                        )
                    });
                assert_eq!(
                    artifact_contract(comfy, file).unwrap().identity,
                    artifact_contract(official, same).unwrap().identity
                );
                assert_eq!(storage_path(comfy, file), storage_path(official, same));
            }
        }
    }

    #[test]
    fn artifact_metadata_requires_exact_pinned_file_membership() {
        let manifest = find_manifest(FL2VA_COMFY).unwrap();
        let original = manifest.files.first().unwrap();

        let mut wrong_size = original.clone();
        wrong_size.size_bytes += 1;
        assert!(artifact_contract(manifest, &wrong_size).is_none());

        let mut wrong_gate = original.clone();
        wrong_gate.gated = !wrong_gate.gated;
        assert!(artifact_contract(manifest, &wrong_gate).is_none());
    }

    #[test]
    fn shared_components_have_one_storage_identity_per_layout() {
        for (fl_name, ref_name) in [
            (FL2VA_OFFICIAL, REF2VA_OFFICIAL),
            (FL2VA_COMFY, REF2VA_COMFY),
            (FL2VA_COMFY_NVFP4, REF2VA_COMFY_NVFP4),
        ] {
            let fl = find_manifest(fl_name).unwrap();
            let reference = find_manifest(ref_name).unwrap();
            let shared = |manifest: &ModelManifest| {
                manifest
                    .files
                    .iter()
                    .filter(|file| {
                        artifact_contract(manifest, file)
                            .is_some_and(|metadata| metadata.compatible_tasks == BOTH_TASKS)
                    })
                    .map(|file| {
                        let identity = artifact_contract(manifest, file).unwrap().identity;
                        (
                            identity.source_repo.to_string(),
                            identity.source_revision.to_string(),
                            identity.source_path.to_string(),
                            identity.sha256.to_string(),
                            storage_path(manifest, file),
                        )
                    })
                    .collect::<std::collections::BTreeSet<_>>()
            };
            assert_eq!(shared(fl), shared(reference));
        }
    }

    #[test]
    fn runtime_availability_agrees_with_the_capability_bool_for_every_task_and_layout() {
        // One authority, two shapes. `generation_capabilities` is what
        // `/api/models`, admission, and the engine registry have always read;
        // `runtime_availability_for` is the same conjunction with the
        // obstacle named. They must never disagree.
        for task in [Task::Fl2va, Task::Ref2va] {
            for layout in [
                Layout::OfficialBf16,
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
                Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq,
            ] {
                assert_eq!(
                    runtime_availability_for(task, layout).is_available(),
                    generation_capabilities(task, layout).runtime_available,
                    "{task:?}/{layout:?}"
                );
            }
        }
    }

    #[test]
    fn runtime_availability_names_the_most_permanent_obstacle_first() {
        // A layout with no loader is unrunnable on every build and for every
        // task, so it outranks the task answer, which in turn outranks "this
        // binary was not compiled with the engine" — otherwise a macOS user
        // asking about Ref2VA would be told the sm89 artifact runs it.
        assert_eq!(
            runtime_availability_for(Task::Fl2va, Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq),
            RuntimeAvailability::Unavailable(RuntimeUnavailableReason::UnsupportedLayout)
        );
        assert_eq!(
            runtime_availability_for(Task::Ref2va, Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq),
            RuntimeAvailability::Unavailable(RuntimeUnavailableReason::UnsupportedLayout)
        );
        // Both released partitions execute since #825, so the task axis no
        // longer refuses either of them; the layout answer above still
        // outranks the build answer below.
        for task in [Task::Fl2va, Task::Ref2va] {
            assert_eq!(
                runtime_availability_for(task, Layout::ComfyPrunedInt8ConvrotNvfp4Awq),
                if engine_is_built() {
                    RuntimeAvailability::Available
                } else {
                    RuntimeAvailability::Unavailable(RuntimeUnavailableReason::EngineNotBuilt)
                },
                "{task:?}"
            );
        }
    }

    #[test]
    fn official_bf16_is_a_qualification_reference_and_never_runnable() {
        // No engine arm ever reads it: `base_compact_model` answers `None`,
        // which is the gate admission consults. Advertising a runtime for it
        // was a latent contradiction (#1276).
        assert!(!layout_runtime_available(Layout::OfficialBf16));
        for name in [FL2VA_OFFICIAL, REF2VA_OFFICIAL] {
            assert_eq!(layout_for_model(name), Some(Layout::OfficialBf16), "{name}");
            assert!(base_compact_model(name).is_none(), "{name}");
            assert!(!model_runtime_availability(name).is_available(), "{name}");
            assert_eq!(
                model_runtime_availability(name).reason(),
                Some(RuntimeUnavailableReason::UnsupportedLayout),
                "{name}"
            );
        }
    }

    #[test]
    fn every_reviewed_ref2va_identity_runs_wherever_the_engine_is_built() {
        // #825 qualified the ordered-reference route, so the reviewed compact
        // Ref2VA identity is runnable on exactly the builds that link the
        // engine — never refused for its task, and never for its layout,
        // which it shares with the FL2VA compact checkpoint.
        assert!(task_runtime_available(Task::Ref2va));
        assert!(task_runtime_available(Task::Fl2va));
        assert_eq!(
            model_runtime_availability(REF2VA_COMFY).reason(),
            if engine_is_built() {
                None
            } else {
                Some(RuntimeUnavailableReason::EngineNotBuilt)
            }
        );
    }

    #[test]
    fn every_visible_h3_manifest_resolves_to_a_runtime_answer() {
        // The catalog asks this of every H3 manifest row. An identity that
        // did not resolve would silently fall to the fail-closed arm, so pin
        // that none does.
        for manifest in crate::manifest::known_manifests()
            .iter()
            .filter(|manifest| is_family(&manifest.family))
        {
            let contract = capability_contract_for_model(&manifest.name)
                .unwrap_or_else(|| panic!("{} has no capability contract", manifest.name));
            assert_eq!(
                model_runtime_availability(&manifest.name),
                runtime_availability_for(contract.task, contract.layout),
                "{}",
                manifest.name
            );
        }
    }

    #[test]
    fn every_runtime_unavailable_reason_has_a_distinct_non_empty_message() {
        let messages = [
            RuntimeUnavailableReason::UnsupportedLayout.message(),
            RuntimeUnavailableReason::UnsupportedTask.message(),
            RuntimeUnavailableReason::EngineNotBuilt.message(),
        ];
        for message in messages {
            assert!(!message.trim().is_empty());
            // Nothing here is a licensing statement.
            assert!(!message.to_ascii_lowercase().contains("licen"), "{message}");
        }
        assert_eq!(
            messages
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            messages.len()
        );
    }

    #[test]
    fn pruned_nvfp4_is_downloadable_and_never_runnable() {
        for (name, task) in [
            (FL2VA_COMFY_NVFP4, Task::Fl2va),
            (REF2VA_COMFY_NVFP4, Task::Ref2va),
        ] {
            // Recognized as an exact identity, which is what grants
            // acquisition. Aliases deliberately do not reach this layout.
            assert_eq!(resolve_model_name(name), Some(name));
            assert_eq!(task_for_model(name), Some(task));
            assert_eq!(
                layout_for_model(name),
                Some(Layout::ComfyPrunedNvfp4ConvrotNvfp4Awq)
            );

            // ...and refused at the route: no engine partition, no runtime.
            assert!(base_compact_model(name).is_none(), "{name}");
            assert!(!layout_runtime_available(
                layout_for_model(name).expect("a pinned layout")
            ));
            assert!(
                runnable_capability_contract_for_model(name).is_none(),
                "{name}"
            );
            assert!(turbo_tier_for_model(name).is_none(), "{name}");
        }
    }

    #[test]
    fn pruned_nvfp4_keeps_the_reviewed_compact_envelope() {
        // The whole reason metadata must be right before the runtime exists:
        // a row that advertises the official ladder would offer sizes the
        // engine could never render.
        for name in [FL2VA_COMFY_NVFP4, REF2VA_COMFY_NVFP4] {
            assert!(uses_reviewed_compact_envelope(FAMILY, name), "{name}");
            // The compact tags take the FAMILY frame grid; nothing pins a
            // single clip length any more.
            assert_eq!(fixed_frames_for_model(FAMILY, name), None);
            assert_eq!(recommended_frames_for_model(FAMILY, name, 200), 192);
            for frames in [MIN_FRAMES, REVIEWED_COMPACT_FRAMES, MAX_FRAMES] {
                assert!(
                    valid_frame_count_for_model(FAMILY, name, frames),
                    "{name} {frames}"
                );
            }
            assert!(!valid_frame_count_for_model(FAMILY, name, 121), "{name}");
            assert!(!valid_frame_count_for_model(FAMILY, name, 362), "{name}");
            // A source image never moves the canvas off the compact rule.
            for source in [(1920, 1080), (512, 900)] {
                let fitted = source_fit_dimensions(name, source.0, source.1);
                assert!(
                    is_admitted_compact_canvas(fitted.0, fitted.1),
                    "{name} {source:?} -> {fitted:?}"
                );
            }

            let defaults = &find_manifest(name).unwrap().defaults;
            assert_eq!(defaults.width, DEFAULT_WIDTH);
            assert_eq!(defaults.height, DEFAULT_HEIGHT);
            assert_eq!(defaults.frames, Some(REVIEWED_COMPACT_FRAMES));
            assert_eq!(defaults.fps, Some(FIXED_FPS));
            assert_eq!(defaults.steps, COMFY_DEFAULT_STEPS);
        }
    }

    #[test]
    fn pruned_nvfp4_shares_the_compact_graph_and_owns_only_its_transformer() {
        for (int8_name, nvfp4_name) in [
            (FL2VA_COMFY, FL2VA_COMFY_NVFP4),
            (REF2VA_COMFY, REF2VA_COMFY_NVFP4),
        ] {
            let int8 = find_manifest(int8_name).unwrap();
            let nvfp4 = find_manifest(nvfp4_name).unwrap();

            // Every non-transformer file is byte-identical to the INT8 stack
            // and lands on the same shared path, so an installed compact
            // variant makes this pull exactly one file.
            let non_transformer = |manifest: &ModelManifest| {
                manifest
                    .files
                    .iter()
                    .filter(|file| file.component != ModelComponent::Transformer)
                    .map(|file| {
                        (
                            file.hf_repo.clone(),
                            file.hf_filename.clone(),
                            file.size_bytes,
                            file.sha256,
                            storage_path(manifest, file),
                        )
                    })
                    .collect::<std::collections::BTreeSet<_>>()
            };
            assert_eq!(non_transformer(int8), non_transformer(nvfp4));

            // The transformers are the only difference, and they never
            // collide on disk.
            let transformer = |manifest: &ModelManifest| {
                manifest
                    .files
                    .iter()
                    .find(|file| file.component == ModelComponent::Transformer)
                    .map(|file| (file.hf_repo.clone(), storage_path(manifest, file)))
                    .expect("every compact manifest pins one transformer")
            };
            let (int8_repo, int8_path) = transformer(int8);
            let (nvfp4_repo, nvfp4_path) = transformer(nvfp4);
            assert_eq!(int8_repo, COMFY_REPO);
            assert_eq!(nvfp4_repo, NVFP4_REPO);
            assert_ne!(int8_path, nvfp4_path);

            // Turbo tiers collapse onto the INT8 base directory; an NVFP4
            // base must keep its own.
            assert_eq!(storage_identity(nvfp4_name), nvfp4_name);
        }
    }

    #[test]
    fn pruned_nvfp4_transformer_pins_its_published_bytes() {
        for (name, filename, size, sha) in [
            (
                FL2VA_COMFY_NVFP4,
                "MiniMax_H3_FL2VA_pruned_nvfp4.safetensors",
                12_528_636_865_u64,
                "6ab7f0c48141e7919b32f925ca3def22e06a6aebeb9e0b6f5a0be0fe8409976f",
            ),
            (
                REF2VA_COMFY_NVFP4,
                "MiniMax_H3_Ref2VA_pruned_nvfp4.safetensors",
                12_528_636_866,
                "3e1be702c95bc057c05a7d1867e8aeea33073dcf5743835f2f27f06a2f34c596",
            ),
        ] {
            let manifest = find_manifest(name).unwrap();
            let transformer = manifest
                .files
                .iter()
                .find(|file| file.component == ModelComponent::Transformer)
                .unwrap();
            assert_eq!(transformer.hf_repo, NVFP4_REPO);
            assert_eq!(transformer.hf_filename, filename);
            // The size and digest cover the appended `L2P_bypass` marker
            // documented on `NVFP4_REPO`; a re-upload without it is a
            // different artifact and must be re-pinned.
            assert_eq!(transformer.size_bytes, size);
            assert_eq!(transformer.sha256, Some(sha));
            assert_eq!(file_revision(NVFP4_REPO, filename), Some(NVFP4_REVISION));
        }
    }

    #[test]
    fn artifact_dtypes_are_exact_per_concrete_layout() {
        let assert_dtype = |manifest_name: &str, component: ModelComponent, expected: &str| {
            let manifest = find_manifest(manifest_name).unwrap();
            let matches = manifest
                .files
                .iter()
                .filter(|file| file.component == component)
                .map(|file| artifact_contract(manifest, file).unwrap().dtype)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(matches, std::collections::BTreeSet::from([expected]));
        };

        for manifest in [FL2VA_OFFICIAL, REF2VA_OFFICIAL] {
            assert_dtype(manifest, ModelComponent::TransformerShard, "bf16");
            assert_dtype(manifest, ModelComponent::TextEncoder, "bf16");
            assert_dtype(manifest, ModelComponent::Vae, "fp32");
            assert_dtype(manifest, ModelComponent::AudioVae, "fp32");
        }
        for manifest in [FL2VA_COMFY, REF2VA_COMFY] {
            assert_dtype(manifest, ModelComponent::Transformer, "int8-convrot-pruned");
            assert_dtype(manifest, ModelComponent::TextEncoder, "nvfp4-awq");
            assert_dtype(manifest, ModelComponent::Vae, "fp16");
            assert_dtype(manifest, ModelComponent::AudioVae, "fp32");
        }
        // Only the transformer differs from the INT8 stack. The conditioner
        // and both VAEs are the same Comfy-Org artifacts at the same dtypes,
        // which is what lets the two layouts share one on-disk graph.
        for manifest in [FL2VA_COMFY_NVFP4, REF2VA_COMFY_NVFP4] {
            assert_dtype(manifest, ModelComponent::Transformer, "nvfp4-pruned");
            assert_dtype(manifest, ModelComponent::TextEncoder, "nvfp4-awq");
            assert_dtype(manifest, ModelComponent::Vae, "fp16");
            assert_dtype(manifest, ModelComponent::AudioVae, "fp32");
        }
    }

    #[test]
    fn request_wire_round_trip_preserves_contract_then_reviewed_model_is_admitted() {
        let mut req = request();
        req.output_format = None;
        req.normalise_output_format(Some(FAMILY));
        assert_eq!(req.output_format, Some(OutputFormat::Mp4));

        let wire = serde_json::to_value(&req).unwrap();
        let parsed: GenerateRequest = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(serde_json::to_value(&parsed).unwrap(), wire);
        assert_eq!(
            validate_request_contract(&parsed, Task::Fl2va).unwrap(),
            Mode::TextToAudioVideo
        );

        // Admission still consults the activation authority, which since
        // #1276 answers for this build rather than for the family: a binary
        // compiled without the engine refuses the reviewed compact model it
        // happily downloads.
        let admitted = crate::validate_generate_request_with_family(&parsed, Some(FAMILY));
        if engine_is_built() {
            admitted.unwrap();
        } else {
            let message = admitted.unwrap_err();
            assert!(
                message.contains(crate::MINIMAX_H3_RUNTIME_UNAVAILABLE),
                "{message}"
            );
            assert!(
                message.contains(RuntimeUnavailableReason::EngineNotBuilt.message()),
                "{message}"
            );
        }
    }

    #[test]
    fn rust_contract_pins_match_the_revision_locked_conformance_authority() {
        let conformance: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/minimax_h3/conformance-manifest.json"
        )))
        .unwrap();
        let source_revision = |id: &str| {
            conformance["sources"]
                .as_array()
                .unwrap()
                .iter()
                .find(|source| source["id"] == id)
                .and_then(|source| source["revision"].as_str())
                .unwrap()
        };
        assert_eq!(source_revision("minimax-official-model"), OFFICIAL_REVISION);
        assert_eq!(source_revision("comfy-checkpoints"), COMFY_REVISION);
        assert_eq!(
            source_revision("minimax-official-code"),
            OFFICIAL_IMPLEMENTATION_REVISION
        );
        assert_eq!(source_revision("comfyui"), COMFY_IMPLEMENTATION_REVISION);
        assert_eq!(source_revision("diffusers"), DIFFUSERS_REFERENCE_REVISION);
        assert_eq!(source_revision("nvfp4-checkpoints"), NVFP4_REVISION);
        assert_eq!(
            source_revision("lightx2v-turbo-adapters"),
            LIGHTX2V_REVISION
        );
        assert_eq!(
            source_revision("drbaph-resized-loras"),
            DRBAPH_TURBO_LORA_REVISION
        );
        assert_eq!(
            conformance["numerical_authority"]["precision"],
            "official-bf16-fp32-mixed"
        );
        assert_eq!(conformance["day_zero_contract"]["fps"], FIXED_FPS);
        assert_eq!(conformance["day_zero_contract"]["frame_step"], FRAME_STEP);
        assert_eq!(
            conformance["day_zero_contract"]["frame_offset"],
            FRAME_OFFSET
        );
        assert_eq!(
            conformance["day_zero_contract"]["nominal_15_second_frames"],
            MAX_FRAMES
        );
        let synthetic: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/minimax_h3/synthetic-v1.json"
        )))
        .unwrap();
        let noise_order = synthetic["noise_allocation_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|draw| draw["domain"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(noise_order, NOISE_STREAMS);

        let pin = |id: &str| {
            conformance["component_indexes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["id"] == id)
                .unwrap()
        };
        assert_eq!(pin("official-license")["sha256"], LICENSE_SHA256);
        for (id, manifest_name, path) in [
            ("official-model-index", FL2VA_OFFICIAL, "model_index.json"),
            (
                "official-modular-index",
                FL2VA_OFFICIAL,
                "modular_model_index.json",
            ),
            (
                "official-fl2va-transformer-index",
                FL2VA_OFFICIAL,
                "transformer/diffusion_pytorch_model.safetensors.index.json",
            ),
            (
                "official-ref2va-transformer-index",
                REF2VA_OFFICIAL,
                "transformer_ref/diffusion_pytorch_model.safetensors.index.json",
            ),
            (
                "official-text-encoder-index",
                FL2VA_OFFICIAL,
                "text_encoder/model.safetensors.index.json",
            ),
            (
                "official-video-vae-index",
                FL2VA_OFFICIAL,
                "vae/diffusion_pytorch_model.safetensors.index.json",
            ),
            (
                "official-audio-vae-config",
                FL2VA_OFFICIAL,
                "audio_vae/config.json",
            ),
        ] {
            let manifest = find_manifest(manifest_name).unwrap();
            let artifact = manifest
                .files
                .iter()
                .find(|file| file.hf_filename == path)
                .unwrap_or_else(|| panic!("missing {path} in {manifest_name}"));
            assert_eq!(pin(id)["relative_path"], path);
            assert_eq!(pin(id)["sha256"].as_str(), artifact.sha256);
        }
    }
}
