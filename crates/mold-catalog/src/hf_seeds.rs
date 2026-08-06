//! Curated HF foundation repos. Used by `family_from_hf` to mark these
//! repos as `FamilyRole::Foundation`; everything else gets `Finetune`.
//! Adding a new family means: declare it in `families.rs`, then add at
//! least one seed here.

use crate::families::Family;

pub fn seeds_for(family: Family) -> &'static [&'static str] {
    use Family::*;
    match family {
        Flux => &[
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-schnell",
        ],
        Flux2 => &[
            "black-forest-labs/FLUX.2-dev",
            "black-forest-labs/FLUX.2-Klein-9B",
        ],
        Sd15 => &[
            "runwayml/stable-diffusion-v1-5",
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
        ],
        Sdxl => &["stabilityai/stable-diffusion-xl-base-1.0"],
        ZImage => &["Tongyi-MAI/Z-Image-Turbo", "Tongyi-MAI/Z-Image-Base"],
        LtxVideo => &["Lightricks/LTX-Video"],
        Ltx2 => &["Lightricks/LTX-Video-2", "Lightricks/LTX-Video-2.3"],
        // Upstream's own releases plus the two repack lines mold's manifests
        // actually pull from — a user searching "wan" should see those ranked
        // as foundations rather than buried under community fine-tunes.
        Wan => &[
            "Wan-AI/Wan2.1-T2V-1.3B",
            "Wan-AI/Wan2.1-T2V-14B",
            "Wan-AI/Wan2.2-TI2V-5B",
            "Wan-AI/Wan2.2-T2V-A14B",
            "Wan-AI/Wan2.2-I2V-A14B",
            "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            "QuantStack/Wan2.2-T2V-A14B-GGUF",
            "QuantStack/Wan2.2-I2V-A14B-GGUF",
        ],
        QwenImage => &["Qwen/Qwen-Image"],
        Wuerstchen => &["warp-ai/wuerstchen"],
    }
}
