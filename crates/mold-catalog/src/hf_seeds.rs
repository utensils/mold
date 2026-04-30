//! Curated HF foundation repos used as the starting point for the HF
//! stage's `base_model:` walk. Adding a new family means: declare it in
//! `families.rs`, then add at least one seed here.

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
        QwenImage => &["Qwen/Qwen-Image"],
        Wuerstchen => &["warp-ai/wuerstchen"],
    }
}
