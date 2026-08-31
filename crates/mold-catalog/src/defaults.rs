#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CatalogRuntimeDefaults {
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f64,
    pub is_schnell: Option<bool>,
    /// Default video frame count; None for image families.
    pub frames: Option<u32>,
    /// Default video FPS; None for image families.
    pub fps: Option<u32>,
}

pub fn runtime_defaults_for_family(
    family: &str,
    sub_family: Option<&str>,
) -> CatalogRuntimeDefaults {
    match family {
        "flux" => match sub_family {
            Some("flux1-s") => CatalogRuntimeDefaults {
                width: 1024,
                height: 1024,
                steps: 4,
                guidance: 0.0,
                is_schnell: Some(true),
                frames: None,
                fps: None,
            },
            _ => CatalogRuntimeDefaults {
                width: 1024,
                height: 1024,
                steps: 25,
                guidance: 3.5,
                is_schnell: Some(false),
                frames: None,
                fps: None,
            },
        },
        "flux2" => CatalogRuntimeDefaults {
            width: 1024,
            height: 1024,
            steps: 4,
            guidance: 1.0,
            is_schnell: None,
            frames: None,
            fps: None,
        },
        "z-image" => CatalogRuntimeDefaults {
            width: 1024,
            height: 1024,
            steps: 9,
            guidance: 0.0,
            is_schnell: None,
            frames: None,
            fps: None,
        },
        "ltx2" => CatalogRuntimeDefaults {
            width: 1216,
            height: 704,
            steps: 8,
            guidance: 3.0,
            is_schnell: None,
            frames: Some(97),
            fps: Some(24),
        },
        "ltx-video" => CatalogRuntimeDefaults {
            width: 1216,
            height: 704,
            steps: 30,
            guidance: 8.0,
            is_schnell: None,
            frames: Some(25),
            fps: Some(30),
        },
        // Wan splits on VAE generation, not on version number. TI2V-5B pairs
        // with the 2.2 VAE and renders 121 frames at 24 fps; everything else
        // in the family — 2.1 at either size and both 2.2 A14B experts — pairs
        // with the 2.1 VAE at 81 frames and 16 fps. Frame counts must stay on
        // the family's 4n+1 grid or the request is rejected before it renders.
        "wan" => match sub_family {
            Some("wan22-ti2v-5b") => CatalogRuntimeDefaults {
                width: 1280,
                height: 704,
                steps: 20,
                guidance: 5.0,
                is_schnell: None,
                frames: Some(121),
                fps: Some(24),
            },
            // The 1.3B has its own proven recipe — ComfyUI's Wan 2.1 template,
            // which is what the shipped manifest uses. Without this arm a
            // catalog-installed 1.3B renders at 20 steps / guidance 3.5 while
            // the identical model installed by manifest renders at 30 / 6.0,
            // so the same checkpoint produces materially different quality
            // depending on how it was pulled.
            // The 2.1 14B renders at the same resolution with the same
            // template as the 1.3B, and the shipped manifests use that recipe,
            // so a catalog-installed 14B finetune must not land on the generic
            // 20 / 3.5 fallback while the manifest tier renders at 30 / 6.0.
            Some("wan21-t2v-1.3b") | Some("wan21-t2v-14b") => CatalogRuntimeDefaults {
                width: 832,
                height: 480,
                steps: 30,
                guidance: 6.0,
                is_schnell: None,
                frames: Some(81),
                fps: Some(16),
            },
            _ => CatalogRuntimeDefaults {
                width: 832,
                height: 480,
                steps: 20,
                guidance: 3.5,
                is_schnell: None,
                frames: Some(81),
                fps: Some(16),
            },
        },
        "minimax-h3" | "minimax_h3" | "minimaxh3" => CatalogRuntimeDefaults {
            width: mold_core::minimax_h3::DEFAULT_WIDTH,
            height: mold_core::minimax_h3::DEFAULT_HEIGHT,
            steps: mold_core::minimax_h3::DEFAULT_STEPS,
            guidance: 0.0,
            is_schnell: None,
            frames: Some(mold_core::minimax_h3::DEFAULT_COMPACT_FRAMES),
            fps: Some(mold_core::minimax_h3::FIXED_FPS),
        },
        "sdxl" => CatalogRuntimeDefaults {
            width: 1024,
            height: 1024,
            steps: 25,
            guidance: 7.5,
            is_schnell: None,
            frames: None,
            fps: None,
        },
        "sd3" | "sd3.5" => CatalogRuntimeDefaults {
            width: 1024,
            height: 1024,
            steps: 28,
            guidance: 4.0,
            is_schnell: None,
            frames: None,
            fps: None,
        },
        "sd15" => CatalogRuntimeDefaults {
            width: 512,
            height: 512,
            steps: 25,
            guidance: 7.5,
            is_schnell: None,
            frames: None,
            fps: None,
        },
        "qwen-image" | "qwen-image-edit" => CatalogRuntimeDefaults {
            width: 1328,
            height: 1328,
            steps: 50,
            guidance: 4.0,
            is_schnell: None,
            frames: None,
            fps: None,
        },
        // Image-to-3D. `width`/`height` record the image-CONDITIONING size the
        // checkpoints letterbox their source to (`image_processor.size` in the
        // upstream config), not an output canvas: a mesh has none. `frames`
        // and `fps` must stay `None` — `families_roundtrip` asserts they agree
        // with `Family::is_video()`, and a mesh is not video. The mini tier
        // conditions at 1022; the 1.1B tiers at 512, which is what the
        // family-level default names.
        "hunyuan3d" | "hunyuan-3d" => CatalogRuntimeDefaults {
            width: 512,
            height: 512,
            steps: 5,
            guidance: 5.0,
            is_schnell: None,
            frames: None,
            fps: None,
        },
        _ => CatalogRuntimeDefaults {
            width: 1024,
            height: 1024,
            steps: 20,
            guidance: 3.5,
            is_schnell: None,
            frames: None,
            fps: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Video families must carry frame defaults matching their manifest
    /// counterparts (ltx2 → 97@24, ltx-video → 25@30); image families none.
    #[test]
    fn video_families_carry_frame_defaults() {
        let ltx2 = runtime_defaults_for_family("ltx2", None);
        assert_eq!(ltx2.frames, Some(97));
        assert_eq!(ltx2.fps, Some(24));

        let ltx_video = runtime_defaults_for_family("ltx-video", None);
        assert_eq!(ltx_video.frames, Some(25));
        assert_eq!(ltx_video.fps, Some(30));

        let h3 = runtime_defaults_for_family("minimax-h3", None);
        assert_eq!(
            h3.frames,
            Some(mold_core::minimax_h3::DEFAULT_COMPACT_FRAMES)
        );
        assert_eq!(h3.fps, Some(mold_core::minimax_h3::FIXED_FPS));
        assert_eq!((h3.width, h3.height), (1344, 768));
        assert_eq!(h3.guidance, 0.0);

        let flux = runtime_defaults_for_family("flux", None);
        assert_eq!(flux.frames, None);
        assert_eq!(flux.fps, None);

        let sd3 = runtime_defaults_for_family("sd3", None);
        assert_eq!((sd3.width, sd3.height), (1024, 1024));
        assert_eq!((sd3.steps, sd3.guidance), (28, 4.0));
        assert_eq!((sd3.frames, sd3.fps), (None, None));
    }

    /// Wan's timing splits on VAE generation, and the sub-family name is the
    /// only signal a catalog row carries. Getting this backwards gives a 5B
    /// checkpoint the 14B's 16 fps, or hands every A14B row 121 frames at 24
    /// fps — both render, at the wrong length and the wrong speed.
    #[test]
    fn wan_timing_follows_the_vae_generation_not_the_version_number() {
        let five_b = runtime_defaults_for_family("wan", Some("wan22-ti2v-5b"));
        assert_eq!((five_b.frames, five_b.fps), (Some(121), Some(24)));

        // Both 2.2 A14B experts share the 2.1 VAE and therefore the 2.1
        // timing, despite the 2.2 in their name.
        for sub in [
            Some("wan21-t2v-1.3b"),
            Some("wan21-t2v-14b"),
            Some("wan22-t2v-a14b"),
            Some("wan22-i2v-a14b"),
            None,
        ] {
            let defaults = runtime_defaults_for_family("wan", sub);
            assert_eq!(
                (defaults.frames, defaults.fps),
                (Some(81), Some(16)),
                "{sub:?}"
            );
        }

        // A catalog-installed checkpoint must render like the same model
        // installed by manifest. The 1.3B's proven recipe is 30 steps at
        // guidance 6.0 (ComfyUI's Wan 2.1 template, which the manifest ships);
        // falling through to the family arm would give it 20 / 3.5.
        let one_three_b = runtime_defaults_for_family("wan", Some("wan21-t2v-1.3b"));
        assert_eq!((one_three_b.steps, one_three_b.guidance), (30, 6.0));
        assert_eq!((one_three_b.width, one_three_b.height), (832, 480));
        assert_eq!((one_three_b.frames, one_three_b.fps), (Some(81), Some(16)));

        // The 2.1 14B now has shipped tiers behind it (#797) and renders at
        // the same 480p template as the 1.3B, so it takes that recipe rather
        // than the generic family fallback.
        let fourteen_b = runtime_defaults_for_family("wan", Some("wan21-t2v-14b"));
        assert_eq!((fourteen_b.steps, fourteen_b.guidance), (30, 6.0));
        assert_eq!((fourteen_b.width, fourteen_b.height), (832, 480));

        // The family fallback — an unrecognized wan sub-family — is unchanged.
        let fallback = runtime_defaults_for_family("wan", Some("wan-something-new"));
        assert_eq!((fallback.steps, fallback.guidance), (20, 3.5));

        // Every default sits on the family's 4n+1 frame grid.
        for sub in [Some("wan22-ti2v-5b"), None] {
            let frames = runtime_defaults_for_family("wan", sub).frames.unwrap();
            assert_eq!((frames - 1) % 4, 0, "{sub:?} must be 4n+1, got {frames}");
        }
    }

    #[test]
    fn flux_dev_and_schnell_subfamilies_have_distinct_defaults() {
        assert_eq!(
            runtime_defaults_for_family("flux", Some("flux1-d")),
            CatalogRuntimeDefaults {
                width: 1024,
                height: 1024,
                steps: 25,
                guidance: 3.5,
                is_schnell: Some(false),
                frames: None,
                fps: None,
            }
        );
        assert_eq!(
            runtime_defaults_for_family("flux", Some("flux1-s")),
            CatalogRuntimeDefaults {
                width: 1024,
                height: 1024,
                steps: 4,
                guidance: 0.0,
                is_schnell: Some(true),
                frames: None,
                fps: None,
            }
        );
    }
}
