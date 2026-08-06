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
        "sdxl" => CatalogRuntimeDefaults {
            width: 1024,
            height: 1024,
            steps: 25,
            guidance: 7.5,
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

        let flux = runtime_defaults_for_family("flux", None);
        assert_eq!(flux.frames, None);
        assert_eq!(flux.fps, None);
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
