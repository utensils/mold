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
