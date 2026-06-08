#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CatalogRuntimeDefaults {
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f64,
    pub is_schnell: Option<bool>,
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
            },
            _ => CatalogRuntimeDefaults {
                width: 1024,
                height: 1024,
                steps: 25,
                guidance: 3.5,
                is_schnell: Some(false),
            },
        },
        "flux2" => CatalogRuntimeDefaults {
            width: 1024,
            height: 1024,
            steps: 4,
            guidance: 1.0,
            is_schnell: None,
        },
        "z-image" => CatalogRuntimeDefaults {
            width: 1024,
            height: 1024,
            steps: 9,
            guidance: 0.0,
            is_schnell: None,
        },
        "ltx2" => CatalogRuntimeDefaults {
            width: 1216,
            height: 704,
            steps: 8,
            guidance: 3.0,
            is_schnell: None,
        },
        "ltx-video" => CatalogRuntimeDefaults {
            width: 1216,
            height: 704,
            steps: 30,
            guidance: 8.0,
            is_schnell: None,
        },
        "sdxl" => CatalogRuntimeDefaults {
            width: 1024,
            height: 1024,
            steps: 25,
            guidance: 7.5,
            is_schnell: None,
        },
        "sd15" => CatalogRuntimeDefaults {
            width: 512,
            height: 512,
            steps: 25,
            guidance: 7.5,
            is_schnell: None,
        },
        "qwen-image" | "qwen-image-edit" => CatalogRuntimeDefaults {
            width: 1328,
            height: 1328,
            steps: 50,
            guidance: 4.0,
            is_schnell: None,
        },
        _ => CatalogRuntimeDefaults {
            width: 1024,
            height: 1024,
            steps: 20,
            guidance: 3.5,
            is_schnell: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            }
        );
    }
}
