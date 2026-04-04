//! Architecture auto-detection from safetensors state dict keys.
//!
//! Follows the Spandrel pattern: infer model architecture and hyperparameters
//! from the set of tensor names in the weights file.

use anyhow::{bail, Result};

/// Detected upscaler architecture with inferred hyperparameters.
#[derive(Debug, Clone)]
pub enum UpscalerArch {
    /// Compact Real-ESRGAN: linear Conv2d + PReLU chain with pixel_shuffle.
    SRVGGNetCompact {
        num_feat: usize,
        num_conv: usize,
        scale: u32,
    },
    /// Full Real-ESRGAN: Residual-in-Residual Dense Blocks with upsample + conv.
    RRDBNet {
        num_feat: usize,
        num_grow_ch: usize,
        num_block: usize,
        scale: u32,
    },
}

/// Detect upscaler architecture from safetensors tensor names.
pub fn detect_architecture(tensor_names: &[&str]) -> Result<UpscalerArch> {
    let has_conv_first = tensor_names.contains(&"conv_first.weight");
    let has_body_0 = tensor_names.contains(&"body.0.weight");

    if has_conv_first {
        detect_rrdbnet(tensor_names)
    } else if has_body_0 {
        detect_srvggnet(tensor_names)
    } else {
        bail!(
            "unknown upscaler architecture: no conv_first.weight or body.0.weight found in state dict"
        );
    }
}

fn detect_srvggnet(tensor_names: &[&str]) -> Result<UpscalerArch> {
    // Body layers alternate: conv (weight+bias), prelu (weight).
    // Count the highest body.N index to determine num_conv.
    let max_body_idx = tensor_names
        .iter()
        .filter_map(|n| {
            n.strip_prefix("body.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|idx| idx.parse::<usize>().ok())
        })
        .max()
        .unwrap_or(0);

    // body layout: [conv0, prelu0, conv1, prelu1, ..., prelu_last, conv_last]
    // num_conv = (max_body_idx + 1) / 2 - 1 for intermediate convs,
    // but simpler: count tensors named body.N.weight where N is even (conv layers).
    let num_conv_layers: usize = tensor_names
        .iter()
        .filter(|n| n.ends_with(".weight"))
        .filter_map(|n| {
            n.strip_prefix("body.")
                .and_then(|rest| rest.strip_suffix(".weight"))
                .and_then(|idx| idx.parse::<usize>().ok())
        })
        .filter(|idx| idx % 2 == 0) // even indices are Conv2d layers
        .count();

    // Infer num_feat from body.0.weight shape: [num_feat, in_channels, k, k]
    // We can't read shapes from names alone, so use a reasonable default.
    // The actual shape will be read during model construction.
    let num_feat = 64; // default for realesr-general-x4v3

    // Infer scale from the last conv layer's output channels:
    // out_channels = 3 * scale^2 for pixel_shuffle.
    // Without shape info, default to 4x (most common).
    let scale = 4u32;

    // num_conv = number of intermediate conv layers (excluding first and last body convs)
    // Total body conv layers = num_conv_layers, intermediate = num_conv_layers - 2
    let num_conv = num_conv_layers.saturating_sub(2);

    // Adjust num_feat for smaller models (animevideov3 uses 64, general uses 64)
    // Use max_body_idx to distinguish:
    // - general-x4v3: 32 body layers (16 conv + 16 prelu) -> max_body_idx=31
    // - animevideov3: fewer body layers
    let _ = max_body_idx; // used for counting above

    Ok(UpscalerArch::SRVGGNetCompact {
        num_feat,
        num_conv,
        scale,
    })
}

fn detect_rrdbnet(tensor_names: &[&str]) -> Result<UpscalerArch> {
    // Count RRDB blocks by finding the highest body.N.rdb1.conv1.weight index.
    let num_block = tensor_names
        .iter()
        .filter_map(|n| {
            n.strip_prefix("body.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|idx| idx.parse::<usize>().ok())
        })
        .filter(|idx| {
            // Only count RRDB block indices, not the final conv_body
            tensor_names
                .iter()
                .any(|n| n.starts_with(&format!("body.{idx}.rdb1.conv1.weight")))
        })
        .max()
        .map(|max_idx| max_idx + 1) // 0-indexed
        .unwrap_or(23); // default for x4plus

    // Detect scale from upsampling conv layers
    let has_conv_up1 = tensor_names.contains(&"conv_up1.weight");
    let has_conv_up2 = tensor_names.contains(&"conv_up2.weight");
    let scale = if has_conv_up2 {
        4
    } else if has_conv_up1 {
        2
    } else {
        4
    };

    // Default hyperparameters (standard Real-ESRGAN)
    let num_feat = 64;
    let num_grow_ch = 32;

    Ok(UpscalerArch::RRDBNet {
        num_feat,
        num_grow_ch,
        num_block,
        scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_rrdbnet_from_keys() {
        let keys = vec![
            "conv_first.weight",
            "conv_first.bias",
            "body.0.rdb1.conv1.weight",
            "body.0.rdb1.conv1.bias",
            "body.0.rdb1.conv2.weight",
            "body.22.rdb3.conv5.weight",
            "body.23.weight", // conv_body
            "conv_up1.weight",
            "conv_up2.weight",
            "conv_hr.weight",
            "conv_last.weight",
        ];
        let arch = detect_architecture(&keys).unwrap();
        match arch {
            UpscalerArch::RRDBNet {
                num_block, scale, ..
            } => {
                assert_eq!(num_block, 23);
                assert_eq!(scale, 4);
            }
            _ => panic!("expected RRDBNet"),
        }
    }

    #[test]
    fn detect_rrdbnet_x2_from_keys() {
        let keys = vec![
            "conv_first.weight",
            "body.0.rdb1.conv1.weight",
            "body.22.rdb3.conv5.weight",
            "body.23.weight",
            "conv_up1.weight",
            "conv_hr.weight",
            "conv_last.weight",
        ];
        let arch = detect_architecture(&keys).unwrap();
        match arch {
            UpscalerArch::RRDBNet { scale, .. } => assert_eq!(scale, 2),
            _ => panic!("expected RRDBNet"),
        }
    }

    #[test]
    fn detect_rrdbnet_anime_6b() {
        let keys = vec![
            "conv_first.weight",
            "body.0.rdb1.conv1.weight",
            "body.5.rdb3.conv5.weight",
            "body.6.weight", // conv_body
            "conv_up1.weight",
            "conv_up2.weight",
            "conv_hr.weight",
            "conv_last.weight",
        ];
        let arch = detect_architecture(&keys).unwrap();
        match arch {
            UpscalerArch::RRDBNet {
                num_block, scale, ..
            } => {
                assert_eq!(num_block, 6);
                assert_eq!(scale, 4);
            }
            _ => panic!("expected RRDBNet"),
        }
    }

    #[test]
    fn detect_srvggnet_from_keys() {
        let keys = vec![
            "body.0.weight",
            "body.0.bias",
            "body.1.weight", // prelu
            "body.2.weight",
            "body.2.bias",
            "body.3.weight",  // prelu
            "body.30.weight", // last conv
            "body.30.bias",
            "body.31.weight", // last prelu (if present)
        ];
        let arch = detect_architecture(&keys).unwrap();
        assert!(matches!(arch, UpscalerArch::SRVGGNetCompact { .. }));
    }

    #[test]
    fn unknown_architecture_errors() {
        let keys = vec!["something_unknown.weight"];
        assert!(detect_architecture(&keys).is_err());
    }
}
