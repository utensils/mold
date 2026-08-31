//! Header-only safetensors probes.
//!
//! These read just the 8-byte length prefix + JSON header of a
//! `.safetensors` file — never the tensor data — so they can inspect
//! multi-GB checkpoints cheaply. Pure std + serde_json, no candle, so
//! they live in `mold-core` and are shared by the catalog resolver
//! (`mold-catalog`) and the inference loaders (`mold-inference`).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde_json::Value;

/// Header-only view of a safetensors file. Metadata values follow the
/// Lightricks convention: JSON strings are decoded into JSON values while
/// ordinary strings remain strings.
#[derive(Debug, Clone, PartialEq)]
pub struct SafetensorsHeader {
    pub metadata: BTreeMap<String, Value>,
    pub tensor_names: Vec<String>,
    /// Declared shape per tensor, in header order.
    ///
    /// Present so a loader can infer a checkpoint's ARCHITECTURE from the file
    /// rather than from its filename — Hunyuan3D's DiT geometry comes entirely
    /// out of two `Linear` shapes and a block-prefix count, exactly as
    /// `comfy/model_detection.py` does it. Still header-only: the shapes are
    /// declared in the JSON, so nothing reads a tensor byte.
    pub tensor_shapes: BTreeMap<String, Vec<usize>>,
}

/// Read a safetensors header without touching tensor payload bytes.
pub fn read_safetensors_header(path: &Path) -> std::io::Result<SafetensorsHeader> {
    let mut file = File::open(path)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;
    let mut header: BTreeMap<String, Value> = serde_json::from_slice(&header_buf).map_err(|e| {
        std::io::Error::other(format!(
            "parse safetensors header at {}: {e}",
            path.display()
        ))
    })?;
    let metadata = match header.remove("__metadata__") {
        Some(Value::Object(values)) => values
            .into_iter()
            .map(|(key, value)| {
                let parsed = match value {
                    Value::String(raw) => serde_json::from_str(&raw).unwrap_or(Value::String(raw)),
                    other => other,
                };
                (key, parsed)
            })
            .collect(),
        Some(_) | None => BTreeMap::new(),
    };
    let mut tensor_names = Vec::with_capacity(header.len());
    let mut tensor_shapes = BTreeMap::new();
    for (name, value) in header {
        if let Some(shape) = value.get("shape").and_then(Value::as_array) {
            let dims: Vec<usize> = shape
                .iter()
                .filter_map(Value::as_u64)
                .map(|dim| dim as usize)
                .collect();
            tensor_shapes.insert(name.clone(), dims);
        }
        tensor_names.push(name);
    }
    Ok(SafetensorsHeader {
        metadata,
        tensor_names,
        tensor_shapes,
    })
}

/// Peek the safetensors header (8-byte length prefix + JSON, no tensor data
/// read) to determine whether a single-file checkpoint bundles its VAE.
/// Returns `true` when any VAE-encoder marker key is present.
///
/// The keys we look for are inclusive of the conventions used by Civitai
/// converters and BFL's own export format:
/// - `encoder.conv_in.*`           — diffusers-style root
/// - `first_stage_model.encoder.*` — A1111/ComfyUI-style root
/// - `vae.encoder.*`               — some pruning tools' convention
///
/// Civitai's FLUX and LTX-2 fine-tune conventions are inconsistent — some bundle the VAE
/// (`*_full.safetensors`), some are transformer-only (`*Unet.safetensors` /
/// `*_diffusion.safetensors`). Without this probe the catalog bridge would
/// unconditionally point `cfg.vae` at the primary checkpoint and the engine
/// would crash with `cannot find tensor encoder.conv_in.weight` on every
/// transformer-only fine-tune. LTX-2 uses the `vae.encoder.*` form.
///
/// Returns `Err` only on bona fide I/O / parse failures — never panics.
pub fn single_file_bundles_vae(path: &Path) -> std::io::Result<bool> {
    let header = read_safetensors_header(path)?;
    Ok(header.tensor_names.iter().any(|key| {
        key.starts_with("encoder.conv_in")
            || key.starts_with("first_stage_model.encoder.")
            || key.starts_with("vae.encoder.")
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_safetensors(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "mold-probe-{}-{}-{}.safetensors",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        path
    }

    /// Write a synthetic safetensors at `path` whose header advertises the
    /// given keys (each as a 1-element F32 tensor sharing the same 4-byte
    /// zero blob). Sufficient for the header-peek probe — no dependency on
    /// the `safetensors` crate.
    fn write_safetensors_with_keys(path: &Path, keys: &[&str]) {
        use std::io::Write;
        let mut header = serde_json::Map::new();
        for key in keys {
            header.insert(
                (*key).to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                }),
            );
        }
        let header_json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = File::create(path).expect("create fixture");
        f.write_all(&(header_json.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(&header_json).unwrap();
        f.write_all(&[0u8; 4]).unwrap(); // F32 zero — shared by every key
    }

    #[test]
    fn true_for_bundled_diffusers_prefix() {
        // Diffusers-style root: `encoder.conv_in.weight` is the canonical
        // VAE marker. Anything starting with `encoder.conv_in` (incl.
        // `.bias`) trips the probe.
        let path = temp_safetensors("flux-vae-diffusers");
        write_safetensors_with_keys(
            &path,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "encoder.conv_in.weight",
                "decoder.conv_out.weight",
            ],
        );

        let bundled = single_file_bundles_vae(&path).expect("probe must not error");
        assert!(
            bundled,
            "diffusers-style `encoder.conv_in.weight` must mark the file as VAE-bundled"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn false_for_unet_only() {
        // The cv:994561 case: all `double_blocks.*` / `single_blocks.*` /
        // `img_in.*`, zero VAE encoder markers.
        let path = temp_safetensors("flux-unet-only");
        write_safetensors_with_keys(
            &path,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "double_blocks.0.img_attn.norm.query_norm.scale",
                "single_blocks.0.linear1.weight",
                "img_in.weight",
                "txt_in.weight",
                "final_layer.linear.weight",
            ],
        );

        let bundled = single_file_bundles_vae(&path).expect("probe must not error");
        assert!(
            !bundled,
            "transformer-only checkpoint (no encoder.conv_in / first_stage_model / vae prefix) \
             must NOT be marked as VAE-bundled"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn handles_a1111_prefix() {
        // A1111/ComfyUI-style: VAE keys live under `first_stage_model.encoder.*`.
        let path = temp_safetensors("flux-vae-a1111");
        write_safetensors_with_keys(
            &path,
            &[
                "model.diffusion_model.double_blocks.0.img_attn.proj.weight",
                "first_stage_model.encoder.conv_in.weight",
                "first_stage_model.decoder.conv_out.weight",
            ],
        );

        let bundled = single_file_bundles_vae(&path).expect("probe must not error");
        assert!(
            bundled,
            "A1111 `first_stage_model.encoder.*` prefix must mark the file as VAE-bundled"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn handles_pruner_prefix() {
        // Some pruning tools strip `first_stage_model` and emit `vae.encoder.*`.
        let path = temp_safetensors("flux-vae-pruned");
        write_safetensors_with_keys(
            &path,
            &[
                "double_blocks.0.img_attn.proj.weight",
                "vae.encoder.conv_in.weight",
            ],
        );

        let bundled = single_file_bundles_vae(&path).expect("probe must not error");
        assert!(
            bundled,
            "pruner-style `vae.encoder.*` prefix must mark the file as VAE-bundled"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn io_error_on_missing_file() {
        // Bona fide I/O failure surfaces as Err; the caller translates this
        // into a clear message rather than panicking or silently treating
        // the file as transformer-only.
        let missing = std::env::temp_dir().join("mold-probe-flux-vae-missing.safetensors");
        let _ = std::fs::remove_file(&missing); // ensure it doesn't exist
        let err = single_file_bundles_vae(&missing).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }
}
