//! Qwen Image 2.1 LoRA key mapping and scale.
//!
//! Every adapter is applied at forward time (bypass), never merged: merging
//! into BF16 is lossy and the Viggle turbo distill specifies the unmerged
//! form (`W x + B (A x) · alpha/rank`). This module owns the two questions
//! that are this family's own:
//!
//! 1. **Which linear does a LoRA stem name?** [`map_key`] matches an EXACT
//!    module table rather than an n-gram converter: diffusers' Kohya converter
//!    (`lora_conversion_utils.py:2221-2283`) has no `modulation_1` or
//!    `txt_in_*` entries and splits `gate_layer` into `gate.layer`, all of
//!    which the flattened-table lookup here resolves. ComfyUI's fused
//!    `img_mlp.gate_up` lands on the two halves, GATE rows first
//!    (`comfy/lora.py:331-333`; diffusers `single_file_utils.py:4240-4243`).
//!    A Qwen-Image / 2512 adapter (`add_q_proj`, `txt_mlp`, `img_mod`) or a
//!    text-encoder adapter is refused by name rather than adapting nothing.
//! 2. **What is its scale?** `user_scale · alpha / rank`, with alpha from a
//!    `.alpha` tensor or, failing that, the safetensors `__metadata__`
//!    `lora_adapter_metadata` PEFT writes — whose keys the Viggle files prefix
//!    with `transformer.` (`"transformer.lora_alpha": 256`).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};

use super::transformer::QwenImage21TransformerConfig;

/// Where one LoRA layer lands: `up` rows `[start, start + len)` (all of them
/// when `None`) onto the whole output of `candle_key`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Qwen21LoraTarget {
    pub candle_key: String,
    pub up_rows: Option<(usize, usize)>,
}

impl Qwen21LoraTarget {
    fn whole(module: &str) -> Self {
        Self {
            candle_key: format!("{module}.weight"),
            up_rows: None,
        }
    }
}

/// Top-level linears a LoRA may adapt.
const TOP_LEVEL: [&str; 8] = [
    "img_in",
    "txt_in.in_layer",
    "txt_in.out_layer",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "modulation.1",
    "norm_out.linear",
    "proj_out",
];

/// Per-block linears a LoRA may adapt.
const BLOCK: [&str; 7] = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.proj",
    "img_mlp.gate_layer",
    "img_mlp.out",
];

/// Non-linear modules: an adapter on one is skipped with a warning.
const NON_LINEAR: [&str; 3] = ["attn.norm_q", "attn.norm_k", "txt_in.text_norm"];

/// Module names that only exist in the older Qwen-Image / 2512 architecture.
const QWEN_IMAGE_2512_MARKERS: [&str; 7] = [
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "txt_mlp",
    "img_mod",
    "txt_mod",
];

/// Prefixes a trainer may put in front of the module path.
const PREFIXES: [&str; 5] = [
    "base_model.model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "lycoris_",
];

/// Every adaptable module, dotted.
fn module_table(layers: usize) -> Vec<String> {
    let mut modules: Vec<String> = TOP_LEVEL.iter().map(|m| m.to_string()).collect();
    for index in 0..layers {
        for module in BLOCK {
            modules.push(format!("transformer_blocks.{index}.{module}"));
        }
    }
    modules
}

/// The flattened (Kohya) spelling of every adaptable module.
fn flattened_table(layers: usize) -> HashMap<String, String> {
    module_table(layers)
        .into_iter()
        .map(|module| (module.replace('.', "_"), module))
        .collect()
}

/// Map one LoRA layer stem (the key without its `lora_A`/`lora_down` suffix)
/// to the linears it adapts. An empty answer means "skip with a warning" (a
/// non-linear module); an error means the adapter belongs to another
/// architecture or component.
pub(crate) fn map_key(stem: &str) -> Result<Vec<Qwen21LoraTarget>> {
    map_key_with(stem, QwenImage21TransformerConfig::official().num_layers)
}

fn map_key_with(stem: &str, layers: usize) -> Result<Vec<Qwen21LoraTarget>> {
    if stem.starts_with("text_encoder.") || stem.starts_with("lora_te") {
        bail!(
            "Qwen Image 2.1 LoRA `{stem}` adapts the text encoder, which this engine does not support"
        );
    }
    if QWEN_IMAGE_2512_MARKERS
        .iter()
        .any(|marker| stem.contains(marker))
    {
        bail!(
            "LoRA `{stem}` is a Qwen-Image (2512/Edit) adapter; Qwen Image 2.1 is a different architecture"
        );
    }
    let mut key = stem;
    loop {
        match PREFIXES.iter().find_map(|prefix| key.strip_prefix(prefix)) {
            Some(rest) => key = rest,
            None => break,
        }
    }
    // Kohya: `lora_unet_<flattened>`.
    if let Some(flat) = key.strip_prefix("lora_unet_") {
        if let Some(fused) = flat.strip_suffix("_img_mlp_gate_up") {
            let block = match fused.strip_prefix("transformer_blocks_") {
                Some(index) => format!("transformer_blocks.{index}"),
                None => fused.to_string(),
            };
            return fused_gate_up(&block, layers);
        }
        if let Some(module) = flattened_table(layers).get(flat) {
            return Ok(vec![Qwen21LoraTarget::whole(module)]);
        }
        if NON_LINEAR
            .iter()
            .any(|module| flat.ends_with(&module.replace('.', "_")))
        {
            return Ok(Vec::new());
        }
        bail!("LoRA `{stem}` names no Qwen Image 2.1 module");
    }
    if let Some(block) = key.strip_suffix(".img_mlp.gate_up") {
        return fused_gate_up(block, layers);
    }
    if module_table(layers).iter().any(|module| module == key) {
        return Ok(vec![Qwen21LoraTarget::whole(key)]);
    }
    if NON_LINEAR.iter().any(|module| key.ends_with(module)) {
        return Ok(Vec::new());
    }
    bail!("LoRA `{stem}` names no Qwen Image 2.1 module")
}

/// ComfyUI's fused `gate_up`: rows `[0, mlp)` are the gate, `[mlp, 2·mlp)`
/// the up projection.
fn fused_gate_up(block: &str, layers: usize) -> Result<Vec<Qwen21LoraTarget>> {
    let index: usize = block
        .strip_prefix("transformer_blocks.")
        .and_then(|index| index.parse().ok())
        .filter(|index| *index < layers)
        .with_context(|| format!("fused gate_up LoRA on unknown block `{block}`"))?;
    let mlp = mlp_hidden_dim();
    Ok(vec![
        Qwen21LoraTarget {
            candle_key: format!("transformer_blocks.{index}.img_mlp.gate_layer.weight"),
            up_rows: Some((0, mlp)),
        },
        Qwen21LoraTarget {
            candle_key: format!("transformer_blocks.{index}.img_mlp.proj.weight"),
            up_rows: Some((mlp, mlp)),
        },
    ])
}

/// Width of the SwiGLU hidden layer (`inner · mlp_ratio`).
fn mlp_hidden_dim() -> usize {
    let cfg = QwenImage21TransformerConfig::official();
    cfg.num_attention_heads * cfg.attention_head_dim * cfg.mlp_ratio
}

/// Map every stem of an adapter, refusing one that adapts nothing.
pub(crate) fn map_adapter<'a>(
    stems: impl IntoIterator<Item = &'a str>,
) -> Result<Vec<(&'a str, Vec<Qwen21LoraTarget>)>> {
    let mut mapped = Vec::new();
    let mut skipped = 0usize;
    for stem in stems {
        let targets = map_key(stem)?;
        if targets.is_empty() {
            skipped += 1;
            tracing::warn!(
                stem,
                "Qwen Image 2.1 LoRA layer on a non-linear module skipped"
            );
        }
        mapped.push((stem, targets));
    }
    if mapped.iter().all(|(_, targets)| targets.is_empty()) {
        bail!("the LoRA adapts no Qwen Image 2.1 linear ({skipped} layer(s) skipped)");
    }
    Ok(mapped)
}

/// PEFT's scale metadata (`LoraConfig` as `peft` serializes it into
/// `lora_adapter_metadata`).
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct LoraMetadataScale {
    pub alpha: Option<f64>,
    pub rank: Option<f64>,
    pub alpha_pattern: Vec<(String, f64)>,
}

impl LoraMetadataScale {
    /// Parse the `lora_adapter_metadata` JSON. Keys may be bare (`lora_alpha`)
    /// or component-prefixed (`transformer.lora_alpha`, the Viggle files).
    pub(crate) fn parse(json: &str) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_str(json).context("lora_adapter_metadata is not JSON")?;
        let object = value
            .as_object()
            .context("lora_adapter_metadata is not an object")?;
        let field = |name: &str| {
            object
                .get(name)
                .or_else(|| object.get(&format!("transformer.{name}")))
        };
        let alpha_pattern = field("alpha_pattern")
            .and_then(|value| value.as_object())
            .map(|pattern| {
                pattern
                    .iter()
                    .filter_map(|(key, value)| value.as_f64().map(|alpha| (key.clone(), alpha)))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            alpha: field("lora_alpha").and_then(serde_json::Value::as_f64),
            rank: field("r").and_then(serde_json::Value::as_f64),
            alpha_pattern,
        })
    }

    /// Read it from a safetensors file's `__metadata__`, if present. Only the
    /// header is read, never the tensors.
    pub(crate) fn read(path: &Path) -> Result<Option<Self>> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("failed to open LoRA {}", path.display()))?;
        let mut length = [0u8; 8];
        file.read_exact(&mut length)
            .with_context(|| format!("failed to read LoRA header {}", path.display()))?;
        let length = u64::from_le_bytes(length);
        if length > 100 << 20 {
            bail!("LoRA header of {} claims {length} bytes", path.display());
        }
        let mut header = vec![0u8; length as usize];
        file.read_exact(&mut header)
            .with_context(|| format!("failed to read LoRA header {}", path.display()))?;
        let header: serde_json::Value = serde_json::from_slice(&header)
            .with_context(|| format!("LoRA header of {} is not JSON", path.display()))?;
        header
            .get("__metadata__")
            .and_then(|metadata| metadata.get("lora_adapter_metadata"))
            .and_then(serde_json::Value::as_str)
            .map(Self::parse)
            .transpose()
    }

    /// The alpha for one layer stem: an `alpha_pattern` entry whose key the
    /// stem ends with (PEFT matches by module-name suffix), else the global
    /// `lora_alpha`.
    pub(crate) fn alpha_for(&self, stem: &str) -> Option<f64> {
        self.alpha_pattern
            .iter()
            .filter(|(pattern, _)| stem.ends_with(pattern.as_str()))
            .max_by_key(|(pattern, _)| pattern.len())
            .map(|(_, alpha)| *alpha)
            .or(self.alpha)
    }
}

/// `user_scale · alpha / rank` for a layer whose `down` matrix has `rank`
/// rows. A per-layer `.alpha` tensor wins, then the metadata, then no alpha
/// (scale 1 per unit of `user_scale`, PEFT's `alpha = r` default).
pub(crate) fn effective_scale(
    user_scale: f64,
    rank: usize,
    tensor_alpha: Option<f64>,
    metadata_alpha: Option<f64>,
) -> f64 {
    match tensor_alpha.or(metadata_alpha) {
        Some(alpha) if rank > 0 => user_scale * alpha / rank as f64,
        _ => user_scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(targets: Vec<Qwen21LoraTarget>) -> Vec<(String, Option<(usize, usize)>)> {
        targets
            .into_iter()
            .map(|target| (target.candle_key, target.up_rows))
            .collect()
    }

    #[test]
    fn peft_and_prefixed_stems_map_to_the_exact_module() {
        for stem in [
            "transformer.transformer_blocks.3.attn.to_q",
            "transformer_blocks.3.attn.to_q",
            "base_model.model.transformer_blocks.3.attn.to_q",
            "diffusion_model.transformer_blocks.3.attn.to_q",
            "model.diffusion_model.transformer_blocks.3.attn.to_q",
        ] {
            assert_eq!(
                keys(map_key(stem).unwrap()),
                vec![("transformer_blocks.3.attn.to_q.weight".to_string(), None)],
                "{stem}"
            );
        }
        for (stem, key) in [
            ("transformer.modulation.1", "modulation.1.weight"),
            (
                "transformer.time_text_embed.timestep_embedder.linear_2",
                "time_text_embed.timestep_embedder.linear_2.weight",
            ),
            ("transformer.txt_in.in_layer", "txt_in.in_layer.weight"),
            ("transformer.norm_out.linear", "norm_out.linear.weight"),
            (
                "transformer.transformer_blocks.31.attn.to_out.0",
                "transformer_blocks.31.attn.to_out.0.weight",
            ),
            (
                "transformer.transformer_blocks.0.img_mlp.gate_layer",
                "transformer_blocks.0.img_mlp.gate_layer.weight",
            ),
        ] {
            assert_eq!(keys(map_key(stem).unwrap()), vec![(key.to_string(), None)]);
        }
    }

    #[test]
    fn kohya_flattened_stems_resolve_through_the_table() {
        for (stem, key) in [
            ("lora_unet_modulation_1", "modulation.1.weight"),
            ("lora_unet_txt_in_out_layer", "txt_in.out_layer.weight"),
            (
                "lora_unet_transformer_blocks_12_img_mlp_gate_layer",
                "transformer_blocks.12.img_mlp.gate_layer.weight",
            ),
            (
                "lora_unet_transformer_blocks_12_attn_to_out_0",
                "transformer_blocks.12.attn.to_out.0.weight",
            ),
            (
                "lora_unet_time_text_embed_timestep_embedder_linear_1",
                "time_text_embed.timestep_embedder.linear_1.weight",
            ),
        ] {
            assert_eq!(
                keys(map_key(stem).unwrap()),
                vec![(key.to_string(), None)],
                "{stem}"
            );
        }
    }

    #[test]
    fn comfy_fused_gate_up_splits_gate_first() {
        let mlp = mlp_hidden_dim();
        let expected = vec![
            (
                "transformer_blocks.7.img_mlp.gate_layer.weight".to_string(),
                Some((0, mlp)),
            ),
            (
                "transformer_blocks.7.img_mlp.proj.weight".to_string(),
                Some((mlp, mlp)),
            ),
        ];
        assert_eq!(
            keys(map_key("diffusion_model.transformer_blocks.7.img_mlp.gate_up").unwrap()),
            expected
        );
        assert_eq!(
            keys(map_key("lora_unet_transformer_blocks_7_img_mlp_gate_up").unwrap()),
            expected
        );
        assert!(map_key("transformer_blocks.99.img_mlp.gate_up").is_err());
    }

    #[test]
    fn foreign_architectures_and_components_are_refused() {
        for stem in [
            "transformer.transformer_blocks.0.attn.add_q_proj",
            "transformer_blocks.0.txt_mlp.net.0.proj",
            "transformer_blocks.0.img_mod.1",
            "text_encoder.model.layers.0.self_attn.q_proj",
            "transformer_blocks.0.attn.to_qkv",
            "lora_unet_double_blocks_0_img_attn_qkv",
            "transformer_blocks.32.attn.to_q",
        ] {
            assert!(map_key(stem).is_err(), "{stem}");
        }
        // Non-linear modules are skipped, not refused.
        assert!(map_key("transformer_blocks.0.attn.norm_q")
            .unwrap()
            .is_empty());
        assert!(map_adapter(["transformer_blocks.0.attn.norm_k"]).is_err());
        assert_eq!(
            map_adapter([
                "transformer_blocks.0.attn.norm_k",
                "transformer_blocks.0.attn.to_k"
            ])
            .unwrap()
            .len(),
            2
        );
    }

    #[test]
    fn every_captured_viggle_module_maps() {
        #[derive(serde::Deserialize)]
        struct Layout {
            files: HashMap<String, File>,
        }
        #[derive(serde::Deserialize)]
        struct File {
            modules: HashMap<String, serde_json::Value>,
            #[serde(rename = "__metadata__")]
            metadata: HashMap<String, String>,
        }
        let layout: Layout = serde_json::from_str(include_str!(
            "../../testdata/qwen_image21/viggle_lora_layout.json"
        ))
        .unwrap();
        assert!(!layout.files.is_empty());
        for (name, file) in &layout.files {
            for module in file.modules.keys() {
                let stem = format!("transformer.{}", module.replace("{i}", "5"));
                assert_eq!(map_key(&stem).unwrap().len(), 1, "{name}: {module}");
            }
            let scale = LoraMetadataScale::parse(&file.metadata["lora_adapter_metadata"]).unwrap();
            assert_eq!(scale.alpha, scale.rank, "{name}");
            let rank = scale.rank.unwrap() as usize;
            assert_eq!(effective_scale(1.0, rank, None, scale.alpha), 1.0);
        }
    }

    #[test]
    fn scale_prefers_tensor_alpha_then_metadata_then_none() {
        assert_eq!(effective_scale(0.5, 16, Some(8.0), Some(32.0)), 0.25);
        assert_eq!(effective_scale(0.5, 16, None, Some(32.0)), 1.0);
        assert_eq!(effective_scale(0.5, 16, None, None), 0.5);
        let scale = LoraMetadataScale::parse(
            r#"{"lora_alpha": 16, "r": 32, "alpha_pattern": {"attn.to_q": 64, "to_q": 8}}"#,
        )
        .unwrap();
        assert_eq!(scale.alpha, Some(16.0));
        assert_eq!(scale.rank, Some(32.0));
        // The longest matching pattern wins.
        assert_eq!(
            scale.alpha_for("transformer_blocks.0.attn.to_q"),
            Some(64.0)
        );
        assert_eq!(
            scale.alpha_for("transformer_blocks.0.attn.to_k"),
            Some(16.0)
        );
        assert!(LoraMetadataScale::parse("not json").is_err());
    }
}
#[cfg(test)]
mod viggle_file_tests {
    use super::*;

    /// Both published Viggle adapters: every layer stem maps, and the scale
    /// read from their `__metadata__` (no `.alpha` tensors) is exactly 1.
    #[test]
    #[ignore = "requires QWEN_IMAGE21_FIXTURES (the Viggle files under ../viggle)"]
    fn published_viggle_adapters_map_and_scale_to_one() {
        let Some(fixtures) = std::env::var_os("QWEN_IMAGE21_FIXTURES") else {
            return;
        };
        let dir = std::path::Path::new(&fixtures).join("../viggle");
        for rank in [128usize, 256] {
            let path = dir.join(format!(
                "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r{rank}.safetensors"
            ));
            let scale = LoraMetadataScale::read(&path).unwrap().unwrap();
            assert_eq!(scale.alpha, Some(rank as f64));
            assert_eq!(scale.rank, Some(rank as f64));
            let adapter = crate::flux::lora::LoraAdapter::load(&path).unwrap();
            assert_eq!(adapter.layers.len(), 227);
            let mapped = map_adapter(adapter.layers.keys().map(String::as_str)).unwrap();
            assert!(mapped.iter().all(|(_, targets)| targets.len() == 1));
            for (stem, layer) in &adapter.layers {
                assert!(layer.alpha.is_none());
                let rank = layer.a.dim(0).unwrap();
                assert_eq!(
                    effective_scale(1.0, rank, layer.alpha, scale.alpha_for(stem)),
                    1.0
                );
            }
        }
    }
}
