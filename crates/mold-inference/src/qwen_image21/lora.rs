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
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};

use crate::flux::lora::{LoraAdapter, LoraSpec};
use crate::flux::lora_bypass::{build_registry_with, BypassTarget, LoraRegistry};

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

/// Prefixes a trainer may put in front of the DOTTED module path.
const PREFIXES: [&str; 4] = [
    "base_model.model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
];

/// Prefixes in front of a FLATTENED module path (every `.` replaced by `_`):
/// Kohya's `lora_unet_` and SimpleTuner's lycoris export, which ComfyUI maps
/// as `"lycoris_{}".format(key_lora.replace(".", "_"))`
/// (`comfy/lora.py:338`, the Qwen Image arm that also addresses the 2.1
/// `gate_up` halves). A flattened stem only resolves through the
/// flattened table: stripping `lycoris_` and then looking the remainder up
/// among dotted module names can never match.
const FLATTENED_PREFIXES: [&str; 2] = ["lora_unet_", "lycoris_"];

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
    while let Some(rest) = PREFIXES.iter().find_map(|prefix| key.strip_prefix(prefix)) {
        key = rest;
    }
    // Kohya `lora_unet_<flattened>` and lycoris `lycoris_<flattened>`.
    if let Some(flat) = FLATTENED_PREFIXES
        .iter()
        .find_map(|prefix| key.strip_prefix(prefix))
    {
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

/// One adapter of a render's LoRA stack.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Qwen21LoraEntry {
    pub path: PathBuf,
    pub scale: f64,
}

impl Qwen21LoraEntry {
    fn path_hash(&self) -> u64 {
        crate::flux2::lora::lora_path_hash(&self.path.to_string_lossy())
    }
}

/// One adapter's identity in an installed stack: path hash, scale BITS and
/// [`file_identity`].
pub(crate) type Qwen21LoraFingerprint = (u64, u64, u64);

/// Identity of an installed stack: path hash, scale BITS and file identity
/// per adapter, in order (the `LoraFingerprint` rule every family's residency
/// reads, plus the file). The path alone is not an identity: a retrained
/// adapter written over the same path at the same scale would otherwise
/// match the stack already in the bypass slots and keep rendering the stale
/// weights.
pub(crate) fn fingerprint(entries: &[Qwen21LoraEntry]) -> Vec<Qwen21LoraFingerprint> {
    entries
        .iter()
        .map(|entry| {
            (
                entry.path_hash(),
                entry.scale.to_bits(),
                file_identity(&entry.path),
            )
        })
        .collect()
}

/// A cheap identity for the bytes at `path`, from metadata alone: length,
/// modification time and — on Unix — device, inode and status-change time.
/// `ctime` is the part a writer cannot forge (it is stamped by the kernel on
/// every write, and no syscall sets it), and the inode catches a file
/// replaced by rename. Hashing the content instead would read a multi-GB
/// adapter on every request just to learn that nothing changed. An unreadable
/// file answers 0; loading it fails right after, so that value is never
/// installed.
fn file_identity(path: &Path) -> u64 {
    use std::hash::{Hash, Hasher};
    let Ok(metadata) = std::fs::metadata(path) else {
        return 0;
    };
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    metadata.len().hash(&mut hasher);
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|since| since.as_nanos())
        .hash(&mut hasher);
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        (
            metadata.dev(),
            metadata.ino(),
            metadata.ctime(),
            metadata.ctime_nsec(),
        )
            .hash(&mut hasher);
    }
    hasher.finish()
}

/// Load an adapter, giving every layer the alpha PEFT would scale it by
/// ([`LoraMetadataScale::layer_alpha`]): a layer's own `.alpha` tensor, else
/// its metadata `alpha_pattern` / `lora_alpha` — the only place the Viggle
/// files store it — under rsLoRA's `sqrt(r)` rule when the metadata declares
/// it. A DoRA adapter is refused before any tensor is read.
pub(crate) fn load_adapter(path: &Path) -> Result<LoraAdapter> {
    let header = read_header(path)?;
    refuse_dora(path, &header)?;
    let mut adapter = LoraAdapter::load(path)?;
    if let Some(metadata) = &header.metadata {
        for (stem, layer) in adapter.layers.iter_mut() {
            let rank = layer.a.dim(0)?;
            layer.alpha = metadata
                .layer_alpha(stem, rank, layer.alpha)
                .with_context(|| format!("LoRA {}", path.display()))?;
        }
    }
    Ok(adapter)
}

fn bypass_target(target: Qwen21LoraTarget) -> BypassTarget {
    match target.up_rows {
        None => BypassTarget::Direct {
            candle_key: target.candle_key,
        },
        Some(rows) => BypassTarget::Rows {
            candle_key: target.candle_key,
            up_rows: Some(rows),
            out_offset: None,
        },
    }
}

/// Build the bypass registry for `entries` on `device` at `dtype`, keyed by
/// the transformer's `<module>.weight` names
/// ([`super::transformer::QwenImage21Transformer::install_lora`]). Every
/// adapter is checked against the module table first, so a Qwen-Image 2512
/// or text-encoder LoRA is refused before anything reaches the device.
pub(crate) fn build_registry(
    entries: &[Qwen21LoraEntry],
    device: &Device,
    dtype: DType,
) -> Result<LoraRegistry> {
    let adapters = entries
        .iter()
        .map(|entry| {
            load_adapter(&entry.path)
                .with_context(|| format!("failed to load LoRA {}", entry.path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    for (adapter, entry) in adapters.iter().zip(entries) {
        map_adapter(adapter.layers.keys().map(String::as_str))
            .with_context(|| format!("LoRA {}", entry.path.display()))?;
    }
    let specs: Vec<LoraSpec<'_>> = adapters
        .iter()
        .zip(entries)
        .map(|(adapter, entry)| LoraSpec {
            adapter,
            scale: entry.scale,
            path_hash: entry.path_hash(),
        })
        .collect();
    build_registry_with(
        &specs,
        |stem| Ok(map_key(stem)?.into_iter().map(bypass_target).collect()),
        &HashMap::new(),
        device,
        dtype,
    )
}
/// PEFT's scale metadata (`LoraConfig` as `peft` serializes it into
/// `lora_adapter_metadata`, diffusers `loaders/lora_base.py:65`).
///
/// Every field is read the way PEFT itself applies it when it builds the
/// layer (peft 0.21 `tuners/lora/model.py:233-237`, `tuners/lora/layer.py:
/// 278-281`):
///
/// * the layer's alpha is the FIRST `alpha_pattern` key (in the file's own
///   order) that matches the module name under [`peft_pattern_matches`], else
///   `lora_alpha`; its rank likewise from `rank_pattern`, else `r`;
/// * `use_rslora` scales by `alpha / sqrt(r)` instead of `alpha / r`;
/// * `use_dora` (a `lora_magnitude_vector` per layer) is refused: DoRA
///   renormalizes the whole adapted weight column by column, which a
///   forward-time `W x + B (A x)` bypass cannot express, and dropping the
///   magnitude silently renders a different adapter.
#[derive(Debug, Clone, Default)]
pub(crate) struct LoraMetadataScale {
    pub alpha: Option<f64>,
    pub rank: Option<f64>,
    pub alpha_pattern: Vec<PeftPattern>,
    pub rank_pattern: Vec<PeftPattern>,
    pub use_rslora: bool,
    pub use_dora: bool,
}

/// One `alpha_pattern` / `rank_pattern` entry: its key compiled to PEFT's
/// matcher, and its value.
#[derive(Debug, Clone)]
pub(crate) struct PeftPattern {
    matcher: regex::Regex,
    pub value: f64,
}

impl PeftPattern {
    fn new(key: &str, value: f64) -> Result<Self> {
        // `get_pattern_key` (peft 0.21 `utils/other.py:1524-1532`):
        // `re.match(rf"(.*\.)?({key})$", module_name)` — anchored at the start
        // by `re.match`, at the end by `$`, and the key is a REGEX that may
        // only be preceded by a whole dotted prefix. So `1.attn.to_q` names
        // block 1 and never blocks 11, 21 or 31.
        let matcher = regex::Regex::new(&format!(r"^(?:.*\.)?(?:{key})$")).with_context(|| {
            format!(
                "LoRA metadata pattern `{key}` is not a regular expression this engine can \
                 evaluate the way PEFT does"
            )
        })?;
        Ok(Self { matcher, value })
    }
}

/// Whether PEFT would pick `pattern` for `module_name`.
pub(crate) fn peft_pattern_matches(pattern: &PeftPattern, module_name: &str) -> bool {
    pattern.matcher.is_match(module_name)
}

/// The first pattern (in file order, as Python's dict iteration yields
/// them) that matches `module_name`.
fn first_match<'a>(patterns: &'a [PeftPattern], module_name: &str) -> Option<&'a PeftPattern> {
    patterns
        .iter()
        .find(|pattern| peft_pattern_matches(pattern, module_name))
}

/// JSON parsed with object keys in FILE order. `serde_json::Value` sorts keys
/// unless the `preserve_order` feature happens to be unified in, and PEFT's
/// "first matching pattern" depends on the order the trainer wrote. Only
/// what the scale reads is kept; strings, arrays and nulls parse to `Other`.
#[derive(Debug, Clone)]
enum OrderedJson {
    Bool(bool),
    Number(f64),
    Object(Vec<(String, OrderedJson)>),
    Other,
}

impl OrderedJson {
    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(*value),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    fn as_object(&self) -> Option<&[(String, OrderedJson)]> {
        match self {
            Self::Object(entries) => Some(entries),
            _ => None,
        }
    }
}

impl<'de> serde::Deserialize<'de> for OrderedJson {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = OrderedJson;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("any JSON value")
            }
            fn visit_unit<E>(self) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Other)
            }
            fn visit_none<E>(self) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Other)
            }
            fn visit_bool<E>(self, value: bool) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Bool(value))
            }
            fn visit_i64<E>(self, value: i64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Number(value as f64))
            }
            fn visit_u64<E>(self, value: u64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Number(value as f64))
            }
            fn visit_f64<E>(self, value: f64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Number(value))
            }
            fn visit_str<E>(self, _: &str) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Other)
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<OrderedJson, A::Error> {
                while seq.next_element::<OrderedJson>()?.is_some() {}
                Ok(OrderedJson::Other)
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<OrderedJson, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry::<String, OrderedJson>()? {
                    entries.push(entry);
                }
                Ok(OrderedJson::Object(entries))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

/// The module name PEFT matched its patterns against: the stem without the
/// component prefix diffusers adds (`transformer.`) or any other dotted
/// trainer prefix.
fn peft_module_name(stem: &str) -> &str {
    let mut key = stem;
    while let Some(rest) = PREFIXES.iter().find_map(|prefix| key.strip_prefix(prefix)) {
        key = rest;
    }
    key
}

impl LoraMetadataScale {
    /// Parse the `lora_adapter_metadata` JSON. Keys may be bare (`lora_alpha`)
    /// or component-prefixed (`transformer.lora_alpha`, the Viggle files).
    pub(crate) fn parse(json: &str) -> Result<Self> {
        let value: OrderedJson =
            serde_json::from_str(json).context("lora_adapter_metadata is not JSON")?;
        let object = value
            .as_object()
            .context("lora_adapter_metadata is not an object")?;
        let field = |name: &str| {
            let prefixed = format!("transformer.{name}");
            object
                .iter()
                .find(|(key, _)| key == name)
                .or_else(|| object.iter().find(|(key, _)| *key == prefixed))
                .map(|(_, value)| value)
        };
        let patterns = |name: &str| -> Result<Vec<PeftPattern>> {
            let Some(entries) = field(name).and_then(OrderedJson::as_object) else {
                return Ok(Vec::new());
            };
            entries
                .iter()
                .map(|(key, value)| {
                    let value = value.as_f64().with_context(|| {
                        format!("LoRA metadata {name} entry `{key}` is not a number")
                    })?;
                    PeftPattern::new(key, value)
                })
                .collect()
        };
        Ok(Self {
            alpha: field("lora_alpha").and_then(OrderedJson::as_f64),
            rank: field("r").and_then(OrderedJson::as_f64),
            alpha_pattern: patterns("alpha_pattern")?,
            rank_pattern: patterns("rank_pattern")?,
            use_rslora: field("use_rslora")
                .and_then(OrderedJson::as_bool)
                .unwrap_or(false),
            use_dora: field("use_dora")
                .and_then(OrderedJson::as_bool)
                .unwrap_or(false),
        })
    }

    /// Read it from a safetensors file's `__metadata__`, if present. Only
    /// the header is read, never the tensors ([`load_adapter`] reads the
    /// header itself; this is the tests' door).
    #[cfg(test)]
    pub(crate) fn read(path: &Path) -> Result<Option<Self>> {
        Ok(read_header(path)?.metadata)
    }

    /// The alpha PEFT gives the layer at `stem`: the first matching
    /// `alpha_pattern` key, else the global `lora_alpha`.
    pub(crate) fn alpha_for(&self, stem: &str) -> Option<f64> {
        first_match(&self.alpha_pattern, peft_module_name(stem))
            .map(|pattern| pattern.value)
            .or(self.alpha)
    }

    /// The rank PEFT builds the layer at `stem` with: the first matching
    /// `rank_pattern` key, else the global `r`.
    pub(crate) fn rank_for(&self, stem: &str) -> Option<f64> {
        first_match(&self.rank_pattern, peft_module_name(stem))
            .map(|pattern| pattern.value)
            .or(self.rank)
    }

    /// The alpha to hand the shared bypass registry for the layer at `stem`,
    /// whose `down` matrix has `tensor_rank` rows and which may carry its own
    /// `.alpha` tensor. The registry scales by `alpha / tensor_rank`
    /// (`flux::lora_bypass::build_registry_with`), so rsLoRA's `alpha /
    /// sqrt(r)` is expressed as the alpha `alpha · sqrt(r)`.
    ///
    /// A metadata rank that disagrees with the tensors is refused: PEFT builds
    /// the layer at the metadata rank and would fail to load those tensors
    /// into it, so there is no scale it would have used.
    pub(crate) fn layer_alpha(
        &self,
        stem: &str,
        tensor_rank: usize,
        tensor_alpha: Option<f64>,
    ) -> Result<Option<f64>> {
        if let Some(rank) = self.rank_for(stem) {
            anyhow::ensure!(
                rank == tensor_rank as f64,
                "LoRA layer `{stem}` has rank {tensor_rank} but its PEFT metadata says {rank}"
            );
        }
        let alpha = tensor_alpha.or_else(|| self.alpha_for(stem));
        Ok(match alpha {
            Some(alpha) if self.use_rslora => Some(alpha * (tensor_rank as f64).sqrt()),
            alpha => alpha,
        })
    }
}

/// What [`read_header`] reads out of a safetensors header.
struct AdapterHeader {
    metadata: Option<LoraMetadataScale>,
    tensor_names: Vec<String>,
}

/// Read a safetensors header: its PEFT metadata and its tensor names.
fn read_header(path: &Path) -> Result<AdapterHeader> {
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
    let metadata = header
        .get("__metadata__")
        .and_then(|metadata| metadata.get("lora_adapter_metadata"))
        .and_then(serde_json::Value::as_str)
        .map(LoraMetadataScale::parse)
        .transpose()?;
    let tensor_names = header
        .as_object()
        .map(|object| {
            object
                .keys()
                .filter(|key| *key != "__metadata__")
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    Ok(AdapterHeader {
        metadata,
        tensor_names,
    })
}

/// Refuse a DoRA adapter by name. PEFT stores the per-column magnitude as
/// `<layer>.lora_magnitude_vector` (peft 0.21 `tuners/lora/layer.py:115,
/// 145`) and flags the config `use_dora`; Kohya/LyCORIS exports carry it as
/// `<layer>.dora_scale` (ComfyUI `comfy/weight_adapter/lora.py:153-210`).
fn refuse_dora(path: &Path, header: &AdapterHeader) -> Result<()> {
    let declared = header
        .metadata
        .as_ref()
        .is_some_and(|metadata| metadata.use_dora);
    let magnitude = header
        .tensor_names
        .iter()
        .find(|name| name.contains("lora_magnitude_vector") || name.contains("dora_scale"));
    if declared || magnitude.is_some() {
        bail!(
            "LoRA {} is a DoRA adapter{}; Qwen Image 2.1 applies adapters as a forward-time \
             bypass, which cannot apply DoRA's weight renormalization — export it as a plain LoRA",
            path.display(),
            magnitude
                .map(|name| format!(" (`{name}`)"))
                .unwrap_or_default()
        );
    }
    Ok(())
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

    /// SimpleTuner's lycoris spelling is `lycoris_` + the module path with
    /// every `.` flattened to `_` (ComfyUI `comfy/lora.py:338`, the Qwen
    /// Image arm that also maps the 2.1 `gate_up` halves), so the stripped
    /// key resolves through the flattened table, never the dotted one.
    #[test]
    fn lycoris_stems_resolve_through_the_flattened_table() {
        for (stem, key) in [
            (
                "lycoris_transformer_blocks_3_attn_to_q",
                "transformer_blocks.3.attn.to_q.weight",
            ),
            (
                "lycoris_transformer_blocks_31_attn_to_out_0",
                "transformer_blocks.31.attn.to_out.0.weight",
            ),
            (
                "lycoris_transformer_blocks_0_img_mlp_gate_layer",
                "transformer_blocks.0.img_mlp.gate_layer.weight",
            ),
            (
                "lycoris_transformer_blocks_0_img_mlp_proj",
                "transformer_blocks.0.img_mlp.proj.weight",
            ),
            ("lycoris_modulation_1", "modulation.1.weight"),
            ("lycoris_txt_in_in_layer", "txt_in.in_layer.weight"),
            ("lycoris_img_in", "img_in.weight"),
        ] {
            assert_eq!(
                keys(map_key(stem).unwrap()),
                vec![(key.to_string(), None)],
                "{stem}"
            );
        }
        let mlp = mlp_hidden_dim();
        assert_eq!(
            keys(map_key("lycoris_transformer_blocks_7_img_mlp_gate_up").unwrap()),
            vec![
                (
                    "transformer_blocks.7.img_mlp.gate_layer.weight".to_string(),
                    Some((0, mlp)),
                ),
                (
                    "transformer_blocks.7.img_mlp.proj.weight".to_string(),
                    Some((mlp, mlp)),
                ),
            ]
        );
        assert!(map_key("lycoris_transformer_blocks_0_attn_norm_q")
            .unwrap()
            .is_empty());
        assert!(map_key("lycoris_transformer_blocks_32_attn_to_q").is_err());
        assert!(map_key("lycoris_transformer_blocks_0_attn_add_q_proj").is_err());
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
            assert!(!scale.use_rslora && !scale.use_dora, "{name}");
            let rank = scale.rank.unwrap() as usize;
            // alpha / rank = 1: the registry divides what `layer_alpha` hands it.
            let alpha = scale
                .layer_alpha("transformer.transformer_blocks.5.attn.to_q", rank, None)
                .unwrap();
            assert_eq!(alpha, Some(rank as f64), "{name}");
        }
    }

    /// A retrained adapter written over the same path at the same scale must
    /// not match the stack already installed: the fingerprint carries the
    /// file's identity, not only its name.
    #[test]
    fn a_file_rewritten_in_place_changes_the_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("style.safetensors");
        std::fs::write(&path, b"first training run").unwrap();
        let entries = [Qwen21LoraEntry {
            path: path.clone(),
            scale: 0.8,
        }];
        let before = fingerprint(&entries);
        assert_eq!(before, fingerprint(&entries), "stable while untouched");
        // Same length, same path, same scale: only the content changed.
        std::fs::write(&path, b"secnd training run").unwrap();
        let file = std::fs::File::options().write(true).open(&path).unwrap();
        file.set_modified(std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(7))
            .unwrap();
        drop(file);
        assert_ne!(before, fingerprint(&entries));
        // Replacing the file (a new inode) is a new identity too.
        let replacement = dir.path().join("next.safetensors");
        std::fs::write(&replacement, b"first training run").unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        let replaced = fingerprint(&entries);
        assert_ne!(before, replaced);
        // The scale is still part of it.
        let rescaled = fingerprint(&[Qwen21LoraEntry { path, scale: 0.5 }]);
        assert_ne!(replaced, rescaled);
    }

    #[test]
    fn layer_alpha_prefers_the_tensor_then_the_metadata() {
        let scale = LoraMetadataScale::parse(r#"{"lora_alpha": 16, "r": 32}"#).unwrap();
        assert_eq!(scale.alpha, Some(16.0));
        assert_eq!(scale.rank, Some(32.0));
        let stem = "transformer.transformer_blocks.0.attn.to_q";
        assert_eq!(scale.layer_alpha(stem, 32, Some(8.0)).unwrap(), Some(8.0));
        assert_eq!(scale.layer_alpha(stem, 32, None).unwrap(), Some(16.0));
        let bare = LoraMetadataScale::parse("{}").unwrap();
        assert_eq!(bare.layer_alpha(stem, 32, None).unwrap(), None);
        assert!(LoraMetadataScale::parse("not json").is_err());
    }

    /// PEFT's `get_pattern_key` (peft 0.21 `utils/other.py:1524-1532`) is
    /// `re.match(rf"(.*\.)?({key})$", name)`: the key may only be preceded by
    /// a whole dotted prefix, so `1.attn.to_q` is block 1 and never block 11,
    /// 21 or 31 — which a suffix test (`ends_with`) got wrong.
    #[test]
    fn peft_patterns_match_only_at_a_module_boundary() {
        let scale = LoraMetadataScale::parse(
            r#"{"lora_alpha": 16, "r": 32, "alpha_pattern": {"1.attn.to_q": 4}}"#,
        )
        .unwrap();
        for stem in [
            "transformer.transformer_blocks.1.attn.to_q",
            "transformer_blocks.1.attn.to_q",
            "base_model.model.transformer_blocks.1.attn.to_q",
        ] {
            assert_eq!(scale.alpha_for(stem), Some(4.0), "{stem}");
        }
        for block in [11, 21, 31] {
            let stem = format!("transformer.transformer_blocks.{block}.attn.to_q");
            assert_eq!(scale.alpha_for(&stem), Some(16.0), "{stem}");
        }
        assert_eq!(
            scale.alpha_for("transformer.transformer_blocks.1.attn.to_k"),
            Some(16.0)
        );
    }

    /// The keys are REGULAR EXPRESSIONS ("layer names or regexp expression",
    /// peft 0.21 `tuners/lora/config.py:524-529`).
    #[test]
    fn peft_patterns_are_regular_expressions() {
        let scale = LoraMetadataScale::parse(
            r#"{"lora_alpha": 16, "r": 8,
                "alpha_pattern": {"transformer_blocks\\.(1|2)\\.attn\\.to_[qk]": 32,
                                  "^img_in": 2}}"#,
        )
        .unwrap();
        for (stem, alpha) in [
            ("transformer.transformer_blocks.1.attn.to_q", 32.0),
            ("transformer.transformer_blocks.2.attn.to_k", 32.0),
            ("transformer.transformer_blocks.2.attn.to_v", 16.0),
            ("transformer.transformer_blocks.12.attn.to_q", 16.0),
            // `^` inside the group anchors the key at the start of the name.
            ("transformer.img_in", 2.0),
            ("transformer.transformer_blocks.0.img_in", 16.0),
        ] {
            assert_eq!(scale.alpha_for(stem), Some(alpha), "{stem}");
        }
        // A key this engine cannot evaluate the way Python's `re` would is
        // refused, never silently skipped.
        assert!(LoraMetadataScale::parse(r#"{"alpha_pattern": {"(?<=x)to_q": 1}}"#).is_err());
    }

    /// PEFT takes the FIRST matching key in the dict's order, not the most
    /// specific one — and the file's order survives parsing.
    #[test]
    fn the_first_matching_pattern_in_file_order_wins() {
        let first_general = LoraMetadataScale::parse(
            r#"{"lora_alpha": 16, "alpha_pattern": {"to_q": 8, "attn.to_q": 64}}"#,
        )
        .unwrap();
        let first_specific = LoraMetadataScale::parse(
            r#"{"lora_alpha": 16, "alpha_pattern": {"attn.to_q": 64, "to_q": 8}}"#,
        )
        .unwrap();
        let stem = "transformer.transformer_blocks.0.attn.to_q";
        assert_eq!(first_general.alpha_for(stem), Some(8.0));
        assert_eq!(first_specific.alpha_for(stem), Some(64.0));
    }

    /// `rank_pattern` resolves the same way, and a layer whose tensors
    /// disagree with the rank PEFT would have built it at is refused (PEFT
    /// itself could not load those tensors into the layer).
    #[test]
    fn rank_pattern_resolves_like_peft_and_must_agree_with_the_tensors() {
        let scale = LoraMetadataScale::parse(
            r#"{"lora_alpha": 16, "r": 32, "rank_pattern": {"attn.to_v": 16}}"#,
        )
        .unwrap();
        let to_v = "transformer.transformer_blocks.3.attn.to_v";
        let to_q = "transformer.transformer_blocks.3.attn.to_q";
        assert_eq!(scale.rank_for(to_v), Some(16.0));
        assert_eq!(scale.rank_for(to_q), Some(32.0));
        assert_eq!(scale.layer_alpha(to_v, 16, None).unwrap(), Some(16.0));
        assert!(scale.layer_alpha(to_v, 32, None).is_err());
        assert!(scale.layer_alpha(to_q, 16, None).is_err());
    }

    /// rsLoRA scales by `alpha / sqrt(r)` (peft 0.21 `tuners/lora/layer.py:
    /// 278-281`); the registry divides by the rank, so the alpha it is handed
    /// is `alpha · sqrt(r)`.
    #[test]
    fn rslora_scales_by_the_square_root_of_the_rank() {
        let scale =
            LoraMetadataScale::parse(r#"{"lora_alpha": 16, "r": 64, "use_rslora": true}"#).unwrap();
        assert!(scale.use_rslora);
        let alpha = scale
            .layer_alpha("transformer.transformer_blocks.0.attn.to_q", 64, None)
            .unwrap()
            .unwrap();
        assert!((alpha / 64.0 - 16.0 / 8.0).abs() < 1e-12, "{alpha}");
        let prefixed = LoraMetadataScale::parse(
            r#"{"transformer.lora_alpha": 16, "transformer.r": 64, "transformer.use_rslora": true}"#,
        )
        .unwrap();
        assert!(prefixed.use_rslora);
    }
}

/// Synthetic adapter files, and the production scale arithmetic read back
/// from the registry the engine installs (`build_registry` →
/// `flux::lora_bypass::build_registry_with`'s `alpha / rank`).
#[cfg(test)]
mod registry_tests {
    use super::*;

    /// Write a safetensors file of F32 tensors with an optional
    /// `lora_adapter_metadata` JSON string in its `__metadata__`.
    pub(super) fn write_adapter(
        path: &Path,
        tensors: &[(&str, Vec<usize>)],
        metadata: Option<&str>,
    ) {
        let mut header = serde_json::Map::new();
        if let Some(metadata) = metadata {
            header.insert(
                "__metadata__".into(),
                serde_json::json!({ "lora_adapter_metadata": metadata }),
            );
        }
        let mut data = Vec::new();
        for (index, (name, shape)) in tensors.iter().enumerate() {
            let count: usize = shape.iter().product();
            let start = data.len();
            for i in 0..count {
                let value = ((i + 7 * index) % 13) as f32 / 13.0 - 0.4;
                data.extend_from_slice(&value.to_le_bytes());
            }
            header.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [start, data.len()],
                }),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        std::fs::write(path, bytes).unwrap();
    }

    fn registry_for(path: &Path, scale: f64) -> Result<LoraRegistry> {
        build_registry(
            &[Qwen21LoraEntry {
                path: path.to_path_buf(),
                scale,
            }],
            &Device::Cpu,
            DType::F32,
        )
    }

    /// A DoRA adapter is refused by name — declared in its metadata or
    /// carrying a magnitude vector — instead of rendering its LoRA half alone.
    #[test]
    fn dora_adapters_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let q = "transformer.transformer_blocks.0.attn.to_q";
        let pair = [
            (format!("{q}.lora_A.weight"), vec![4, 8]),
            (format!("{q}.lora_B.weight"), vec![8, 4]),
        ];
        let declared = dir.path().join("declared.safetensors");
        let tensors: Vec<(&str, Vec<usize>)> = pair
            .iter()
            .map(|(name, shape)| (name.as_str(), shape.clone()))
            .collect();
        write_adapter(
            &declared,
            &tensors,
            Some(r#"{"lora_alpha": 4, "r": 4, "use_dora": true}"#),
        );
        let error = format!("{:#}", registry_for(&declared, 1.0).unwrap_err());
        assert!(error.contains("DoRA"), "{error}");

        let magnitude_name = format!("{q}.lora_magnitude_vector.weight");
        let mut tensors = tensors.clone();
        tensors.push((magnitude_name.as_str(), vec![8]));
        let magnitude = dir.path().join("magnitude.safetensors");
        write_adapter(&magnitude, &tensors, None);
        let error = format!("{:#}", registry_for(&magnitude, 1.0).unwrap_err());
        assert!(
            error.contains("DoRA") && error.contains("lora_magnitude_vector"),
            "{error}"
        );

        let kohya = dir.path().join("kohya.safetensors");
        write_adapter(
            &kohya,
            &[
                (
                    "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight",
                    vec![4, 8],
                ),
                (
                    "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight",
                    vec![8, 4],
                ),
                (
                    "lora_unet_transformer_blocks_0_attn_to_q.dora_scale",
                    vec![1, 8],
                ),
            ],
            None,
        );
        assert!(registry_for(&kohya, 1.0).is_err());
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
                    scale.layer_alpha(stem, rank, layer.alpha).unwrap(),
                    Some(rank as f64)
                );
            }
        }
    }
}
