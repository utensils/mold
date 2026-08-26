use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

// Upstream LTX-2 pads Gemma prompts to 256 tokens before the connector stage.
// The connector/register path is sensitive to this absolute layout.
pub const DEFAULT_GEMMA_MAX_LENGTH: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTokens {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
}

impl PromptTokens {
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }

    pub fn valid_len(&self) -> usize {
        self.attention_mask
            .iter()
            .filter(|mask| **mask != 0)
            .count()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedPromptPair {
    pub conditional: PromptTokens,
    pub unconditional: PromptTokens,
    pub pad_token_id: u32,
    pub eos_token_id: Option<u32>,
    pub max_length: usize,
}

impl EncodedPromptPair {
    #[allow(dead_code)]
    pub fn batch_input_ids(&self) -> [&[u32]; 2] {
        [&self.conditional.input_ids, &self.unconditional.input_ids]
    }

    #[allow(dead_code)]
    pub fn batch_attention_mask(&self) -> [&[u8]; 2] {
        [
            &self.conditional.attention_mask,
            &self.unconditional.attention_mask,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GemmaAssets {
    pub root: PathBuf,
    pub tokenizer_json: PathBuf,
    tokenizer_bytes: Option<Vec<u8>>,
    pub packed_weights: Option<PathBuf>,
    pub tokenizer_model: Option<PathBuf>,
    pub special_tokens_map: Option<PathBuf>,
    pub tokenizer_config: Option<PathBuf>,
    /// First `*.gguf` file found in `root` (lexically sorted), if any. Present
    /// when the user has installed a Q4 GGUF variant of Gemma 3 alongside (or
    /// instead of) the BF16 safetensors split.
    pub gguf_path: Option<PathBuf>,
}

impl GemmaAssets {
    pub fn discover(root: &Path) -> Result<Self> {
        if root.is_file() && root.extension().is_some_and(|ext| ext == "safetensors") {
            let tokenizer_bytes = read_packed_byte_tensor(root, "tokenizer_json")?;
            return Ok(Self {
                root: root.to_path_buf(),
                tokenizer_json: root.to_path_buf(),
                tokenizer_bytes: Some(tokenizer_bytes),
                packed_weights: Some(root.to_path_buf()),
                tokenizer_model: None,
                special_tokens_map: None,
                tokenizer_config: None,
                gguf_path: None,
            });
        }
        if !root.is_dir() {
            bail!(
                "Gemma assets '{}' are neither a directory nor a packed .safetensors file",
                root.display()
            );
        }

        let tokenizer_json = root.join("tokenizer.json");
        if !tokenizer_json.is_file() {
            bail!(
                "Gemma asset root '{}' is missing tokenizer.json",
                root.display()
            );
        }

        Ok(Self {
            root: root.to_path_buf(),
            tokenizer_json,
            tokenizer_bytes: None,
            packed_weights: None,
            tokenizer_model: candidate(root, "tokenizer.model"),
            special_tokens_map: candidate(root, "special_tokens_map.json"),
            tokenizer_config: candidate(root, "tokenizer_config.json"),
            gguf_path: discover_gguf(root),
        })
    }

    /// True when the asset root contains BF16 safetensors weights
    /// (`model.safetensors` or sharded `model-*-of-*.safetensors`).
    pub fn has_bf16_weights(&self) -> bool {
        if self.packed_weights.is_some() {
            return true;
        }
        let Ok(entries) = std::fs::read_dir(&self.root) else {
            return false;
        };
        entries.filter_map(|e| e.ok()).any(|entry| {
            entry
                .file_name()
                .to_str()
                .map(|name| {
                    (name == "model.safetensors"
                        || (name.starts_with("model-") && name.ends_with(".safetensors")))
                        && entry.path().is_file()
                })
                .unwrap_or(false)
        })
    }

    pub fn encode_prompt_pair(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
    ) -> Result<EncodedPromptPair> {
        self.encode_prompt_pair_with_max_length(prompt, negative_prompt, DEFAULT_GEMMA_MAX_LENGTH)
    }

    pub fn encode_prompt_pair_with_max_length(
        &self,
        prompt: &str,
        negative_prompt: Option<&str>,
        max_length: usize,
    ) -> Result<EncodedPromptPair> {
        let mut tokenizer = self.load_tokenizer(max_length)?;
        let (pad_token_id, eos_token_id) = self.special_token_ids(&tokenizer)?;
        let conditional = encode_with_tokenizer(&mut tokenizer, prompt)?;
        let unconditional =
            encode_with_tokenizer(&mut tokenizer, negative_prompt.unwrap_or_default())?;

        Ok(EncodedPromptPair {
            conditional,
            unconditional,
            pad_token_id,
            eos_token_id,
            max_length,
        })
    }

    fn load_tokenizer(&self, max_length: usize) -> Result<Tokenizer> {
        let tokenizer = if let Some(bytes) = self.tokenizer_bytes.as_deref() {
            Tokenizer::from_bytes(bytes)
        } else {
            Tokenizer::from_file(&self.tokenizer_json)
        };
        let mut tokenizer = tokenizer.map_err(|err| {
            anyhow!(
                "failed to load Gemma tokenizer '{}': {err}",
                self.tokenizer_json.display()
            )
        })?;
        let (pad_token, pad_token_id) = self.resolve_padding_token(&tokenizer)?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(max_length),
            direction: PaddingDirection::Left,
            pad_to_multiple_of: None,
            pad_id: pad_token_id,
            pad_type_id: 0,
            pad_token,
        }));
        tokenizer
            .with_truncation(Some(TruncationParams {
                direction: TruncationDirection::Right,
                max_length,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
            }))
            .map_err(|err| anyhow!("failed to configure Gemma tokenizer truncation: {err}"))?;
        Ok(tokenizer)
    }

    fn resolve_padding_token(&self, tokenizer: &Tokenizer) -> Result<(String, u32)> {
        let tokens = self.read_special_tokens()?;
        let eos_token = tokens
            .eos_token
            .or_else(|| infer_known_special_token(tokenizer, &["<eos>", "</s>", "<end_of_turn>"]));
        let eos_token_id = eos_token
            .as_deref()
            .and_then(|token| tokenizer.token_to_id(token));

        let pad_token = tokens
            .pad_token
            .or_else(|| infer_known_special_token(tokenizer, &["<pad>"]))
            .or_else(|| eos_token.clone());
        let pad_token_id = pad_token
            .as_deref()
            .and_then(|token| tokenizer.token_to_id(token))
            .or(eos_token_id)
            .ok_or_else(|| {
                anyhow!(
                    "could not determine Gemma pad/eos token from '{}'",
                    self.root.display()
                )
            })?;

        Ok((
            pad_token
                .or_else(|| tokenizer.id_to_token(pad_token_id))
                .unwrap_or_else(|| "[PAD]".to_string()),
            pad_token_id,
        ))
    }

    fn special_token_ids(&self, tokenizer: &Tokenizer) -> Result<(u32, Option<u32>)> {
        let (pad_token, pad_token_id) = self.resolve_padding_token(tokenizer)?;
        let eos_token_id = self
            .read_special_tokens()?
            .eos_token
            .or(Some(pad_token))
            .and_then(|token| tokenizer.token_to_id(&token));
        Ok((pad_token_id, eos_token_id))
    }

    fn read_special_tokens(&self) -> Result<ResolvedSpecialTokens> {
        let mut resolved = ResolvedSpecialTokens::default();
        for path in [
            self.special_tokens_map.as_ref(),
            self.tokenizer_config.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let data = fs::read(path).with_context(|| {
                format!(
                    "failed to read Gemma special-token metadata '{}'",
                    path.display()
                )
            })?;
            let parsed: SpecialTokensFile = serde_json::from_slice(&data).with_context(|| {
                format!(
                    "failed to parse Gemma special-token metadata '{}'",
                    path.display()
                )
            })?;
            if resolved.pad_token.is_none() {
                resolved.pad_token = parsed.pad_token.map(SpecialTokenValue::into_content);
            }
            if resolved.eos_token.is_none() {
                resolved.eos_token = parsed.eos_token.map(SpecialTokenValue::into_content);
            }
        }
        Ok(resolved)
    }
}

/// Read one U8/I8 byte tensor without mapping or copying the multi-GB weight
/// payload surrounding it. Safetensors stores both integer variants as raw
/// bytes, so their signedness is irrelevant for reconstructing tokenizer JSON.
fn read_packed_byte_tensor(path: &Path, name: &str) -> Result<Vec<u8>> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open packed Gemma assets '{}'", path.display()))?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length)?;
    let header_len = u64::from_le_bytes(length);
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let header: serde_json::Value = serde_json::from_slice(&header)
        .with_context(|| format!("invalid safetensors header in '{}'", path.display()))?;
    let tensor = header.get(name).ok_or_else(|| {
        anyhow!(
            "packed Gemma assets '{}' are missing {name}",
            path.display()
        )
    })?;
    let dtype = tensor.get("dtype").and_then(serde_json::Value::as_str);
    if !matches!(dtype, Some("U8" | "I8")) {
        bail!(
            "packed Gemma {name} in '{}' must use U8 or I8 storage, got {dtype:?}",
            path.display()
        );
    }
    let offsets = tensor
        .get("data_offsets")
        .and_then(serde_json::Value::as_array)
        .filter(|offsets| offsets.len() == 2)
        .ok_or_else(|| anyhow!("packed Gemma {name} has invalid data_offsets"))?;
    let start = offsets[0]
        .as_u64()
        .ok_or_else(|| anyhow!("packed Gemma {name} has invalid start offset"))?;
    let end = offsets[1]
        .as_u64()
        .ok_or_else(|| anyhow!("packed Gemma {name} has invalid end offset"))?;
    let byte_len: usize = end
        .checked_sub(start)
        .and_then(|len| len.try_into().ok())
        .ok_or_else(|| anyhow!("packed Gemma {name} byte range is invalid"))?;
    file.seek(SeekFrom::Start(8 + header_len + start))?;
    let mut bytes = vec![0u8; byte_len];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

#[allow(dead_code)]
pub fn pad_to_alignment(
    input_ids: &[u32],
    attention_mask: &[u8],
    pad_token_id: u32,
    alignment: usize,
) -> PromptTokens {
    assert_eq!(
        input_ids.len(),
        attention_mask.len(),
        "Gemma token ids and mask must have the same length"
    );
    assert!(alignment > 0, "alignment must be positive");

    let padded_len = input_ids.len().div_ceil(alignment) * alignment;
    let padding = padded_len - input_ids.len();
    let mut padded_ids = input_ids.to_vec();
    let mut padded_mask = attention_mask.to_vec();
    padded_ids.extend(std::iter::repeat_n(pad_token_id, padding));
    padded_mask.extend(std::iter::repeat_n(0, padding));
    PromptTokens {
        input_ids: padded_ids,
        attention_mask: padded_mask,
    }
}

#[allow(dead_code)]
pub fn left_pad_batch(sequences: &[Vec<u32>], pad_token_id: u32) -> (Vec<Vec<u32>>, Vec<Vec<u8>>) {
    let width = sequences
        .iter()
        .map(|sequence| sequence.len())
        .max()
        .unwrap_or(0);
    let mut padded_ids = Vec::with_capacity(sequences.len());
    let mut padded_masks = Vec::with_capacity(sequences.len());
    for sequence in sequences {
        let pad = width.saturating_sub(sequence.len());
        let mut ids = Vec::with_capacity(width);
        let mut mask = Vec::with_capacity(width);
        ids.extend(std::iter::repeat_n(pad_token_id, pad));
        ids.extend(sequence.iter().copied());
        mask.extend(std::iter::repeat_n(0, pad));
        mask.extend(std::iter::repeat_n(1, sequence.len()));
        padded_ids.push(ids);
        padded_masks.push(mask);
    }
    (padded_ids, padded_masks)
}

fn encode_with_tokenizer(tokenizer: &mut Tokenizer, text: &str) -> Result<PromptTokens> {
    let encoding = tokenizer
        .encode(text.trim(), true)
        .map_err(|err| anyhow!("Gemma tokenization failed: {err}"))?;
    let mut input_ids = encoding.get_ids().to_vec();
    let mut attention_mask = encoding
        .get_attention_mask()
        .iter()
        .map(|value| u8::from(*value != 0))
        .collect::<Vec<_>>();
    if let Some(bos_id) = tokenizer.token_to_id("<bos>") {
        let first_valid = attention_mask.iter().position(|mask| *mask != 0);
        if first_valid.is_none_or(|index| input_ids[index] != bos_id) {
            match first_valid {
                Some(index) if index > 0 => {
                    input_ids[index - 1] = bos_id;
                    attention_mask[index - 1] = 1;
                }
                Some(0) if !input_ids.is_empty() => {
                    input_ids.rotate_right(1);
                    input_ids[0] = bos_id;
                    attention_mask.rotate_right(1);
                    attention_mask[0] = 1;
                }
                None if !input_ids.is_empty() => {
                    let last = input_ids.len() - 1;
                    input_ids[last] = bos_id;
                    attention_mask[last] = 1;
                }
                _ => {}
            }
        }
    }
    Ok(PromptTokens {
        input_ids,
        attention_mask,
    })
}

fn candidate(root: &Path, filename: &str) -> Option<PathBuf> {
    let path = root.join(filename);
    path.is_file().then_some(path)
}

/// Find the first `*.gguf` file in `root` (lexically sorted). Returns `None`
/// when no GGUF is present. Used by [`GemmaAssets::discover`] to surface a
/// Q4 GGUF Gemma 3 weight file when the user has installed one.
fn discover_gguf(root: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(root).ok()?;
    let mut matches: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let is_gguf = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false);
            (is_gguf && path.is_file()).then_some(path)
        })
        .collect();
    matches.sort();
    matches.into_iter().next()
}

/// Which Gemma 3 weight format is loaded for the LTX-2 prompt encoder.
///
/// `Bf16Safetensors` matches the historical default — the unquantized
/// `model-*-of-*.safetensors` shards from `google/gemma-3-12b-it-qat-q4_0-unquantized`.
/// `Q4Gguf` is the new ~7 GB Q4 quantized variant from
/// `google/gemma-3-12b-it-qat-q4_0-gguf` that fits comfortably alongside a
/// streaming 22B LTX-2 transformer on a 24 GB GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmaVariant {
    Bf16Safetensors,
    Q4Gguf,
}

/// Resolve which Gemma 3 weight format to load given the assets present and
/// the user's `MOLD_LTX2_GEMMA_VARIANT` override.
///
/// Precedence (highest first):
/// 1. `MOLD_LTX2_GEMMA_VARIANT={q4,bf16}` — explicit override; errors if the
///    requested variant's files aren't present in the asset root.
/// 2. `MOLD_LTX2_GEMMA_VARIANT=auto` (or unset) with **only** GGUF on disk →
///    `Q4Gguf`.
/// 3. Auto with **only** BF16 on disk → `Bf16Safetensors`.
/// 4. Auto with both present → `Bf16Safetensors` (preserves historical
///    behavior so existing installs don't switch backends silently — opt in
///    explicitly via the env var).
/// 5. Auto with neither present → `Err`.
#[cfg(test)]
pub fn resolve_gemma_variant(assets: &GemmaAssets) -> Result<GemmaVariant> {
    resolve_gemma_variant_with_preference(
        assets,
        std::env::var("MOLD_LTX2_GEMMA_VARIANT").ok().as_deref(),
    )
}

pub fn resolve_gemma_variant_with_preference(
    assets: &GemmaAssets,
    preference: Option<&str>,
) -> Result<GemmaVariant> {
    let has_bf16 = assets.has_bf16_weights();
    let has_gguf = assets.gguf_path.is_some();

    if let Some(raw) = preference {
        let normalized = raw.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "q4" | "gguf" | "q4_gguf" => {
                if !has_gguf {
                    bail!(
                        "MOLD_LTX2_GEMMA_VARIANT=q4 requested but no .gguf file found in '{}'",
                        assets.root.display()
                    );
                }
                return Ok(GemmaVariant::Q4Gguf);
            }
            "bf16" | "safetensors" | "bf16_safetensors" => {
                if !has_bf16 {
                    bail!(
                        "MOLD_LTX2_GEMMA_VARIANT=bf16 requested but no model*.safetensors files \
                         found in '{}'",
                        assets.root.display()
                    );
                }
                return Ok(GemmaVariant::Bf16Safetensors);
            }
            "auto" | "" => { /* fall through to auto-detection */ }
            other => {
                tracing::warn!(
                    value = %other,
                    "unrecognised MOLD_LTX2_GEMMA_VARIANT value; expected q4/bf16/auto — \
                     falling back to auto-detection"
                );
            }
        }
    }

    match (has_bf16, has_gguf) {
        (true, _) => Ok(GemmaVariant::Bf16Safetensors),
        (false, true) => Ok(GemmaVariant::Q4Gguf),
        (false, false) => bail!(
            "Gemma asset root '{}' contains neither model*.safetensors nor *.gguf weights",
            assets.root.display()
        ),
    }
}

fn infer_known_special_token(tokenizer: &Tokenizer, candidates: &[&str]) -> Option<String> {
    candidates.iter().find_map(|candidate| {
        tokenizer
            .token_to_id(candidate)
            .map(|_| (*candidate).to_string())
    })
}

#[derive(Debug, Default, Clone)]
struct ResolvedSpecialTokens {
    pad_token: Option<String>,
    eos_token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SpecialTokensFile {
    #[serde(default)]
    pad_token: Option<SpecialTokenValue>,
    #[serde(default)]
    eos_token: Option<SpecialTokenValue>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SpecialTokenValue {
    String(String),
    Object { content: String },
}

impl SpecialTokenValue {
    fn into_content(self) -> String {
        match self {
            Self::String(value) => value,
            Self::Object { content } => content,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        left_pad_batch, pad_to_alignment, resolve_gemma_variant, EncodedPromptPair, GemmaAssets,
        GemmaVariant, DEFAULT_GEMMA_MAX_LENGTH,
    };
    use std::fs;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};

    /// Variant-resolver tests mutate `MOLD_LTX2_GEMMA_VARIANT`. Process-global
    /// env state can only be touched from one test at a time.
    fn variant_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// Take the env lock, snapshot the variant var, run `body`, restore.
    fn with_variant_env<F: FnOnce()>(value: Option<&str>, body: F) {
        let _guard = variant_env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let prior = std::env::var_os("MOLD_LTX2_GEMMA_VARIANT");
        // SAFETY: serialized through `variant_env_lock`.
        unsafe {
            match value {
                Some(v) => std::env::set_var("MOLD_LTX2_GEMMA_VARIANT", v),
                None => std::env::remove_var("MOLD_LTX2_GEMMA_VARIANT"),
            }
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
        unsafe {
            std::env::remove_var("MOLD_LTX2_GEMMA_VARIANT");
            if let Some(v) = prior {
                std::env::set_var("MOLD_LTX2_GEMMA_VARIANT", v);
            }
        }
        if let Err(payload) = result {
            std::panic::resume_unwind(payload);
        }
    }

    fn write_minimal_asset_root(
        dir: &Path,
        with_safetensors_index: bool,
        with_gguf: bool,
    ) -> GemmaAssets {
        // tokenizer.json is required by `discover` — always write a stub.
        fs::write(dir.join("tokenizer.json"), tokenizer_json_with_pad()).unwrap();
        if with_safetensors_index {
            fs::write(dir.join("model-00001-of-00005.safetensors"), b"stub").unwrap();
        }
        if with_gguf {
            fs::write(dir.join("gemma-3-12b-it-q4_0.gguf"), b"stub").unwrap();
        }
        GemmaAssets::discover(dir).unwrap()
    }

    fn tokenizer_json_with_pad() -> &'static str {
        r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "WhitespaceSplit"
  },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<eos>": 7,
      "<pad>": 8,
      "hello": 11,
      "negative": 12
    },
    "unk_token": "<eos>"
  }
}"#
    }

    fn tokenizer_json_without_pad() -> &'static str {
        r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "WhitespaceSplit"
  },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<eos>": 7,
      "hello": 11,
      "negative": 12
    },
    "unk_token": "<eos>"
  }
}"#
    }

    #[test]
    fn packed_i8_tokenizer_is_read_in_place_and_receives_leading_bos() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("gemma4-packed.safetensors");
        let tokenizer =
            tokenizer_json_with_pad().replace("\"<eos>\": 7,", "\"<bos>\": 2, \"<eos>\": 7,");
        let header = serde_json::json!({
            "tokenizer_json": {
                "dtype": "I8",
                "shape": [tokenizer.len()],
                "data_offsets": [0, tokenizer.len()]
            }
        });
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(tokenizer.as_bytes()).unwrap();

        let assets = GemmaAssets::discover(&path).unwrap();
        assert_eq!(assets.packed_weights.as_deref(), Some(path.as_path()));
        let pair = assets
            .encode_prompt_pair_with_max_length("hello", None, 8)
            .unwrap();
        let first_valid = pair
            .conditional
            .attention_mask
            .iter()
            .position(|mask| *mask != 0)
            .unwrap();
        assert_eq!(pair.conditional.input_ids[first_valid], 2);
        assert_eq!(pair.conditional.valid_len(), 2);
    }

    fn write_gemma_assets(
        dir: &Path,
        tokenizer_json: &str,
        special_tokens_json: Option<&str>,
    ) -> GemmaAssets {
        fs::write(dir.join("tokenizer.json"), tokenizer_json).unwrap();
        if let Some(json) = special_tokens_json {
            fs::write(dir.join("special_tokens_map.json"), json).unwrap();
        }
        GemmaAssets::discover(dir).unwrap()
    }

    fn assert_prompt_pair_shape(tokens: &EncodedPromptPair) {
        assert_eq!(tokens.conditional.len(), DEFAULT_GEMMA_MAX_LENGTH);
        assert_eq!(tokens.unconditional.len(), DEFAULT_GEMMA_MAX_LENGTH);
        assert_eq!(tokens.batch_input_ids()[0].len(), DEFAULT_GEMMA_MAX_LENGTH);
        assert_eq!(
            tokens.batch_attention_mask()[1].len(),
            DEFAULT_GEMMA_MAX_LENGTH
        );
    }

    use std::path::Path;

    #[test]
    fn pad_to_alignment_extends_to_multiple_of_eight() {
        let padded = pad_to_alignment(&[1, 2, 3, 4, 5], &[1, 1, 1, 1, 1], 0, 8);
        assert_eq!(padded.input_ids, vec![1, 2, 3, 4, 5, 0, 0, 0]);
        assert_eq!(padded.attention_mask, vec![1, 1, 1, 1, 1, 0, 0, 0]);
    }

    #[test]
    fn left_pad_batch_keeps_valid_tokens_right_aligned() {
        let (ids, masks) = left_pad_batch(&[vec![10, 20], vec![30, 40, 50]], 0);
        assert_eq!(ids, vec![vec![0, 10, 20], vec![30, 40, 50]]);
        assert_eq!(masks, vec![vec![0, 1, 1], vec![1, 1, 1]]);
    }

    #[test]
    fn gemma_assets_encode_prompt_pair_with_fixed_left_padding() {
        let temp_dir = tempfile::tempdir().unwrap();
        let assets = write_gemma_assets(
            temp_dir.path(),
            tokenizer_json_with_pad(),
            Some(r#"{"pad_token":"<pad>","eos_token":"<eos>"}"#),
        );

        let encoded = assets
            .encode_prompt_pair("hello", Some("negative"))
            .unwrap();
        assert_prompt_pair_shape(&encoded);
        assert_eq!(encoded.pad_token_id, 8);
        assert_eq!(encoded.eos_token_id, Some(7));
        assert_eq!(encoded.conditional.valid_len(), 1);
        assert_eq!(encoded.unconditional.valid_len(), 1);
        assert_eq!(
            encoded.conditional.input_ids[DEFAULT_GEMMA_MAX_LENGTH - 1],
            11
        );
        assert_eq!(
            encoded.unconditional.input_ids[DEFAULT_GEMMA_MAX_LENGTH - 1],
            12
        );
        assert_eq!(
            encoded.conditional.attention_mask[DEFAULT_GEMMA_MAX_LENGTH - 1],
            1
        );
        assert!(
            encoded.conditional.attention_mask[..DEFAULT_GEMMA_MAX_LENGTH - 1]
                .iter()
                .all(|value| *value == 0)
        );
    }

    #[test]
    fn gemma_assets_fall_back_to_eos_when_pad_token_is_missing() {
        let temp_dir = tempfile::tempdir().unwrap();
        let assets = write_gemma_assets(
            temp_dir.path(),
            tokenizer_json_without_pad(),
            Some(r#"{"eos_token":{"content":"<eos>"}}"#),
        );

        let encoded = assets.encode_prompt_pair("hello", None).unwrap();
        assert_prompt_pair_shape(&encoded);
        assert_eq!(encoded.pad_token_id, 7);
        assert_eq!(encoded.eos_token_id, Some(7));
        assert_eq!(encoded.unconditional.valid_len(), 0);
        assert!(encoded.unconditional.input_ids.iter().all(|id| *id == 7));
    }

    #[test]
    fn default_gemma_length_matches_upstream_ltx2_contract() {
        assert_eq!(DEFAULT_GEMMA_MAX_LENGTH, 256);
    }

    // ── Variant resolver tests ───────────────────────────────────────────

    #[test]
    fn discover_finds_gguf_when_present() {
        let temp_dir = tempfile::tempdir().unwrap();
        let assets = write_minimal_asset_root(temp_dir.path(), false, true);
        assert!(assets.gguf_path.is_some());
        assert!(assets
            .gguf_path
            .as_ref()
            .unwrap()
            .ends_with("gemma-3-12b-it-q4_0.gguf"));
    }

    #[test]
    fn discover_returns_none_for_gguf_when_absent() {
        let temp_dir = tempfile::tempdir().unwrap();
        let assets = write_minimal_asset_root(temp_dir.path(), true, false);
        assert!(assets.gguf_path.is_none());
        assert!(assets.has_bf16_weights());
    }

    #[test]
    fn discover_finds_namespaced_gguf_lexically_first() {
        let temp_dir = tempfile::tempdir().unwrap();
        // Write two GGUFs; discover should pick the lexically-first one for
        // determinism — users are expected to drop a single GGUF into the root.
        fs::write(
            temp_dir.path().join("tokenizer.json"),
            tokenizer_json_with_pad(),
        )
        .unwrap();
        fs::write(temp_dir.path().join("gemma-3-12b-it-q4_0.gguf"), b"a").unwrap();
        fs::write(temp_dir.path().join("zzz-leftover.gguf"), b"b").unwrap();
        let assets = GemmaAssets::discover(temp_dir.path()).unwrap();
        assert!(assets
            .gguf_path
            .as_ref()
            .unwrap()
            .ends_with("gemma-3-12b-it-q4_0.gguf"));
    }

    #[test]
    fn resolver_picks_q4_when_env_set_and_gguf_present() {
        with_variant_env(Some("q4"), || {
            let temp_dir = tempfile::tempdir().unwrap();
            let assets = write_minimal_asset_root(temp_dir.path(), true, true);
            assert_eq!(
                resolve_gemma_variant(&assets).unwrap(),
                GemmaVariant::Q4Gguf
            );
        });
    }

    #[test]
    fn resolver_errors_when_q4_requested_but_no_gguf_on_disk() {
        with_variant_env(Some("q4"), || {
            let temp_dir = tempfile::tempdir().unwrap();
            let assets = write_minimal_asset_root(temp_dir.path(), true, false);
            let err = resolve_gemma_variant(&assets).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("q4"),
                "error mentions the requested variant: {msg}"
            );
            assert!(
                msg.contains(".gguf"),
                "error mentions the missing file kind: {msg}"
            );
        });
    }

    #[test]
    fn resolver_picks_bf16_when_env_set_and_safetensors_present() {
        with_variant_env(Some("bf16"), || {
            let temp_dir = tempfile::tempdir().unwrap();
            let assets = write_minimal_asset_root(temp_dir.path(), true, true);
            assert_eq!(
                resolve_gemma_variant(&assets).unwrap(),
                GemmaVariant::Bf16Safetensors
            );
        });
    }

    #[test]
    fn resolver_errors_when_bf16_requested_but_no_safetensors_on_disk() {
        with_variant_env(Some("bf16"), || {
            let temp_dir = tempfile::tempdir().unwrap();
            let assets = write_minimal_asset_root(temp_dir.path(), false, true);
            let err = resolve_gemma_variant(&assets).unwrap_err();
            assert!(err.to_string().contains("safetensors"));
        });
    }

    #[test]
    fn resolver_auto_prefers_bf16_when_both_present() {
        with_variant_env(Some("auto"), || {
            let temp_dir = tempfile::tempdir().unwrap();
            let assets = write_minimal_asset_root(temp_dir.path(), true, true);
            assert_eq!(
                resolve_gemma_variant(&assets).unwrap(),
                GemmaVariant::Bf16Safetensors,
                "auto must default to BF16 when both are present — opt into Q4 explicitly"
            );
        });
    }

    #[test]
    fn resolver_auto_picks_gguf_when_only_gguf_present() {
        with_variant_env(None, || {
            let temp_dir = tempfile::tempdir().unwrap();
            let assets = write_minimal_asset_root(temp_dir.path(), false, true);
            assert_eq!(
                resolve_gemma_variant(&assets).unwrap(),
                GemmaVariant::Q4Gguf
            );
        });
    }

    #[test]
    fn resolver_unset_env_falls_through_to_auto() {
        with_variant_env(None, || {
            let temp_dir = tempfile::tempdir().unwrap();
            let assets = write_minimal_asset_root(temp_dir.path(), true, false);
            assert_eq!(
                resolve_gemma_variant(&assets).unwrap(),
                GemmaVariant::Bf16Safetensors
            );
        });
    }

    #[test]
    fn resolver_unrecognised_value_falls_back_to_auto() {
        with_variant_env(Some("nonsense"), || {
            let temp_dir = tempfile::tempdir().unwrap();
            let assets = write_minimal_asset_root(temp_dir.path(), false, true);
            // Auto with only GGUF → Q4. We don't fail hard on unrecognised values
            // because that would lock users out of the assets they just installed.
            assert_eq!(
                resolve_gemma_variant(&assets).unwrap(),
                GemmaVariant::Q4Gguf
            );
        });
    }

    #[test]
    fn resolver_errors_when_no_weights_present() {
        with_variant_env(Some("auto"), || {
            let temp_dir = tempfile::tempdir().unwrap();
            let assets = write_minimal_asset_root(temp_dir.path(), false, false);
            let err = resolve_gemma_variant(&assets).unwrap_err();
            assert!(err.to_string().contains("neither"));
        });
    }
}
