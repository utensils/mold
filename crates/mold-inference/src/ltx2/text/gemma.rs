#![allow(dead_code)]

use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use std::fs;
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
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

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
    pub fn batch_input_ids(&self) -> [&[u32]; 2] {
        [&self.conditional.input_ids, &self.unconditional.input_ids]
    }

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
    pub tokenizer_model: Option<PathBuf>,
    pub special_tokens_map: Option<PathBuf>,
    pub tokenizer_config: Option<PathBuf>,
}

impl GemmaAssets {
    pub fn discover(root: &Path) -> Result<Self> {
        if !root.is_dir() {
            bail!("Gemma asset root '{}' is not a directory", root.display());
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
            tokenizer_model: candidate(root, "tokenizer.model"),
            special_tokens_map: candidate(root, "special_tokens_map.json"),
            tokenizer_config: candidate(root, "tokenizer_config.json"),
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
        let mut tokenizer = Tokenizer::from_file(&self.tokenizer_json).map_err(|err| {
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
            .or_else(|| Some(pad_token))
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
    Ok(PromptTokens {
        input_ids: encoding.get_ids().to_vec(),
        attention_mask: encoding
            .get_attention_mask()
            .iter()
            .map(|value| u8::from(*value != 0))
            .collect(),
    })
}

fn candidate(root: &Path, filename: &str) -> Option<PathBuf> {
    let path = root.join(filename);
    path.is_file().then_some(path)
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
        left_pad_batch, pad_to_alignment, EncodedPromptPair, GemmaAssets, DEFAULT_GEMMA_MAX_LENGTH,
    };
    use std::fs;

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
}
