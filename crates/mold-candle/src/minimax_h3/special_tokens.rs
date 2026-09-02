//! The MiniMax extra special tokens that the released `tokenizer.json` omits.
//!
//! H3's official `processor/tokenizer_config.json` declares twenty
//! `additional_special_tokens`. Thirteen of them already carry ids in
//! `tokenizer.json`; the remaining seven — the dialogue, lyric and caption
//! delimiters — do not. The released `tokenizer.json` stops at id 151668, and
//! HuggingFace assigns the missing seven 151669..=151675 in declaration order
//! when the tokenizer is built from the directory. Building from
//! `tokenizer.json` alone skips that step.
//!
//! The cost of skipping it is not a missing niceness. Byte-level BPE smears an
//! unregistered `<d>` into its neighbours — `>` merges rightward with the `[`
//! of `[English]`, and `</d>` merges leftward with the preceding period — so no
//! token marks where dialogue begins and the delimiter is not even recoverable
//! downstream. MiniMax's own reference request writes
//! `<d>[English] Follow the wind, live free.</d>`, and the official README
//! states that "we add several special tokens, such as `<d>`, to the tokenizer
//! configuration \[…\] the tokenizer and associated configuration files
//! provided in the H3 repository are required".
//!
//! Note that mold's H3 port otherwise tracks ComfyUI, which does *not* register
//! these tokens. This is a deliberate, documented divergence in favour of the
//! model authors' own contract; see `.claude/rules/minimax-h3.md`.
//!
//! Refs: issue #1430.

use std::error::Error;
use std::fmt;

use tokenizers::{AddedToken, Tokenizer};

/// The seven configured special tokens absent from the released
/// `tokenizer.json`, with the ids HuggingFace assigns them, in declaration
/// order.
pub const H3_EXTRA_SPECIAL_TOKENS: [(&str, u32); 7] = [
    ("<d>", 151_669),
    ("</d>", 151_670),
    ("<|cutoff|>", 151_671),
    ("<|lyrics_start|>", 151_672),
    ("<|lyrics_end|>", 151_673),
    ("<|caption_start|>", 151_674),
    ("<|caption_end|>", 151_675),
];

/// Base vocabulary of the released `tokenizer.json`, before added tokens.
pub const H3_BASE_VOCABULARY_SIZE: usize = 151_643;
/// Added tokens carried by the released `tokenizer.json` itself.
pub const H3_RELEASED_ADDED_TOKEN_COUNT: usize = 26;
/// Total entries in the released `tokenizer.json`, as loaded from the file.
pub const H3_RELEASED_VOCABULARY_SIZE: usize =
    H3_BASE_VOCABULARY_SIZE + H3_RELEASED_ADDED_TOKEN_COUNT;
/// Highest id present in the released `tokenizer.json`.
pub const H3_RELEASED_MAX_TOKEN_ID: u32 = 151_668;

/// Added tokens once the configured extras are registered.
pub const H3_REGISTERED_ADDED_TOKEN_COUNT: usize =
    H3_RELEASED_ADDED_TOKEN_COUNT + H3_EXTRA_SPECIAL_TOKENS.len();
/// Total entries once the configured extras are registered.
pub const H3_REGISTERED_VOCABULARY_SIZE: usize =
    H3_BASE_VOCABULARY_SIZE + H3_REGISTERED_ADDED_TOKEN_COUNT;
/// Highest id once the configured extras are registered. Stays far below the
/// checkpoint's 151936-row embedding table.
pub const H3_REGISTERED_MAX_TOKEN_ID: u32 = 151_675;

/// Why the configured special tokens could not be registered. Every variant is
/// a refusal to load: a tokenizer whose extra tokens land anywhere but their
/// released ids would silently mis-condition every dialogue prompt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum H3SpecialTokenError {
    /// `additional_special_tokens` is absent or not an array of strings.
    Config(String),
    /// The configured tokens missing from the vocabulary are not exactly the
    /// seven expected ones, in order.
    UnexpectedSet { found: Vec<String> },
    /// A token did not land on its released id after registration.
    UnexpectedId {
        token: &'static str,
        found: Option<u32>,
        expected: u32,
    },
}

impl fmt::Display for H3SpecialTokenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(detail) => {
                write!(formatter, "tokenizer config special tokens: {detail}")
            }
            Self::UnexpectedSet { found } => write!(
                formatter,
                "tokenizer config declares unregistered special tokens {found:?}, expected exactly {:?}",
                H3_EXTRA_SPECIAL_TOKENS.map(|(token, _)| token)
            ),
            Self::UnexpectedId {
                token,
                found,
                expected,
            } => write!(
                formatter,
                "special token {token} registered as {found:?}, expected {expected}"
            ),
        }
    }
}

impl Error for H3SpecialTokenError {}

/// Register the configured special tokens the released `tokenizer.json` omits.
///
/// Reads `additional_special_tokens` from the pinned `tokenizer_config.json`
/// bytes the caller already holds — no new file is opened — and adds only the
/// entries the vocabulary does not already resolve, in declaration order, so
/// the ids match HuggingFace's own assignment byte for byte.
///
/// Fails closed. If the pinned config ever declares a different set, this
/// refuses rather than registering whatever it happens to find, and it verifies
/// every resulting id before returning.
pub fn register_extra_special_tokens(
    tokenizer: &mut Tokenizer,
    tokenizer_config: &[u8],
) -> Result<(), H3SpecialTokenError> {
    let value: serde_json::Value = serde_json::from_slice(tokenizer_config)
        .map_err(|error| H3SpecialTokenError::Config(error.to_string()))?;
    let declared = value
        .get("additional_special_tokens")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            H3SpecialTokenError::Config(
                "additional_special_tokens is absent or not an array".into(),
            )
        })?;

    let mut missing = Vec::new();
    for entry in declared {
        let content = entry.as_str().ok_or_else(|| {
            H3SpecialTokenError::Config("additional_special_tokens holds a non-string".into())
        })?;
        if tokenizer.token_to_id(content).is_none() {
            missing.push(content.to_owned());
        }
    }

    let expected = H3_EXTRA_SPECIAL_TOKENS.map(|(token, _)| token);
    if missing != expected {
        return Err(H3SpecialTokenError::UnexpectedSet { found: missing });
    }

    let added = missing
        .iter()
        .map(|content| AddedToken::from(content.clone(), true))
        .collect::<Vec<_>>();
    tokenizer.add_special_tokens(&added);

    for (token, expected_id) in H3_EXTRA_SPECIAL_TOKENS {
        let found = tokenizer.token_to_id(token);
        if found != Some(expected_id) {
            return Err(H3SpecialTokenError::UnexpectedId {
                token,
                found,
                expected: expected_id,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The thirteen configured tokens that the released `tokenizer.json`
    /// already carries, plus the ids it assigns them.
    const PRESENT: [(&str, u32); 13] = [
        ("<|im_start|>", 151_644),
        ("<|im_end|>", 151_645),
        ("<|object_ref_start|>", 151_646),
        ("<|object_ref_end|>", 151_647),
        ("<|box_start|>", 151_648),
        ("<|box_end|>", 151_649),
        ("<|quad_start|>", 151_650),
        ("<|quad_end|>", 151_651),
        ("<|vision_start|>", 151_652),
        ("<|vision_end|>", 151_653),
        ("<|vision_pad|>", 151_654),
        ("<|image_pad|>", 151_655),
        ("<|video_pad|>", 151_656),
    ];

    fn config_with(tokens: &[&str]) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({ "additional_special_tokens": tokens })).unwrap()
    }

    fn released_config() -> Vec<u8> {
        let mut tokens = PRESENT.iter().map(|(t, _)| *t).collect::<Vec<_>>();
        tokens.extend(H3_EXTRA_SPECIAL_TOKENS.iter().map(|(t, _)| *t));
        config_with(&tokens)
    }

    /// A stand-in for the released `tokenizer.json`: a word-level vocabulary of
    /// the same size, carrying the same 26 added tokens up to 151668.
    fn released_shape_tokenizer() -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        for id in 0..u32::try_from(H3_BASE_VOCABULARY_SIZE).unwrap() {
            vocab.insert(format!("token-{id}"), id.into());
        }
        let added = (u32::try_from(H3_BASE_VOCABULARY_SIZE).unwrap()..=H3_RELEASED_MAX_TOKEN_ID)
            .map(|id| {
                let content = PRESENT
                    .iter()
                    .find(|(_, present_id)| *present_id == id)
                    .map_or_else(
                        || format!("<|added_{id}|>"),
                        |(token, _)| (*token).to_owned(),
                    );
                serde_json::json!({
                    "id": id, "content": content, "single_word": false,
                    "lstrip": false, "rstrip": false, "normalized": false, "special": true
                })
            })
            .collect::<Vec<_>>();
        let bytes = serde_json::to_vec(&serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": added, "normalizer": null, "pre_tokenizer": null,
            "post_processor": null, "decoder": null,
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "token-0" }
        }))
        .unwrap();
        Tokenizer::from_bytes(bytes).unwrap()
    }

    #[test]
    fn released_config_registers_the_seven_in_declaration_order() {
        let mut tokenizer = released_shape_tokenizer();
        assert_eq!(tokenizer.get_vocab_size(true), H3_RELEASED_VOCABULARY_SIZE);

        register_extra_special_tokens(&mut tokenizer, &released_config()).unwrap();

        assert_eq!(
            tokenizer.get_vocab_size(true),
            H3_REGISTERED_VOCABULARY_SIZE
        );
        assert_eq!(
            tokenizer.get_vocab(true).values().copied().max(),
            Some(H3_REGISTERED_MAX_TOKEN_ID)
        );
        for (token, expected) in H3_EXTRA_SPECIAL_TOKENS {
            assert_eq!(tokenizer.token_to_id(token), Some(expected));
        }
        // Registration must not disturb the ids the file already carried.
        for (token, expected) in PRESENT {
            assert_eq!(tokenizer.token_to_id(token), Some(expected));
        }
    }

    /// The presentation encodes with `add_special_tokens = false`. That flag
    /// governs BOS/EOS wrapping only — the added-vocabulary trie still matches
    /// registered tokens inside ordinary text. The whole fix rests on this.
    #[test]
    fn registered_tags_collapse_to_one_id_when_encoding_without_special_tokens() {
        let mut tokenizer = released_shape_tokenizer();
        let before = tokenizer.encode("token-1 <d> token-2", false).unwrap();
        assert!(!before.get_ids().contains(&H3_EXTRA_SPECIAL_TOKENS[0].1));

        register_extra_special_tokens(&mut tokenizer, &released_config()).unwrap();

        let after = tokenizer.encode("token-1 <d> token-2", false).unwrap();
        assert!(
            after.get_ids().contains(&151_669),
            "the tag must survive as its own id, got {:?}",
            after.get_ids()
        );
        assert_eq!(
            after
                .get_tokens()
                .iter()
                .filter(|token| *token == "<d>")
                .count(),
            1,
            "the tag must be exactly one token, not a run of pieces"
        );
    }

    #[test]
    fn a_config_missing_the_extras_is_refused() {
        let mut tokenizer = released_shape_tokenizer();
        let tokens = PRESENT.iter().map(|(t, _)| *t).collect::<Vec<_>>();
        let error =
            register_extra_special_tokens(&mut tokenizer, &config_with(&tokens)).unwrap_err();
        assert!(matches!(
            error,
            H3SpecialTokenError::UnexpectedSet { ref found } if found.is_empty()
        ));
    }

    #[test]
    fn an_unexpected_extra_token_is_refused_before_registration() {
        let mut tokenizer = released_shape_tokenizer();
        let mut tokens = PRESENT.iter().map(|(t, _)| *t).collect::<Vec<_>>();
        tokens.push("<|surprise|>");
        tokens.extend(H3_EXTRA_SPECIAL_TOKENS.iter().map(|(t, _)| *t));
        let error =
            register_extra_special_tokens(&mut tokenizer, &config_with(&tokens)).unwrap_err();
        assert!(matches!(error, H3SpecialTokenError::UnexpectedSet { .. }));
        assert_eq!(tokenizer.get_vocab_size(true), H3_RELEASED_VOCABULARY_SIZE);
    }

    #[test]
    fn an_already_registered_tokenizer_is_refused_rather_than_double_registered() {
        let mut tokenizer = released_shape_tokenizer();
        register_extra_special_tokens(&mut tokenizer, &released_config()).unwrap();
        let error = register_extra_special_tokens(&mut tokenizer, &released_config()).unwrap_err();
        assert!(matches!(
            error,
            H3SpecialTokenError::UnexpectedSet { ref found } if found.is_empty()
        ));
        assert_eq!(
            tokenizer.get_vocab_size(true),
            H3_REGISTERED_VOCABULARY_SIZE
        );
    }

    #[test]
    fn a_malformed_config_is_refused() {
        let mut tokenizer = released_shape_tokenizer();
        assert!(matches!(
            register_extra_special_tokens(&mut tokenizer, b"not json").unwrap_err(),
            H3SpecialTokenError::Config(_)
        ));
        assert!(matches!(
            register_extra_special_tokens(&mut tokenizer, b"{}").unwrap_err(),
            H3SpecialTokenError::Config(_)
        ));
        assert!(matches!(
            register_extra_special_tokens(&mut tokenizer, br#"{"additional_special_tokens":[7]}"#)
                .unwrap_err(),
            H3SpecialTokenError::Config(_)
        ));
    }

    /// Compile-time tripwires on the released numbers. These are the values the
    /// checkpoint was published with, so a silent edit is a correctness bug,
    /// not a preference -- fail the build rather than a test run.
    const _: () = {
        const EMBEDDING_ROWS: u32 = 151_936;
        assert!(H3_REGISTERED_MAX_TOKEN_ID < EMBEDDING_ROWS);
        assert!(H3_REGISTERED_VOCABULARY_SIZE == H3_REGISTERED_MAX_TOKEN_ID as usize + 1);
        assert!(H3_RELEASED_VOCABULARY_SIZE == H3_RELEASED_MAX_TOKEN_ID as usize + 1);
        assert!(H3_REGISTERED_ADDED_TOKEN_COUNT == H3_RELEASED_ADDED_TOKEN_COUNT + 7);
    };
}
