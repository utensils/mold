use thiserror::Error;
use tokenizers::Tokenizer;

use super::processor::{create_mm_token_type_ids, QwenMmTokenType};

pub const H3_IMAGE_PAD_TOKEN_ID: u32 = 151_655;
pub const H3_VIDEO_PAD_TOKEN_ID: u32 = 151_656;
pub const H3_VISION_START_TOKEN_ID: u32 = 151_652;
pub const H3_VISION_END_TOKEN_ID: u32 = 151_653;

/// MiniMax H3 transformer's row tag. This is intentionally a different type
/// from Qwen's internal text/image/video token discriminator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum H3ModalityTag {
    Vision = 0,
    Text = 1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3Presentation {
    pub token_ids: Vec<u32>,
    pub h3_tags: Vec<H3ModalityTag>,
    pub qwen_mm_types: Vec<QwenMmTokenType>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RefPresentation {
    pub kind: RefPresentationKind,
    /// A pure audio reference and a visual reference carrying a soundtrack
    /// both emit an audio label before any visual label.
    pub has_audio: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RefPresentationKind {
    Audio,
    Image { vision_tokens: usize },
    Video { blocks: Vec<VideoPresentationBlock> },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoPresentationBlock {
    pub timestamp_seconds: f64,
    pub vision_tokens: usize,
}

#[derive(Debug, Error, PartialEq)]
pub enum PresentationError {
    #[error("MiniMax H3 raw tokenization failed: {0}")]
    Tokenizer(String),
    #[error("MiniMax H3 processor token {token} resolved to {found:?}, expected {expected}")]
    SpecialToken {
        token: &'static str,
        found: Option<u32>,
        expected: u32,
    },
    #[error("MiniMax H3 presentation has {actual} tokens, exceeding the {maximum}-token conditioner context")]
    ExcessContext { actual: usize, maximum: usize },
    #[error("unsupported MiniMax H3 reference presentation: {0}")]
    UnsupportedReference(String),
}

pub trait H3RawTokenizer {
    fn encode_raw(&self, text: &str) -> Result<Vec<u32>, PresentationError>;
    fn token_id(&self, token: &str) -> Option<u32>;
}

impl H3RawTokenizer for Tokenizer {
    fn encode_raw(&self, text: &str) -> Result<Vec<u32>, PresentationError> {
        self.encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| PresentationError::Tokenizer(error.to_string()))
    }

    fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id(token)
    }
}

pub fn build_text_presentation(
    tokenizer: &impl H3RawTokenizer,
    prompt: &str,
    maximum_tokens: usize,
) -> Result<H3Presentation, PresentationError> {
    let mut builder = PresentationBuilder::new(tokenizer, maximum_tokens)?;
    builder.text(prompt)?;
    builder.finish()
}

pub fn build_fl2va_presentation(
    tokenizer: &impl H3RawTokenizer,
    prompt: &str,
    keyframe_vision_tokens: &[usize],
    maximum_tokens: usize,
) -> Result<H3Presentation, PresentationError> {
    let mut builder = PresentationBuilder::new(tokenizer, maximum_tokens)?;
    for (index, &vision_tokens) in keyframe_vision_tokens.iter().enumerate() {
        if vision_tokens == 0 {
            return Err(PresentationError::UnsupportedReference(format!(
                "Picture {} has no merged vision tokens",
                index + 1
            )));
        }
        builder.text(&format!("<Picture {}>: ", index + 1))?;
        builder.vision(H3_IMAGE_PAD_TOKEN_ID, vision_tokens);
    }
    builder.text(prompt)?;
    builder.finish()
}

pub fn build_ref2va_presentation(
    tokenizer: &impl H3RawTokenizer,
    prompt: &str,
    references: &[RefPresentation],
    maximum_tokens: usize,
) -> Result<H3Presentation, PresentationError> {
    let mut builder = PresentationBuilder::new(tokenizer, maximum_tokens)?;
    let (mut picture, mut video, mut audio) = (0, 0, 0);
    for reference in references {
        if reference.has_audio {
            audio += 1;
            builder.text(&format!("<Audio {audio}>: "))?;
        }
        match &reference.kind {
            RefPresentationKind::Audio => {
                if !reference.has_audio {
                    return Err(PresentationError::UnsupportedReference(
                        "an audio reference must set has_audio".into(),
                    ));
                }
            }
            RefPresentationKind::Image { vision_tokens } => {
                if *vision_tokens == 0 {
                    return Err(PresentationError::UnsupportedReference(
                        "an image reference has no merged vision tokens".into(),
                    ));
                }
                picture += 1;
                builder.text(&format!("<Picture {picture}>: "))?;
                builder.vision(H3_IMAGE_PAD_TOKEN_ID, *vision_tokens);
            }
            RefPresentationKind::Video { blocks } => {
                if blocks.is_empty() {
                    return Err(PresentationError::UnsupportedReference(
                        "a video reference has no temporal vision blocks".into(),
                    ));
                }
                video += 1;
                builder.text(&format!("<Video {video}>: "))?;
                for block in blocks {
                    if !block.timestamp_seconds.is_finite() || block.timestamp_seconds < 0.0 {
                        return Err(PresentationError::UnsupportedReference(
                            "a video block timestamp is not finite and non-negative".into(),
                        ));
                    }
                    if block.vision_tokens == 0 {
                        return Err(PresentationError::UnsupportedReference(
                            "a video block has no merged vision tokens".into(),
                        ));
                    }
                    builder.text(&format!("<{:.1} seconds>", block.timestamp_seconds))?;
                    builder.vision(H3_VIDEO_PAD_TOKEN_ID, block.vision_tokens);
                }
            }
        }
    }
    builder.text(prompt)?;
    builder.finish()
}

struct PresentationBuilder<'a, T> {
    tokenizer: &'a T,
    maximum_tokens: usize,
    token_ids: Vec<u32>,
    h3_tags: Vec<H3ModalityTag>,
}

impl<'a, T: H3RawTokenizer> PresentationBuilder<'a, T> {
    fn new(tokenizer: &'a T, maximum_tokens: usize) -> Result<Self, PresentationError> {
        for (token, expected) in [
            ("<|vision_start|>", H3_VISION_START_TOKEN_ID),
            ("<|vision_end|>", H3_VISION_END_TOKEN_ID),
            ("<|image_pad|>", H3_IMAGE_PAD_TOKEN_ID),
            ("<|video_pad|>", H3_VIDEO_PAD_TOKEN_ID),
        ] {
            let found = tokenizer.token_id(token);
            if found != Some(expected) {
                return Err(PresentationError::SpecialToken {
                    token,
                    found,
                    expected,
                });
            }
        }
        Ok(Self {
            tokenizer,
            maximum_tokens,
            token_ids: Vec::new(),
            h3_tags: Vec::new(),
        })
    }

    fn text(&mut self, text: &str) -> Result<(), PresentationError> {
        let ids = self.tokenizer.encode_raw(text)?;
        self.h3_tags
            .extend(std::iter::repeat_n(H3ModalityTag::Text, ids.len()));
        self.token_ids.extend(ids);
        Ok(())
    }

    fn vision(&mut self, pad_token: u32, count: usize) {
        self.token_ids.push(H3_VISION_START_TOKEN_ID);
        self.token_ids.extend(std::iter::repeat_n(pad_token, count));
        self.token_ids.push(H3_VISION_END_TOKEN_ID);
        self.h3_tags
            .extend(std::iter::repeat_n(H3ModalityTag::Vision, count + 2));
    }

    fn finish(self) -> Result<H3Presentation, PresentationError> {
        if self.token_ids.len() > self.maximum_tokens {
            return Err(PresentationError::ExcessContext {
                actual: self.token_ids.len(),
                maximum: self.maximum_tokens,
            });
        }
        let qwen_mm_types = create_mm_token_type_ids(&self.token_ids);
        debug_assert_eq!(self.token_ids.len(), self.h3_tags.len());
        Ok(H3Presentation {
            token_ids: self.token_ids,
            h3_tags: self.h3_tags,
            qwen_mm_types,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    struct RecordingTokenizer {
        pieces: BTreeMap<String, u32>,
    }

    impl RecordingTokenizer {
        fn new() -> Self {
            Self {
                pieces: BTreeMap::from([
                    ("<|vision_start|>".into(), H3_VISION_START_TOKEN_ID),
                    ("<|vision_end|>".into(), H3_VISION_END_TOKEN_ID),
                    ("<|image_pad|>".into(), H3_IMAGE_PAD_TOKEN_ID),
                    ("<|video_pad|>".into(), H3_VIDEO_PAD_TOKEN_ID),
                ]),
            }
        }
    }

    impl H3RawTokenizer for RecordingTokenizer {
        fn encode_raw(&self, text: &str) -> Result<Vec<u32>, PresentationError> {
            // A stable one-token recording of each exact raw segment is enough
            // to prove ordering without teaching the test a second BPE.
            if text.is_empty() {
                return Ok(Vec::new());
            }
            Ok(vec![text.bytes().fold(1_u32, |hash, byte| {
                hash.wrapping_mul(16_777_619).wrapping_add(u32::from(byte))
            })])
        }

        fn token_id(&self, token: &str) -> Option<u32> {
            self.pieces.get(token).copied()
        }
    }

    #[test]
    fn text_prompt_is_raw_and_empty_stays_empty() {
        let tokenizer = RecordingTokenizer::new();
        let presentation = build_text_presentation(&tokenizer, "raw prompt", 20).unwrap();
        assert_eq!(
            presentation.token_ids,
            tokenizer.encode_raw("raw prompt").unwrap()
        );
        let empty = build_text_presentation(&tokenizer, "", 20).unwrap();
        assert!(empty.token_ids.is_empty());
        assert!(!empty.token_ids.contains(&151_643));
    }

    #[test]
    fn mixed_references_keep_authoritative_order_and_separate_tag_types() {
        let tokenizer = RecordingTokenizer::new();
        let references = vec![
            RefPresentation {
                kind: RefPresentationKind::Video {
                    blocks: vec![VideoPresentationBlock {
                        timestamp_seconds: 0.25,
                        vision_tokens: 2,
                    }],
                },
                has_audio: true,
            },
            RefPresentation {
                kind: RefPresentationKind::Image { vision_tokens: 1 },
                has_audio: false,
            },
            RefPresentation {
                kind: RefPresentationKind::Audio,
                has_audio: true,
            },
        ];
        let result = build_ref2va_presentation(&tokenizer, "prompt", &references, 100).unwrap();
        let expected_segments = [
            "<Audio 1>: ",
            "<Video 1>: ",
            "<0.2 seconds>",
            "<Picture 1>: ",
            "<Audio 2>: ",
            "prompt",
        ];
        let observed_text: Vec<_> = result
            .token_ids
            .iter()
            .copied()
            .filter(|id| {
                ![
                    H3_VISION_START_TOKEN_ID,
                    H3_VISION_END_TOKEN_ID,
                    H3_IMAGE_PAD_TOKEN_ID,
                    H3_VIDEO_PAD_TOKEN_ID,
                ]
                .contains(id)
            })
            .collect();
        assert_eq!(
            observed_text,
            expected_segments
                .iter()
                .flat_map(|segment| tokenizer.encode_raw(segment).unwrap())
                .collect::<Vec<_>>()
        );
        let image_pad = result
            .token_ids
            .iter()
            .position(|id| *id == H3_IMAGE_PAD_TOKEN_ID)
            .unwrap();
        assert_eq!(result.h3_tags[image_pad], H3ModalityTag::Vision);
        assert_eq!(result.qwen_mm_types[image_pad], QwenMmTokenType::Image);
        let image_start = image_pad - 1;
        assert_eq!(result.h3_tags[image_start], H3ModalityTag::Vision);
        assert_eq!(result.qwen_mm_types[image_start], QwenMmTokenType::Text);
    }

    #[test]
    fn excess_context_and_unsupported_reference_fail_before_encoding() {
        let tokenizer = RecordingTokenizer::new();
        assert!(matches!(
            build_fl2va_presentation(&tokenizer, "prompt", &[1], 3),
            Err(PresentationError::ExcessContext { .. })
        ));
        assert!(matches!(
            build_ref2va_presentation(
                &tokenizer,
                "prompt",
                &[RefPresentation {
                    kind: RefPresentationKind::Video { blocks: vec![] },
                    has_audio: false,
                }],
                100,
            ),
            Err(PresentationError::UnsupportedReference(_))
        ));
    }
}
