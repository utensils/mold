//! Qwen Image 2.1's joint text/image sequence and its block-causal attention
//! plan.
//!
//! Upstream builds the joint sequence inside the transformer's forward
//! (diffusers `e0abab83b` `transformer_qwenimage21.py`, cited `T:`): every
//! `<|image_pad|>` slot of the Qwen3-VL sequence is repeated four times and the
//! repeated positions are OVERWRITTEN with `img_in(latents)` in raster order
//! (`T:905-923`), the target image's slots are appended last
//! (`pipeline_qwenimage21.py:740-745`), and RoPE, the block-causal mask and the
//! prefix/target split are all derived from that expanded mask. Here the same
//! facts are computed ONCE per request branch on the host, as plain data:
//!
//! - which joint position reads which row (`gather_index`, one `index_select`
//!   over `cat[txt_in(text), img_in(cond…, target)]`);
//! - the 3-axis RoPE coordinates (`T:652-710`);
//! - the key-padding mask (`T:939-949`); and
//! - the attention [`BlockCausalPlan`] — the one seam every attention backend
//!   dispatches through (`T:309-324`, `T:462-545`).
//!
//! Text-to-image is the special case with no condition images, so the old
//! `t2i_rope` and text/target split fall out of the same builder.

use std::ops::Range;
use std::sync::Arc;

use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};

/// Each vision-language image slot stands for a 2x2 group of latent tokens
/// (`T:38-39`, `_IMG_TOKENS_PER_SLOT`).
pub(crate) const IMG_TOKENS_PER_SLOT: usize = 4;

/// Per-batch-row key validity over the joint sequence. `None` wherever it is
/// held means no row has a padded key.
pub(crate) type KeyValid = Arc<[Vec<bool>]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SegmentKind {
    /// A run of Qwen3-VL text rows (including right padding).
    Text,
    /// The `index`-th condition image, in caller order.
    ConditionImage { index: usize },
    /// The image being denoised. Always the last segment.
    Target,
}

/// One contiguous run of the joint sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct JointSegment {
    pub kind: SegmentKind,
    /// First joint position.
    pub start: usize,
    pub len: usize,
    /// Text: the rows of the Qwen3-VL embeddings it reads. Image: the rows of
    /// the packed `[cond…, target]` latents it reads.
    pub source: Range<usize>,
    /// Latent `(height, width)` of an image segment.
    pub latent_hw: Option<(usize, usize)>,
}

impl JointSegment {
    pub(crate) fn end(&self) -> usize {
        self.start + self.len
    }
}

/// How a query segment sees its keys `[0, kv_len)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SegmentMask {
    /// Text: query row `i` sees keys `[0, kv_len - q_len + i]` — causal inside
    /// its own segment, everything before it. This is FlashAttention-2's
    /// causal mask with bottom-right alignment when `q_len != kv_len`.
    CausalBottomRight,
    /// Image blocks (condition or target): every key in `[0, kv_len)`,
    /// bidirectional inside the block.
    Full,
}

/// One attention call: queries `[q_start, q_start + q_len)` of the query
/// tensor against keys and values `[0, kv_len)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AttentionSegment {
    pub q_start: usize,
    pub q_len: usize,
    pub kv_len: usize,
    pub mask: SegmentMask,
}

/// The block-causal mask `(q >= kv) or same_image_block`, with padded keys
/// excluded (`T:257-306`), decomposed exactly into per-segment calls the way
/// upstream's default `QwenImage21AttnProcessor` runs it (`T:510-545`).
///
/// This is the ONE seam every attention backend plugs into: the math path
/// builds per-segment biases, Metal's fused SDPA takes unbiased `Full`
/// segments, and a flash backend maps `CausalBottomRight` to its windowed
/// causal kernel and padded rows to varlen.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BlockCausalPlan {
    pub segments: Vec<AttentionSegment>,
    pub key_valid: Option<KeyValid>,
}

impl BlockCausalPlan {
    /// Keys valid for `rows` of a segment that sees `[0, kv_len)`, or `None`
    /// when no row has a padded key in that range.
    pub(crate) fn key_valid_prefix(&self, kv_len: usize) -> Option<Vec<&[bool]>> {
        let key_valid = self.key_valid.as_ref()?;
        let rows: Vec<&[bool]> = key_valid.iter().map(|row| &row[..kv_len]).collect();
        rows.iter()
            .any(|row| row.iter().any(|valid| !valid))
            .then_some(rows)
    }
}

/// The resolved joint layout of one conditioning branch.
#[derive(Debug, Clone)]
pub(crate) struct QwenImage21JointLayout {
    segments: Vec<JointSegment>,
    total_len: usize,
    prefix_len: usize,
    text_len: usize,
    condition_tokens: usize,
    key_valid: Option<KeyValid>,
    rope: Vec<[i32; 3]>,
    gather_index: Vec<u32>,
}

impl QwenImage21JointLayout {
    /// Build the layout of one branch.
    ///
    /// - `image_slots`: the trimmed Qwen3-VL `image_pad_mask` (one entry per
    ///   text row). Every batch row shares one layout (`T:911-912`), so this
    ///   is one row.
    /// - `valid`: per batch row, which text rows are real tokens (right
    ///   padding is `false`).
    /// - `condition`: latent `(height, width)` of each condition image, in
    ///   caller order.
    /// - `target`: latent `(height, width)` of the image being denoised.
    ///
    /// Block boundaries come from the SHAPES, not from runs of `true`
    /// (`T:808-850`): two adjacent condition images stay two blocks. The slot
    /// count must equal `Σ h·w / 4` over the condition images, and each image
    /// block must be one contiguous run of slots.
    pub(crate) fn build(
        image_slots: &[bool],
        valid: &[Vec<bool>],
        condition: &[(usize, usize)],
        target: (usize, usize),
    ) -> Result<Self> {
        let text_len = image_slots.len();
        ensure!(text_len > 0, "Qwen Image 2.1 text conditioning is empty");
        ensure!(
            !valid.is_empty()
                && valid
                    .iter()
                    .all(|row| row.len() == text_len && row.first() == Some(&true)),
            "Qwen Image 2.1 requires a right-padded text mask with a valid first token"
        );
        for row in valid {
            let real = row.iter().take_while(|value| **value).count();
            ensure!(
                row[real..].iter().all(|value| !value),
                "Qwen Image 2.1 text mask must be right-padded"
            );
            ensure!(
                image_slots
                    .iter()
                    .zip(row)
                    .all(|(slot, valid)| !slot || *valid),
                "Qwen Image 2.1 image slots cannot be padding"
            );
        }
        let (target_h, target_w) = target;
        ensure!(
            target_h > 0 && target_w > 0,
            "Qwen Image 2.1 latent dimensions must be positive"
        );
        let target_tokens = target_h * target_w;
        ensure!(
            target_tokens.is_multiple_of(IMG_TOKENS_PER_SLOT),
            "Qwen Image 2.1 target latent {target_h}x{target_w} does not tile 2x2 slots"
        );
        let mut condition_tokens = 0usize;
        for (index, &(height, width)) in condition.iter().enumerate() {
            ensure!(
                height > 0 && width > 0 && (height * width).is_multiple_of(IMG_TOKENS_PER_SLOT),
                "Qwen Image 2.1 condition image {index} latent {height}x{width} does not tile 2x2 slots"
            );
            condition_tokens += height * width;
        }
        let slots = image_slots.iter().filter(|slot| **slot).count();
        ensure!(
            slots * IMG_TOKENS_PER_SLOT == condition_tokens,
            "Qwen Image 2.1 image slots account for {} latent tokens but the condition images have {condition_tokens}",
            slots * IMG_TOKENS_PER_SLOT
        );

        // Walk the Qwen3-VL rows, expanding image slots four-fold, and cut the
        // image runs into blocks by the shapes.
        let mut segments = Vec::new();
        let mut gather_index = Vec::with_capacity(text_len + condition_tokens + target_tokens);
        let mut text_rows: Vec<usize> = Vec::new();
        let mut joint = 0usize;
        let mut row = 0usize;
        let mut latent_row = 0usize;
        let latent_source = |latent_row: usize| (text_len + latent_row) as u32;
        let flush_text =
            |segments: &mut Vec<JointSegment>, text_rows: &mut Vec<usize>, joint: usize| {
                if let (Some(&first), Some(&last)) = (text_rows.first(), text_rows.last()) {
                    segments.push(JointSegment {
                        kind: SegmentKind::Text,
                        start: joint - text_rows.len(),
                        len: text_rows.len(),
                        source: first..last + 1,
                        latent_hw: None,
                    });
                    text_rows.clear();
                }
            };
        for (index, &(height, width)) in condition.iter().enumerate() {
            let block_slots = height * width / IMG_TOKENS_PER_SLOT;
            // Text rows up to this block's first slot.
            while row < text_len && !image_slots[row] {
                gather_index.push(row as u32);
                text_rows.push(row);
                joint += 1;
                row += 1;
            }
            flush_text(&mut segments, &mut text_rows, joint);
            ensure!(
                row + block_slots <= text_len && image_slots[row..row + block_slots].iter().all(|slot| *slot),
                "Qwen Image 2.1 condition image {index} ({height}x{width} latents) is not one contiguous run of {block_slots} image slots"
            );
            let tokens = height * width;
            gather_index.extend((latent_row..latent_row + tokens).map(latent_source));
            segments.push(JointSegment {
                kind: SegmentKind::ConditionImage { index },
                start: joint,
                len: tokens,
                source: latent_row..latent_row + tokens,
                latent_hw: Some((height, width)),
            });
            joint += tokens;
            latent_row += tokens;
            row += block_slots;
        }
        while row < text_len {
            ensure!(
                !image_slots[row],
                "Qwen Image 2.1 has image slots after the last condition image"
            );
            gather_index.push(row as u32);
            text_rows.push(row);
            joint += 1;
            row += 1;
        }
        flush_text(&mut segments, &mut text_rows, joint);
        let prefix_len = joint;
        gather_index.extend((latent_row..latent_row + target_tokens).map(latent_source));
        segments.push(JointSegment {
            kind: SegmentKind::Target,
            start: joint,
            len: target_tokens,
            source: latent_row..latent_row + target_tokens,
            latent_hw: Some(target),
        });
        let total_len = joint + target_tokens;

        let rope = Self::rope_coordinates(&segments, total_len);
        let key_valid = if valid.iter().all(|row| row.iter().all(|value| *value)) {
            None
        } else {
            // Image positions are always valid keys; text positions take their
            // row's bit (`T:939-949`).
            let rows: Vec<Vec<bool>> = valid
                .iter()
                .map(|row_valid| {
                    let mut joint_valid = vec![true; total_len];
                    for segment in &segments {
                        if segment.kind == SegmentKind::Text {
                            joint_valid[segment.start..segment.end()]
                                .copy_from_slice(&row_valid[segment.source.clone()]);
                        }
                    }
                    joint_valid
                })
                .collect();
            Some(rows.into())
        };
        Ok(Self {
            segments,
            total_len,
            prefix_len,
            text_len,
            condition_tokens,
            key_valid,
            rope,
            gather_index,
        })
    }

    /// Text-to-image: no condition images, so the joint sequence is the text
    /// rows followed by the target block.
    pub(crate) fn text_to_image(valid: &[Vec<bool>], target: (usize, usize)) -> Result<Self> {
        let text_len = valid.first().map_or(0, Vec::len);
        Self::build(&vec![false; text_len], valid, &[], target)
    }

    /// 3-axis RoPE coordinates (`QwenImage21Rope.forward`, `T:677-710`).
    ///
    /// Text advances one shared position on every axis. Each image block
    /// freezes the frame axis at the running position, lays its tokens out on
    /// an h/w grid centred on zero (`range(-(H - H//2), H//2)`), and then
    /// advances the position by `max(h, w)`.
    fn rope_coordinates(segments: &[JointSegment], total_len: usize) -> Vec<[i32; 3]> {
        let mut coords = Vec::with_capacity(total_len);
        let mut position = 0i32;
        for segment in segments {
            match segment.latent_hw {
                None => {
                    for _ in 0..segment.len {
                        coords.push([position, position, position]);
                        position += 1;
                    }
                }
                Some((height, width)) => {
                    let (height, width) = (height as i32, width as i32);
                    let h_start = -(height - height / 2);
                    let w_start = -(width - width / 2);
                    for h in 0..height {
                        for w in 0..width {
                            coords.push([position, h_start + h, w_start + w]);
                        }
                    }
                    position += height.max(width);
                }
            }
        }
        coords
    }

    #[cfg(test)]
    pub(crate) fn segments(&self) -> &[JointSegment] {
        &self.segments
    }

    #[cfg(test)]
    pub(crate) fn total_len(&self) -> usize {
        self.total_len
    }

    /// Every non-target position: text plus condition images. These rows
    /// modulate from t=0 and are what the prefix KV cache retains.
    pub(crate) fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    /// Rows of the Qwen3-VL embeddings this layout reads from.
    pub(crate) fn text_len(&self) -> usize {
        self.text_len
    }

    pub(crate) fn condition_tokens(&self) -> usize {
        self.condition_tokens
    }

    pub(crate) fn target_tokens(&self) -> usize {
        self.total_len - self.prefix_len
    }

    #[cfg(test)]
    pub(crate) fn key_valid(&self) -> Option<&KeyValid> {
        self.key_valid.as_ref()
    }

    pub(crate) fn rope(&self) -> &[[i32; 3]] {
        &self.rope
    }

    /// `index_select` rows over `cat[txt_in(text), img_in(cond…, target)]`.
    pub(crate) fn gather_index(&self) -> &[u32] {
        &self.gather_index
    }

    /// Whether the joint sequence is simply `[text rows; target]`, i.e. the
    /// gather is the identity and a plain concatenation assembles it.
    pub(crate) fn gather_is_identity(&self) -> bool {
        self.gather_index
            .iter()
            .enumerate()
            .all(|(position, &source)| source as usize == position)
    }

    /// The attention plan for a full (uncached) forward or for a cached step
    /// whose queries are the target block alone.
    pub(crate) fn attention_plan(&self, cached: bool) -> BlockCausalPlan {
        let segments = if cached {
            // Target rows see the entire prefix and their own block, so the
            // block-causal mask degenerates to full attention (`T:953-962`).
            vec![AttentionSegment {
                q_start: 0,
                q_len: self.target_tokens(),
                kv_len: self.total_len,
                mask: SegmentMask::Full,
            }]
        } else {
            self.segments
                .iter()
                .map(|segment| AttentionSegment {
                    q_start: segment.start,
                    q_len: segment.len,
                    kv_len: segment.end(),
                    mask: match segment.kind {
                        SegmentKind::Text => SegmentMask::CausalBottomRight,
                        SegmentKind::ConditionImage { .. } | SegmentKind::Target => {
                            SegmentMask::Full
                        }
                    },
                })
                .collect()
        };
        BlockCausalPlan {
            segments,
            key_valid: self.key_valid.clone(),
        }
    }

    /// Build rotary `cos`/`sin` tables `[positions, head_dim / 2]` for the
    /// interleaved complex-pair layout (`apply_rotary_emb_qwen(...,
    /// use_real=False)`), theta 10000 per axis.
    ///
    /// Angles are evaluated in f64 and rounded once to f32, the rounding
    /// boundary mold's text-to-image path has always had. Metal keeps F32
    /// tables for its BF16 denoiser; other devices take `dtype`.
    pub(crate) fn rope_tables(
        coords: &[[i32; 3]],
        axes_dims: [usize; 3],
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let half = axes_dims.iter().sum::<usize>() / 2;
        let mut cos = Vec::with_capacity(coords.len() * half);
        let mut sin = Vec::with_capacity(coords.len() * half);
        for coordinate in coords {
            for (axis, axis_dim) in axes_dims.iter().copied().enumerate() {
                for index in (0..axis_dim).step_by(2) {
                    let frequency = 1.0 / 10_000.0f64.powf(index as f64 / axis_dim as f64);
                    let angle = coordinate[axis] as f64 * frequency;
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }
        let dtype = if device.is_metal() { DType::F32 } else { dtype };
        Ok((
            Tensor::from_vec(cos, (coords.len(), half), device)?.to_dtype(dtype)?,
            Tensor::from_vec(sin, (coords.len(), half), device)?.to_dtype(dtype)?,
        ))
    }

    /// Upstream's `image_ids` (`T:808-850`): `-1` at text positions, a unique
    /// id per image block in order, target last.
    #[cfg(test)]
    pub(crate) fn image_ids(&self) -> Vec<i64> {
        let mut ids = vec![-1i64; self.total_len];
        let mut block = 0i64;
        for segment in &self.segments {
            if segment.kind != SegmentKind::Text {
                ids[segment.start..segment.end()].fill(block);
                block += 1;
            }
        }
        ids
    }

    /// Dense additive block-causal mask `[B, 1, T, T]` built from upstream's
    /// `mask_mod` (`T:292-296`): `((q >= kv) | same_image_block) & key_valid`.
    /// Only a test oracle — the production path never materializes it.
    #[cfg(test)]
    pub(crate) fn dense_bias(&self, batch: usize, device: &Device) -> Result<Tensor> {
        let ids = self.image_ids();
        let n = self.total_len;
        let mut values = Vec::with_capacity(batch * n * n);
        for row in 0..batch {
            for q in 0..n {
                for kv in 0..n {
                    let same_block = ids[q] >= 0 && ids[q] == ids[kv];
                    let valid = self.key_valid.as_ref().is_none_or(|rows| rows[row][kv]);
                    values.push(if (q >= kv || same_block) && valid {
                        0.0f32
                    } else {
                        f32::NEG_INFINITY
                    });
                }
            }
        }
        Ok(Tensor::from_vec(values, (batch, 1, n, n), device)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// text 3, reference 2x4 (2 slots), text 2, one pad row, target 4x4.
    fn hand_built() -> QwenImage21JointLayout {
        let slots = [false, false, false, true, true, false, false, false];
        let valid = vec![vec![true, true, true, true, true, true, true, false]];
        QwenImage21JointLayout::build(&slots, &valid, &[(2, 4)], (4, 4)).unwrap()
    }

    #[test]
    fn hand_built_layout_places_blocks_rope_and_padding() {
        let layout = hand_built();
        assert_eq!(layout.total_len(), 3 + 8 + 3 + 16);
        assert_eq!(layout.prefix_len(), 14);
        assert_eq!(layout.text_len(), 8);
        assert_eq!(layout.condition_tokens(), 8);
        assert_eq!(layout.target_tokens(), 16);
        assert_eq!(
            layout.segments(),
            &[
                JointSegment {
                    kind: SegmentKind::Text,
                    start: 0,
                    len: 3,
                    source: 0..3,
                    latent_hw: None
                },
                JointSegment {
                    kind: SegmentKind::ConditionImage { index: 0 },
                    start: 3,
                    len: 8,
                    source: 0..8,
                    latent_hw: Some((2, 4))
                },
                JointSegment {
                    kind: SegmentKind::Text,
                    start: 11,
                    len: 3,
                    source: 5..8,
                    latent_hw: None
                },
                JointSegment {
                    kind: SegmentKind::Target,
                    start: 14,
                    len: 16,
                    source: 8..24,
                    latent_hw: Some((4, 4))
                },
            ]
        );
        // Text rows 0..3, the 8 reference latents (sources 8..16), text rows
        // 5..8, then the 16 target latents (sources 16..32).
        let mut gather: Vec<u32> = vec![0, 1, 2];
        gather.extend(8..16);
        gather.extend([5, 6, 7]);
        gather.extend(16..32);
        assert_eq!(layout.gather_index(), gather.as_slice());
        assert!(!layout.gather_is_identity());

        let ids = layout.image_ids();
        assert_eq!(&ids[..3], &[-1, -1, -1]);
        assert!(ids[3..11].iter().all(|id| *id == 0));
        assert_eq!(&ids[11..14], &[-1, -1, -1]);
        assert!(ids[14..].iter().all(|id| *id == 1));

        let rope = layout.rope();
        assert_eq!(&rope[..3], &[[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
        // Reference block: frame frozen at 3, h in -1..1, w in -2..2.
        assert_eq!(rope[3], [3, -1, -2]);
        assert_eq!(rope[10], [3, 0, 1]);
        // Cursor advanced by max(2, 4) = 4, then three text rows.
        assert_eq!(&rope[11..14], &[[7, 7, 7], [8, 8, 8], [9, 9, 9]]);
        assert_eq!(rope[14], [10, -2, -2]);
        assert_eq!(rope[29], [10, 1, 1]);

        // The pad row (VL row 7 → joint 13) is the only invalid key.
        let key_valid = layout.key_valid().unwrap();
        assert_eq!(key_valid.len(), 1);
        let invalid: Vec<usize> = (0..layout.total_len())
            .filter(|&index| !key_valid[0][index])
            .collect();
        assert_eq!(invalid, vec![13]);
    }

    #[test]
    fn plans_decompose_the_prefix_and_collapse_when_cached() {
        let layout = hand_built();
        let plan = layout.attention_plan(false);
        assert_eq!(
            plan.segments,
            vec![
                AttentionSegment {
                    q_start: 0,
                    q_len: 3,
                    kv_len: 3,
                    mask: SegmentMask::CausalBottomRight
                },
                AttentionSegment {
                    q_start: 3,
                    q_len: 8,
                    kv_len: 11,
                    mask: SegmentMask::Full
                },
                AttentionSegment {
                    q_start: 11,
                    q_len: 3,
                    kv_len: 14,
                    mask: SegmentMask::CausalBottomRight
                },
                AttentionSegment {
                    q_start: 14,
                    q_len: 16,
                    kv_len: 30,
                    mask: SegmentMask::Full
                },
            ]
        );
        // The padded key (13) is beyond the reference block's keys.
        assert!(plan.key_valid_prefix(11).is_none());
        assert!(plan.key_valid_prefix(14).is_some());
        let cached = layout.attention_plan(true);
        assert_eq!(
            cached.segments,
            vec![AttentionSegment {
                q_start: 0,
                q_len: 16,
                kv_len: 30,
                mask: SegmentMask::Full
            }]
        );
        assert_eq!(cached.key_valid, plan.key_valid);
    }

    #[test]
    fn adjacent_condition_images_stay_separate_blocks() {
        // Two references whose slots touch with no text between them.
        let slots = [false, true, true, true, false];
        let valid = vec![vec![true; 5]];
        let layout =
            QwenImage21JointLayout::build(&slots, &valid, &[(2, 2), (2, 4)], (2, 2)).unwrap();
        let kinds: Vec<SegmentKind> = layout.segments().iter().map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            vec![
                SegmentKind::Text,
                SegmentKind::ConditionImage { index: 0 },
                SegmentKind::ConditionImage { index: 1 },
                SegmentKind::Text,
                SegmentKind::Target,
            ]
        );
        let ids = layout.image_ids();
        assert!(ids[1..5].iter().all(|id| *id == 0));
        assert!(ids[5..13].iter().all(|id| *id == 1));
        // Each block restarts its centred grid; the frame advances by max(h, w).
        assert_eq!(layout.rope()[1], [1, -1, -1]);
        assert_eq!(layout.rope()[5], [3, -1, -2]);
        assert_eq!(layout.rope()[13], [7, 7, 7]);
        assert!(layout.key_valid().is_none());
    }

    #[test]
    fn slot_count_mismatches_and_bad_masks_fail() {
        let slots = [false, true, true, false];
        let valid = vec![vec![true; 4]];
        // 2 slots = 8 tokens, but the reference claims 4x4 = 16.
        assert!(QwenImage21JointLayout::build(&slots, &valid, &[(4, 4)], (2, 2)).is_err());
        // Slots with no condition image.
        assert!(QwenImage21JointLayout::build(&slots, &valid, &[], (2, 2)).is_err());
        // Left padding.
        let left = vec![vec![false, true, true, true]];
        assert!(QwenImage21JointLayout::build(&[false; 4], &left, &[], (2, 2)).is_err());
        // Padding inside the text.
        let gap = vec![vec![true, false, true, true]];
        assert!(QwenImage21JointLayout::build(&[false; 4], &gap, &[], (2, 2)).is_err());
        // A target that does not tile 2x2 slots.
        assert!(QwenImage21JointLayout::text_to_image(&valid, (3, 3)).is_err());
        // A block split by text: 1 + 1 slots for a 2x4 (2-slot) image.
        let split = [true, false, true, false];
        assert!(QwenImage21JointLayout::build(&split, &valid, &[(2, 4)], (2, 2)).is_err());
    }

    #[test]
    fn text_to_image_layout_is_text_then_target() {
        let valid = vec![vec![true, true, false], vec![true; 3]];
        let layout = QwenImage21JointLayout::text_to_image(&valid, (2, 2)).unwrap();
        assert!(layout.gather_is_identity());
        assert_eq!(layout.prefix_len(), 3);
        assert_eq!(layout.total_len(), 7);
        let plan = layout.attention_plan(false);
        assert_eq!(plan.segments.len(), 2);
        assert_eq!(plan.segments[0].mask, SegmentMask::CausalBottomRight);
        let key_valid = layout.key_valid().unwrap();
        assert_eq!(
            key_valid[0],
            vec![true, true, false, true, true, true, true]
        );
        assert!(key_valid[1].iter().all(|valid| *valid));
    }
}
