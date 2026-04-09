#![allow(dead_code)]

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

#[cfg(test)]
mod tests {
    use super::{left_pad_batch, pad_to_alignment};

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
}
