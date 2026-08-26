//! LTX-2.5 caption-conditioned shot-duration prediction.
//!
//! The head consumes the already-computed video/audio connector tokens, so it
//! adds no second text-encoder pass. Its tiny checkpoint stays independent from
//! the transformer and VAE residency lifecycle.

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear_b, Activation, Linear, Module, VarBuilder};
use mold_core::ltx2_duration::{seconds_to_clamped_frames, AutoDurationBounds};

const VIDEO_DIM: usize = 4096;
const AUDIO_DIM: usize = 2048;
const HIDDEN_DIM: usize = 256;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS;
#[derive(Debug, Clone)]
pub struct Ltx2DurationHead {
    video_input_proj: Linear,
    video_modality_emb: Tensor,
    audio_input_proj: Linear,
    audio_modality_emb: Tensor,
    query_tokens: Tensor,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    mlp_hidden: Linear,
    mlp_out: Linear,
}

impl Ltx2DurationHead {
    pub fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let pooler = vb.pp("attention_pooler");
        let cross_attn = pooler.pp("cross_attn");
        let in_weight = cross_attn.get((HIDDEN_DIM * 3, HIDDEN_DIM), "in_proj_weight")?;
        let in_bias = cross_attn.get(HIDDEN_DIM * 3, "in_proj_bias")?;
        let split_linear = |offset: usize| -> Result<Linear> {
            Ok(Linear::new(
                in_weight.narrow(0, offset, HIDDEN_DIM)?,
                Some(in_bias.narrow(0, offset, HIDDEN_DIM)?),
            ))
        };

        Ok(Self {
            video_input_proj: linear_b(VIDEO_DIM, HIDDEN_DIM, true, vb.pp("video_input_proj"))?,
            video_modality_emb: vb.get(HIDDEN_DIM, "video_modality_emb")?,
            audio_input_proj: linear_b(AUDIO_DIM, HIDDEN_DIM, true, vb.pp("audio_input_proj"))?,
            audio_modality_emb: vb.get(HIDDEN_DIM, "audio_modality_emb")?,
            query_tokens: pooler.get((1, HIDDEN_DIM), "query_tokens")?,
            q_proj: split_linear(0)?,
            k_proj: split_linear(HIDDEN_DIM)?,
            v_proj: split_linear(HIDDEN_DIM * 2)?,
            out_proj: linear_b(HIDDEN_DIM, HIDDEN_DIM, true, cross_attn.pp("out_proj"))?,
            mlp_hidden: linear_b(HIDDEN_DIM, HIDDEN_DIM, true, vb.pp("mlp_hidden"))?,
            mlp_out: linear_b(HIDDEN_DIM, 1, true, vb.pp("mlp_out"))?,
        })
    }

    /// Load the dedicated split-pack duration checkpoint.
    pub fn from_checkpoint(path: &Path, dtype: DType, device: &Device) -> Result<Self> {
        let paths = [PathBuf::from(path)];
        // SAFETY: safetensors validates tensor bounds before exposing mmap slices;
        // the mapped file remains owned by the VarBuilder backend.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, device)? };
        Self::new(vb.pp("duration_head"))
    }

    /// Predict seconds for one batch from either or both connector streams.
    pub fn predict_seconds(
        &self,
        video_tokens: Option<&Tensor>,
        audio_tokens: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut groups = Vec::with_capacity(2);
        if let Some(tokens) = video_tokens {
            ensure_connector_shape(tokens, VIDEO_DIM, "video")?;
            let tokens = tokens.to_dtype(self.video_input_proj.weight().dtype())?;
            groups.push(
                self.video_input_proj
                    .forward(&tokens)?
                    .broadcast_add(&self.video_modality_emb)?,
            );
        }
        if let Some(tokens) = audio_tokens {
            ensure_connector_shape(tokens, AUDIO_DIM, "audio")?;
            let tokens = tokens.to_dtype(self.audio_input_proj.weight().dtype())?;
            groups.push(
                self.audio_input_proj
                    .forward(&tokens)?
                    .broadcast_add(&self.audio_modality_emb)?,
            );
        }
        if groups.is_empty() {
            bail!("duration prediction requires video or audio connector tokens");
        }
        let batch = groups[0].dim(0)?;
        ensure!(
            groups.iter().all(|group| group.dim(0).ok() == Some(batch)),
            "duration connector batch sizes must match"
        );
        let tokens = Tensor::cat(&groups, 1)?;
        let pooled = self.pool(&tokens)?;
        let hidden = self.mlp_hidden.forward(&pooled)?;
        let hidden = Activation::GeluPytorchTanh.forward(&hidden)?;
        let log_seconds = self.mlp_out.forward(&hidden)?.squeeze(1)?;
        Ok(log_seconds.to_dtype(DType::F32)?.exp()?)
    }

    pub fn predict_frames(
        &self,
        video_tokens: Option<&Tensor>,
        audio_tokens: Option<&Tensor>,
        fps: u32,
        bounds: AutoDurationBounds,
    ) -> Result<u32> {
        ensure!(fps > 0, "auto-duration fps must be positive");
        let bounds = bounds.validate()?;
        let predictions = self.predict_seconds(video_tokens, audio_tokens)?;
        ensure!(
            predictions.elem_count() == 1,
            "auto-duration supports a single generation at a time"
        );
        let seconds = predictions.flatten_all()?.to_vec1::<f32>()?[0] as f64;
        seconds_to_clamped_frames(seconds, fps, bounds)
    }

    fn pool(&self, tokens: &Tensor) -> Result<Tensor> {
        let (batch, sequence, hidden) = tokens.dims3()?;
        ensure!(
            hidden == HIDDEN_DIM,
            "duration pooler hidden dimension mismatch"
        );
        let queries = self
            .query_tokens
            .unsqueeze(0)?
            .broadcast_as((batch, 1, hidden))?;
        let q = split_heads(&self.q_proj.forward(&queries)?, batch, 1)?;
        let k = split_heads(&self.k_proj.forward(tokens)?, batch, sequence)?;
        let v = split_heads(&self.v_proj.forward(tokens)?, batch, sequence)?;
        let scores =
            (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * (1.0 / (HEAD_DIM as f64).sqrt()))?;
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = probabilities
            .contiguous()?
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, HIDDEN_DIM))?;
        self.out_proj.forward(&context).map_err(Into::into)
    }
}

fn ensure_connector_shape(tokens: &Tensor, expected_hidden: usize, modality: &str) -> Result<()> {
    let (_, _, hidden) = tokens.dims3()?;
    ensure!(
        hidden == expected_hidden,
        "{modality} duration tokens require hidden size {expected_hidden}, got {hidden}"
    );
    Ok(())
}

fn split_heads(tokens: &Tensor, batch: usize, sequence: usize) -> Result<Tensor> {
    Ok(tokens
        .reshape((batch, sequence, NUM_HEADS, HEAD_DIM))?
        .transpose(1, 2)?
        .contiguous()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_head_uses_shared_frame_policy() {
        assert_eq!(
            seconds_to_clamped_frames(2.0, 24, AutoDurationBounds::default()).unwrap(),
            41
        );
    }
}
