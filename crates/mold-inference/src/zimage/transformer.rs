use anyhow::Result;
use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::{linear, Module, VarBuilder};
use candle_transformers::models::with_tracing::RmsNorm;
use candle_transformers::models::z_image::transformer::{
    create_coordinate_grid, patchify, unpatchify, Config, FinalLayer, RopeEmbedder,
    TimestepEmbedder, ZImageTransformerBlock, ADALN_EMBED_DIM, SEQ_MULTI_OF,
};

fn pad_extra_to_multiple(len: usize, multiple: usize) -> usize {
    if multiple == 0 {
        return 0;
    }
    (multiple - (len % multiple)) % multiple
}

fn pad_token_sequence(
    xs: &Tensor,
    pad_token: &Tensor,
    multiple: usize,
) -> CandleResult<(Tensor, usize)> {
    let (batch, seq_len, dim) = xs.dims3()?;
    let pad_extra = pad_extra_to_multiple(seq_len, multiple);
    if pad_extra == 0 {
        return Ok((xs.clone(), 0));
    }

    let pad = pad_token
        .to_device(xs.device())?
        .to_dtype(xs.dtype())?
        .unsqueeze(0)?
        .broadcast_as((batch, pad_extra, dim))?;
    Ok((Tensor::cat(&[xs, &pad], 1)?, pad_extra))
}

fn pad_position_ids_with_zeros(pos_ids: &Tensor, pad_extra: usize) -> CandleResult<Tensor> {
    if pad_extra == 0 {
        return Ok(pos_ids.clone());
    }
    let pad = Tensor::zeros((pad_extra, 3), DType::U32, pos_ids.device())?;
    Tensor::cat(&[pos_ids, &pad], 0)
}

fn build_basic_unified_sequence(
    image: &Tensor,
    cap: &Tensor,
    image_pos_ids: &Tensor,
    cap_pos_ids: &Tensor,
) -> CandleResult<(Tensor, Tensor)> {
    Ok((
        Tensor::cat(&[image, cap], 1)?,
        Tensor::cat(&[image_pos_ids, cap_pos_ids], 0)?,
    ))
}

/// Z-Image transformer with the reference pad-token protocol.
///
/// The Candle model loads `x_pad_token` and `cap_pad_token`, but its forward
/// path does not currently use them. Z-Image checkpoints are trained with text
/// and image streams padded to 32-token boundaries before attention; skipping
/// that changes RoPE offsets and the token population seen by every block.
pub(crate) struct MoldZImageTransformer2DModel {
    t_embedder: TimestepEmbedder,
    cap_embedder_norm: RmsNorm,
    cap_embedder_linear: candle_nn::Linear,
    x_embedder: candle_nn::Linear,
    final_layer: FinalLayer,
    x_pad_token: Tensor,
    cap_pad_token: Tensor,
    noise_refiner: Vec<ZImageTransformerBlock>,
    context_refiner: Vec<ZImageTransformerBlock>,
    layers: Vec<ZImageTransformerBlock>,
    rope_embedder: RopeEmbedder,
    cfg: Config,
}

impl MoldZImageTransformer2DModel {
    pub(crate) fn new(cfg: &Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let device = vb.device();
        let dtype = vb.dtype();

        let adaln_dim = cfg.dim.min(ADALN_EMBED_DIM);
        let t_embedder = TimestepEmbedder::new(adaln_dim, 1024, vb.pp("t_embedder"))?;

        let cap_embedder_norm = RmsNorm::new(
            cfg.cap_feat_dim,
            cfg.norm_eps,
            vb.pp("cap_embedder").pp("0"),
        )?;
        let cap_embedder_linear = linear(cfg.cap_feat_dim, cfg.dim, vb.pp("cap_embedder").pp("1"))?;

        let patch_dim = cfg.all_f_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.in_channels;
        let x_embedder = linear(patch_dim, cfg.dim, vb.pp("all_x_embedder").pp("2-1"))?;

        let out_channels = cfg.all_patch_size[0]
            * cfg.all_patch_size[0]
            * cfg.all_f_patch_size[0]
            * cfg.in_channels;
        let final_layer =
            FinalLayer::new(cfg.dim, out_channels, vb.pp("all_final_layer").pp("2-1"))?;

        let x_pad_token = vb.get((1, cfg.dim), "x_pad_token")?;
        let cap_pad_token = vb.get((1, cfg.dim), "cap_pad_token")?;

        let mut noise_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        for i in 0..cfg.n_refiner_layers {
            noise_refiner.push(ZImageTransformerBlock::new(
                cfg,
                true,
                vb.pp("noise_refiner").pp(i),
            )?);
        }

        let mut context_refiner = Vec::with_capacity(cfg.n_refiner_layers);
        for i in 0..cfg.n_refiner_layers {
            context_refiner.push(ZImageTransformerBlock::new(
                cfg,
                false,
                vb.pp("context_refiner").pp(i),
            )?);
        }

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(ZImageTransformerBlock::new(
                cfg,
                true,
                vb.pp("layers").pp(i),
            )?);
        }

        let rope_embedder = RopeEmbedder::new(
            cfg.rope_theta,
            cfg.axes_dims.clone(),
            cfg.axes_lens.clone(),
            device,
            dtype,
        )?;

        Ok(Self {
            t_embedder,
            cap_embedder_norm,
            cap_embedder_linear,
            x_embedder,
            final_layer,
            x_pad_token,
            cap_pad_token,
            noise_refiner,
            context_refiner,
            layers,
            rope_embedder,
            cfg: cfg.clone(),
        })
    }

    pub(crate) fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        _cap_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let device = x.device();
        let (_batch, _channels, frames, height, width) = x.dims5()?;
        let patch_size = self.cfg.all_patch_size[0];
        let frame_patch_size = self.cfg.all_f_patch_size[0];

        let t_scaled = (t * self.cfg.t_scale)?;
        let adaln_input = self.t_embedder.forward(&t_scaled)?;

        let cap = self.cap_embedder_norm.forward(cap_feats)?;
        let cap = cap.apply(&self.cap_embedder_linear)?;
        let (mut cap, _) = pad_token_sequence(&cap, &self.cap_pad_token, SEQ_MULTI_OF)?;
        let padded_text_len = cap.dim(1)?;
        let cap_pos_ids = create_coordinate_grid((padded_text_len, 1, 1), (1, 0, 0), device)?;
        let (cap_cos, cap_sin) = self.rope_embedder.forward(&cap_pos_ids)?;

        let (x_patches, orig_size) = patchify(x, patch_size, frame_patch_size)?;
        let x = x_patches.apply(&self.x_embedder)?;
        let (mut image, image_pad_extra) = pad_token_sequence(&x, &self.x_pad_token, SEQ_MULTI_OF)?;
        let padded_image_seq_len = image.dim(1)?;

        let frame_tokens = frames / frame_patch_size;
        let height_tokens = height / patch_size;
        let width_tokens = width / patch_size;
        let image_pos_ids = create_coordinate_grid(
            (frame_tokens, height_tokens, width_tokens),
            (padded_text_len + 1, 0, 0),
            device,
        )?;
        let image_pos_ids = pad_position_ids_with_zeros(&image_pos_ids, image_pad_extra)?;
        let (image_cos, image_sin) = self.rope_embedder.forward(&image_pos_ids)?;

        for layer in &self.context_refiner {
            cap = layer.forward(&cap, None, &cap_cos, &cap_sin, None)?;
        }

        for layer in &self.noise_refiner {
            image = layer.forward(&image, None, &image_cos, &image_sin, Some(&adaln_input))?;
        }

        let (mut unified, unified_pos_ids) =
            build_basic_unified_sequence(&image, &cap, &image_pos_ids, &cap_pos_ids)?;
        let (unified_cos, unified_sin) = self.rope_embedder.forward(&unified_pos_ids)?;

        for layer in &self.layers {
            unified = layer.forward(
                &unified,
                None,
                &unified_cos,
                &unified_sin,
                Some(&adaln_input),
            )?;
        }

        let image = unified.narrow(1, 0, padded_image_seq_len)?;
        let image = self.final_layer.forward(&image, &adaln_input)?;
        unpatchify(
            &image,
            orig_size,
            patch_size,
            frame_patch_size,
            self.cfg.in_channels,
        )
    }
}

/// Dense Z-Image transformer, regardless of original weight source.
pub(crate) enum ZImageTransformer {
    Dense(MoldZImageTransformer2DModel),
    Quantized(super::quantized_transformer::QuantizedZImageTransformer2DModel),
}

impl ZImageTransformer {
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cap_feats: &Tensor,
        cap_mask: &Tensor,
    ) -> Result<Tensor> {
        match self {
            Self::Dense(m) => Ok(m.forward(x, t, cap_feats, cap_mask)?),
            Self::Quantized(m) => Ok(m.forward(x, t, cap_feats, cap_mask)?),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn zimage_padding_rounds_up_to_reference_multiple() {
        assert_eq!(pad_extra_to_multiple(0, 32), 0);
        assert_eq!(pad_extra_to_multiple(1, 32), 31);
        assert_eq!(pad_extra_to_multiple(31, 32), 1);
        assert_eq!(pad_extra_to_multiple(32, 32), 0);
        assert_eq!(pad_extra_to_multiple(33, 32), 31);
    }

    #[test]
    fn zimage_padding_appends_learned_token_values() {
        let xs = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (1, 3, 2),
            &Device::Cpu,
        )
        .unwrap();
        let pad_token = Tensor::from_vec(vec![9.0f32, 10.0], (1, 2), &Device::Cpu).unwrap();

        let (padded, extra) = pad_token_sequence(&xs, &pad_token, 4).unwrap();

        assert_eq!(extra, 1);
        assert_eq!(padded.dims(), &[1, 4, 2]);
        assert_eq!(
            padded.to_vec3::<f32>().unwrap(),
            vec![vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![5.0, 6.0],
                vec![9.0, 10.0],
            ]]
        );
    }

    #[test]
    fn zimage_padding_appends_zero_position_ids_for_image_pads() {
        let ids = Tensor::from_vec(vec![7u32, 0, 0, 7, 0, 1], (2, 3), &Device::Cpu).unwrap();

        let padded = pad_position_ids_with_zeros(&ids, 2).unwrap();

        assert_eq!(
            padded.to_vec2::<u32>().unwrap(),
            vec![vec![7, 0, 0], vec![7, 0, 1], vec![0, 0, 0], vec![0, 0, 0]]
        );
    }

    #[test]
    fn zimage_basic_unified_sequence_keeps_image_tokens_first() {
        let image = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &Device::Cpu).unwrap();
        let cap = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (1, 2, 2), &Device::Cpu).unwrap();
        let image_pos =
            Tensor::from_vec(vec![10u32, 0, 0, 10, 0, 1], (2, 3), &Device::Cpu).unwrap();
        let cap_pos = Tensor::from_vec(vec![1u32, 0, 0, 2, 0, 0], (2, 3), &Device::Cpu).unwrap();

        let (unified, unified_pos) =
            build_basic_unified_sequence(&image, &cap, &image_pos, &cap_pos).unwrap();

        assert_eq!(
            unified.to_vec3::<f32>().unwrap(),
            vec![vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
                vec![5.0, 6.0],
                vec![7.0, 8.0],
            ]]
        );
        assert_eq!(
            unified_pos.to_vec2::<u32>().unwrap(),
            vec![vec![10, 0, 0], vec![10, 0, 1], vec![1, 0, 0], vec![2, 0, 0],]
        );
    }
}
