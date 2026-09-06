//! Shared SD VAE adapted from Candle bedc287458e0d890dd6ed1c298c99e991e066fe1,
//! candle-transformers/src/models/stable_diffusion/vae.rs (MIT licence;
//! see LICENSE-CANDLE-MIT and THIRD_PARTY_NOTICES.md).
//!
//! Existing SD callers retain Candle blocks and default posterior arithmetic.
//! Paint opts into PyTorch CUDA numerical boundaries, bounded log variance and
//! caller-owned noise while sharing the encoder/decoder architecture.
use candle::{Result, Tensor};
use candle_nn as nn;
use candle_nn::Module;
use candle_transformers::models::stable_diffusion::unet_2d_blocks::{
    DownEncoderBlock2D, DownEncoderBlock2DConfig, UNetMidBlock2D, UNetMidBlock2DConfig,
    UpDecoderBlock2D, UpDecoderBlock2DConfig,
};

mod precision;

/// Existing image engines preserve Candle arithmetic. New paint checkpoints use
/// PyTorch CUDA opmath and rounding boundaries, qualified separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaeNumerics {
    Candle,
    Diffusers,
}

#[derive(Debug, Clone)]
struct EncoderConfig {
    // down_block_types: DownEncoderBlock2D
    block_out_channels: Vec<usize>,
    layers_per_block: usize,
    norm_num_groups: usize,
    numerics: VaeNumerics,
    double_z: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![64],
            layers_per_block: 2,
            norm_num_groups: 32,
            numerics: VaeNumerics::Candle,
            double_z: true,
        }
    }
}

#[derive(Debug)]
struct Encoder {
    conv_in: nn::Conv2d,
    down_blocks: Vec<Box<dyn precision::Block>>,
    mid_block: Box<dyn precision::Block>,
    conv_norm_out: precision::Norm,
    conv_out: nn::Conv2d,
    #[allow(dead_code)]
    config: EncoderConfig,
}

impl Encoder {
    fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        config: EncoderConfig,
    ) -> Result<Self> {
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = nn::conv2d(
            in_channels,
            config.block_out_channels[0],
            3,
            conv_cfg,
            vs.pp("conv_in"),
        )?;
        let mut down_blocks = vec![];
        let vs_down_blocks = vs.pp("down_blocks");
        for index in 0..config.block_out_channels.len() {
            let out_channels = config.block_out_channels[index];
            let in_channels = if index > 0 {
                config.block_out_channels[index - 1]
            } else {
                config.block_out_channels[0]
            };
            let is_final = index + 1 == config.block_out_channels.len();
            let cfg = DownEncoderBlock2DConfig {
                num_layers: config.layers_per_block,
                resnet_eps: 1e-6,
                resnet_groups: config.norm_num_groups,
                add_downsample: !is_final,
                downsample_padding: 0,
                ..Default::default()
            };
            let down_block: Box<dyn precision::Block> = if config.numerics == VaeNumerics::Diffusers
            {
                Box::new(precision::Down::new(
                    vs_down_blocks.pp(index.to_string()),
                    in_channels,
                    out_channels,
                    config.layers_per_block,
                    config.norm_num_groups,
                    !is_final,
                )?)
            } else {
                Box::new(DownEncoderBlock2D::new(
                    vs_down_blocks.pp(index.to_string()),
                    in_channels,
                    out_channels,
                    cfg,
                )?)
            };
            down_blocks.push(down_block)
        }
        let last_block_out_channels = *config.block_out_channels.last().unwrap();
        let mid_cfg = UNetMidBlock2DConfig {
            resnet_eps: 1e-6,
            output_scale_factor: 1.,
            attn_num_head_channels: None,
            resnet_groups: Some(config.norm_num_groups),
            ..Default::default()
        };
        let mid_block: Box<dyn precision::Block> = if config.numerics == VaeNumerics::Diffusers {
            Box::new(precision::Mid::new(
                vs.pp("mid_block"),
                last_block_out_channels,
                config.norm_num_groups,
            )?)
        } else {
            Box::new(precision::LegacyMid(UNetMidBlock2D::new(
                vs.pp("mid_block"),
                last_block_out_channels,
                None,
                mid_cfg,
            )?))
        };
        let conv_norm_out = precision::Norm::new(
            vs.pp("conv_norm_out"),
            config.norm_num_groups,
            last_block_out_channels,
            config.numerics,
        )?;
        let conv_out_channels = if config.double_z {
            2 * out_channels
        } else {
            out_channels
        };
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_out = nn::conv2d(
            last_block_out_channels,
            conv_out_channels,
            3,
            conv_cfg,
            vs.pp("conv_out"),
        )?;
        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
            config,
        })
    }
}

impl Encoder {
    fn forward_with_observer(
        &self,
        xs: &Tensor,
        observe: &mut impl FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let mut xs = xs.apply(&self.conv_in)?;
        observe("encoder.conv_in", &xs)?;
        for (index, down_block) in self.down_blocks.iter().enumerate() {
            xs = down_block.forward(&xs)?;
            observe(&format!("encoder.down_blocks.{index}"), &xs)?;
        }
        let xs = self.mid_block.forward(&xs)?;
        observe("encoder.mid_block", &xs)?;
        let xs = xs.apply(&self.conv_norm_out)?;
        observe("encoder.conv_norm_out", &xs)?;
        let xs = precision::silu(&xs, self.config.numerics)?.apply(&self.conv_out)?;
        observe("encoder.conv_out", &xs)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct DecoderConfig {
    // up_block_types: UpDecoderBlock2D
    block_out_channels: Vec<usize>,
    layers_per_block: usize,
    norm_num_groups: usize,
    numerics: VaeNumerics,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            block_out_channels: vec![64],
            layers_per_block: 2,
            norm_num_groups: 32,
            numerics: VaeNumerics::Candle,
        }
    }
}

#[derive(Debug)]
struct Decoder {
    conv_in: nn::Conv2d,
    up_blocks: Vec<Box<dyn precision::Block>>,
    mid_block: Box<dyn precision::Block>,
    conv_norm_out: precision::Norm,
    conv_out: nn::Conv2d,
    #[allow(dead_code)]
    config: DecoderConfig,
}

impl Decoder {
    fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        config: DecoderConfig,
    ) -> Result<Self> {
        let n_block_out_channels = config.block_out_channels.len();
        let last_block_out_channels = *config.block_out_channels.last().unwrap();
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_in = nn::conv2d(
            in_channels,
            last_block_out_channels,
            3,
            conv_cfg,
            vs.pp("conv_in"),
        )?;
        let mid_cfg = UNetMidBlock2DConfig {
            resnet_eps: 1e-6,
            output_scale_factor: 1.,
            attn_num_head_channels: None,
            resnet_groups: Some(config.norm_num_groups),
            ..Default::default()
        };
        let mid_block: Box<dyn precision::Block> = if config.numerics == VaeNumerics::Diffusers {
            Box::new(precision::Mid::new(
                vs.pp("mid_block"),
                last_block_out_channels,
                config.norm_num_groups,
            )?)
        } else {
            Box::new(precision::LegacyMid(UNetMidBlock2D::new(
                vs.pp("mid_block"),
                last_block_out_channels,
                None,
                mid_cfg,
            )?))
        };
        let mut up_blocks = vec![];
        let vs_up_blocks = vs.pp("up_blocks");
        let reversed_block_out_channels: Vec<_> =
            config.block_out_channels.iter().copied().rev().collect();
        for index in 0..n_block_out_channels {
            let out_channels = reversed_block_out_channels[index];
            let in_channels = if index > 0 {
                reversed_block_out_channels[index - 1]
            } else {
                reversed_block_out_channels[0]
            };
            let is_final = index + 1 == n_block_out_channels;
            let cfg = UpDecoderBlock2DConfig {
                num_layers: config.layers_per_block + 1,
                resnet_eps: 1e-6,
                resnet_groups: config.norm_num_groups,
                add_upsample: !is_final,
                ..Default::default()
            };
            let up_block: Box<dyn precision::Block> = if config.numerics == VaeNumerics::Diffusers {
                Box::new(precision::Up::new(
                    vs_up_blocks.pp(index.to_string()),
                    in_channels,
                    out_channels,
                    config.layers_per_block + 1,
                    config.norm_num_groups,
                    !is_final,
                )?)
            } else {
                Box::new(UpDecoderBlock2D::new(
                    vs_up_blocks.pp(index.to_string()),
                    in_channels,
                    out_channels,
                    cfg,
                )?)
            };
            up_blocks.push(up_block)
        }
        let conv_norm_out = precision::Norm::new(
            vs.pp("conv_norm_out"),
            config.norm_num_groups,
            config.block_out_channels[0],
            config.numerics,
        )?;
        let conv_cfg = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv_out = nn::conv2d(
            config.block_out_channels[0],
            out_channels,
            3,
            conv_cfg,
            vs.pp("conv_out"),
        )?;
        Ok(Self {
            conv_in,
            up_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
            config,
        })
    }
}

impl Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.mid_block.forward(&self.conv_in.forward(xs)?)?;
        for up_block in self.up_blocks.iter() {
            xs = up_block.forward(&xs)?
        }
        let xs = self.conv_norm_out.forward(&xs)?;
        let xs = precision::silu(&xs, self.config.numerics)?;
        self.conv_out.forward(&xs)
    }
}

pub use candle_transformers::models::stable_diffusion::vae::AutoEncoderKLConfig;

pub struct DiagonalGaussianDistribution {
    mean: Tensor,
    std: Tensor,
}

impl DiagonalGaussianDistribution {
    /// Diffusers posterior bounds, opt-in so existing SD sampling is unchanged.
    pub fn new_clamped(parameters: &Tensor) -> Result<Self> {
        let channels = parameters.dim(1)?;
        if channels == 0 || !channels.is_multiple_of(2) {
            candle::bail!("posterior requires paired mean/logvar channels")
        }
        let mean = parameters.narrow(1, 0, channels / 2)?;
        let logvar = parameters
            .narrow(1, channels / 2, channels / 2)?
            .clamp(-30., 20.)?;
        let std = (logvar * 0.5)?.exp()?;
        Ok(Self { mean, std })
    }

    pub fn std(&self) -> &Tensor {
        &self.std
    }

    /// Caller-owned noise makes posterior draws part of the pipeline RNG order.
    pub fn sample_with_noise(&self, noise: &Tensor) -> Result<Tensor> {
        if noise.dims() != self.mean.dims() {
            candle::bail!("posterior noise shape does not match mean")
        }
        &self.mean + &self.std * noise
    }

    pub fn new(parameters: &Tensor) -> Result<Self> {
        let mut parameters = parameters.chunk(2, 1)?.into_iter();
        let mean = parameters.next().unwrap();
        let logvar = parameters.next().unwrap();
        let std = (logvar * 0.5)?.exp()?;
        Ok(DiagonalGaussianDistribution { mean, std })
    }

    pub fn mode(&self) -> Result<Tensor> {
        Ok(self.mean.clone())
    }

    pub fn sample(&self) -> Result<Tensor> {
        let sample = self.mean.randn_like(0., 1.);
        &self.mean + &self.std * sample
    }
}

// https://github.com/huggingface/diffusers/blob/970e30606c2944e3286f56e8eb6d3dc6d1eb85f7/src/diffusers/models/vae.py#L485
// This implementation is specific to the config used in stable-diffusion-v1-5
// https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/vae/config.json
#[derive(Debug)]
pub struct AutoEncoderKL {
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: Option<nn::Conv2d>,
    post_quant_conv: Option<nn::Conv2d>,
    pub config: AutoEncoderKLConfig,
}

impl AutoEncoderKL {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        config: AutoEncoderKLConfig,
    ) -> Result<Self> {
        Self::new_with_numerics(vs, in_channels, out_channels, config, VaeNumerics::Candle)
    }

    /// Build with an explicit numerical contract without changing existing SD callers.
    pub fn new_with_numerics(
        vs: nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        config: AutoEncoderKLConfig,
        numerics: VaeNumerics,
    ) -> Result<Self> {
        let latent_channels = config.latent_channels;
        let encoder_cfg = EncoderConfig {
            block_out_channels: config.block_out_channels.clone(),
            layers_per_block: config.layers_per_block,
            norm_num_groups: config.norm_num_groups,
            numerics,
            double_z: true,
        };
        let encoder = Encoder::new(vs.pp("encoder"), in_channels, latent_channels, encoder_cfg)?;
        let decoder_cfg = DecoderConfig {
            block_out_channels: config.block_out_channels.clone(),
            layers_per_block: config.layers_per_block,
            norm_num_groups: config.norm_num_groups,
            numerics,
        };
        let decoder = Decoder::new(vs.pp("decoder"), latent_channels, out_channels, decoder_cfg)?;
        let conv_cfg = Default::default();

        let quant_conv = {
            if config.use_quant_conv {
                Some(nn::conv2d(
                    2 * latent_channels,
                    2 * latent_channels,
                    1,
                    conv_cfg,
                    vs.pp("quant_conv"),
                )?)
            } else {
                None
            }
        };
        let post_quant_conv = {
            if config.use_post_quant_conv {
                Some(nn::conv2d(
                    latent_channels,
                    latent_channels,
                    1,
                    conv_cfg,
                    vs.pp("post_quant_conv"),
                )?)
            } else {
                None
            }
        };
        Ok(Self {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            config,
        })
    }

    /// Returns the distribution in the latent space.
    pub fn encode(&self, xs: &Tensor) -> Result<DiagonalGaussianDistribution> {
        DiagonalGaussianDistribution::new(&self.encode_moments(xs)?)
    }

    /// Raw mean/logvar channels for pipelines with an explicit posterior policy.
    pub fn encode_moments(&self, xs: &Tensor) -> Result<Tensor> {
        self.encode_moments_with_observer(xs, |_, _| Ok(()))
    }

    /// Observe encoder boundaries without retaining tensors or changing arithmetic.
    /// An observer error stops execution, allowing callers to cancel capture.
    pub fn encode_moments_with_observer(
        &self,
        xs: &Tensor,
        mut observe: impl FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let xs = self.encoder.forward_with_observer(xs, &mut observe)?;
        match &self.quant_conv {
            None => Ok(xs),
            Some(quant_conv) => {
                let xs = quant_conv.forward(&xs)?;
                observe("quant_conv", &xs)?;
                Ok(xs)
            }
        }
    }

    /// Takes as input some sampled values.
    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = match &self.post_quant_conv {
            None => xs,
            Some(post_quant_conv) => &post_quant_conv.forward(xs)?,
        };
        self.decoder.forward(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn max_error(actual: &Tensor, expected: &Tensor) -> f32 {
        (actual - expected)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn shared_vae_preserves_candle_and_matches_diffusers() {
        let device = Device::Cpu;
        let weights = candle::safetensors::load_buffer(
            include_bytes!(
                "../../../../tests/fixtures/hunyuan3d/paint-vae-tiny-weights.safetensors"
            ),
            &device,
        )
        .unwrap();
        let tensors = candle::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-vae-tiny.safetensors"),
            &device,
        )
        .unwrap();
        for quant in [true, false] {
            let cfg = AutoEncoderKLConfig {
                block_out_channels: vec![8, 16],
                layers_per_block: 1,
                latent_channels: 4,
                norm_num_groups: 4,
                use_quant_conv: quant,
                use_post_quant_conv: quant,
            };
            let vb = nn::VarBuilder::from_tensors(weights.clone(), DType::F32, &device);
            let model = AutoEncoderKL::new(vb.clone(), 3, 3, cfg.clone()).unwrap();
            let precise = AutoEncoderKL::new_with_numerics(
                vb.clone(),
                3,
                3,
                cfg.clone(),
                VaeNumerics::Diffusers,
            )
            .unwrap();
            assert!(
                max_error(
                    &model.encode_moments(&tensors["pixels"]).unwrap(),
                    &precise.encode_moments(&tensors["pixels"]).unwrap()
                ) < 1e-4
            );
            assert!(
                max_error(
                    &model.decode(&tensors["noise"]).unwrap(),
                    &precise.decode(&tensors["noise"]).unwrap()
                ) < 1e-4
            );
            let legacy_weights = weights
                .iter()
                .map(|(name, tensor)| {
                    let name = name
                        .replace(".to_q.", ".query.")
                        .replace(".to_k.", ".key.")
                        .replace(".to_v.", ".value.")
                        .replace(".to_out.0.", ".proj_attn.");
                    let tensor = if name.contains(".attentions.")
                        && name.ends_with(".weight")
                        && tensor.rank() == 2
                    {
                        tensor.unsqueeze(2).unwrap().unsqueeze(3).unwrap()
                    } else {
                        tensor.clone()
                    };
                    (name, tensor)
                })
                .collect();
            let legacy = AutoEncoderKL::new_with_numerics(
                nn::VarBuilder::from_tensors(legacy_weights, DType::F32, &device),
                3,
                3,
                cfg.clone(),
                VaeNumerics::Diffusers,
            )
            .unwrap();
            assert_eq!(
                max_error(
                    &legacy.encode_moments(&tensors["pixels"]).unwrap(),
                    &precise.encode_moments(&tensors["pixels"]).unwrap()
                ),
                0.
            );
            assert_eq!(
                max_error(
                    &legacy.decode(&tensors["noise"]).unwrap(),
                    &precise.decode(&tensors["noise"]).unwrap()
                ),
                0.
            );
            let original = candle_transformers::models::stable_diffusion::vae::AutoEncoderKL::new(
                vb, 3, 3, cfg,
            )
            .unwrap();
            let moments = model.encode_moments(&tensors["pixels"]).unwrap();
            let mut stages = Vec::new();
            let traced = model
                .encode_moments_with_observer(&tensors["pixels"], |name, value| {
                    stages.push((name.to_string(), value.dims().to_vec()));
                    Ok(())
                })
                .unwrap();
            assert_eq!(max_error(&moments, &traced), 0.);
            assert_eq!(stages.first().unwrap().0, "encoder.conv_in");
            assert_eq!(
                stages.last().unwrap().0,
                if quant {
                    "quant_conv"
                } else {
                    "encoder.conv_out"
                }
            );
            assert!(model
                .encode_moments_with_observer(&tensors["pixels"], |_, _| {
                    candle::bail!("observer cancelled")
                })
                .is_err());
            let mean = model.encode(&tensors["pixels"]).unwrap().mode().unwrap();
            assert_eq!(
                max_error(
                    &mean,
                    &original.encode(&tensors["pixels"]).unwrap().mode().unwrap()
                ),
                0.
            );
            let decoded = model.decode(&tensors["noise"]).unwrap();
            assert_eq!(
                max_error(&decoded, &original.decode(&tensors["noise"]).unwrap()),
                0.
            );
            if quant {
                let posterior = DiagonalGaussianDistribution::new_clamped(&moments).unwrap();
                assert!(
                    max_error(&mean, &tensors["mean"]) < 5e-5,
                    "VAE mean error {}",
                    max_error(&mean, &tensors["mean"])
                );
                assert!(max_error(posterior.std(), &tensors["std"]) < 5e-5);
                let sampled =
                    (posterior.sample_with_noise(&tensors["noise"]).unwrap() * 0.18215).unwrap();
                assert!(max_error(&sampled, &tensors["sampled"]) < 1e-5);
                let decoded = model.decode(&(&sampled / 0.18215).unwrap()).unwrap();
                assert!(max_error(&decoded, &tensors["decoded"]) < 1e-4);
            }
        }
    }

    #[test]
    fn explicit_posterior_noise_matches_diffusers_fixture() {
        let tensors = candle::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-vae-tiny.safetensors"),
            &Device::Cpu,
        )
        .unwrap();
        let logvar = (tensors["std"].log().unwrap() * 2.).unwrap();
        let parameters = Tensor::cat(&[&tensors["mean"], &logvar], 1).unwrap();
        let actual = (DiagonalGaussianDistribution::new_clamped(&parameters)
            .unwrap()
            .sample_with_noise(&tensors["noise"])
            .unwrap()
            * 0.18215)
            .unwrap();
        let error = (actual - &tensors["sampled"])
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(error < 1e-6, "posterior sampling error {error}");
    }

    #[test]
    fn posterior_bounds_prevent_exponential_overflow() {
        let parameters =
            Tensor::from_vec(vec![2f32, 3., -100., 100.], (1, 4, 1, 1), &Device::Cpu).unwrap();
        let noise = Tensor::ones((1, 2, 1, 1), DType::F32, &Device::Cpu).unwrap();
        let sampled = DiagonalGaussianDistribution::new_clamped(&parameters)
            .unwrap()
            .sample_with_noise(&noise)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((sampled[0] - 2.).abs() < 1e-6);
        assert!((sampled[1] - (3. + 10f32.exp())).abs() < 0.01);
    }
}
