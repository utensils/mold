use candle::{bail, DType, Device, IndexOp, Result, Tensor};

pub const H3_IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const H3_IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub const H3_LATENTS_MEAN: [f32; 24] = [
    0.85809034,
    -0.96065915,
    1.066164,
    -0.50903255,
    -0.2727582,
    -1.3675414,
    -0.2553255,
    -0.26907554,
    -0.5376841,
    -0.04640973,
    0.66573703,
    0.19690128,
    -0.5460608,
    -0.4035342,
    -0.23683025,
    0.25928453,
    -0.30133945,
    0.21134199,
    -1.1206849,
    0.35819334,
    -0.042251438,
    0.260483,
    0.22864093,
    0.7056032,
];

pub const H3_LATENTS_STD: [f32; 24] = [
    1.2223774, 1.2767264, 1.6831775, 1.7549455, 1.5636216, 2.1941435, 0.9653138, 1.0569886,
    0.8419489, 0.7729953, 1.8955938, 0.94684184, 0.79968095, 0.449889, 0.71974, 0.6936293,
    2.961095, 2.76942, 3.0496185, 2.1088054, 3.2762263, 3.1627357, 2.2816813, 2.6127844,
];

/// The released conditioning contract is stochastic and is the default.
/// ComfyUI's mean-only shortcut is exposed only under an explicit name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConditionEncodeMode {
    #[default]
    OfficialFreshSeed42,
    ComfyMeanOnly,
}

/// One RGB condition in the official uint8 input domain.
#[derive(Clone, Debug)]
pub struct Uint8RgbPixels(Tensor);

impl Uint8RgbPixels {
    pub fn new(tensor: Tensor) -> Result<Self> {
        validate_rgb_condition_shape(&tensor)?;
        if tensor.dtype() != DType::U8 {
            bail!(
                "H3 uint8 condition requires U8 pixels, got {:?}; use UnitRgbPixels for [0,1] float input",
                tensor.dtype()
            )
        }
        Ok(Self(tensor))
    }

    pub(crate) fn into_imagenet(self) -> Result<ImageNetPixels> {
        let unit = self.0.to_dtype(DType::F32)?.affine(1.0 / 255.0, 0.0)?;
        normalize_imagenet(unit)
    }
}

/// One RGB condition in an explicitly typed floating-point `[0,1]` domain.
#[derive(Clone, Debug)]
pub struct UnitRgbPixels(Tensor);

impl UnitRgbPixels {
    pub fn new(tensor: Tensor) -> Result<Self> {
        validate_rgb_condition_shape(&tensor)?;
        if !matches!(
            tensor.dtype(),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64
        ) {
            bail!("H3 unit RGB condition requires a floating-point tensor")
        }
        let f32_tensor = tensor.to_dtype(DType::F32)?;
        let min = f32_tensor.min_all()?.to_scalar::<f32>()?;
        let max = f32_tensor.max_all()?.to_scalar::<f32>()?;
        if min < 0.0 || max > 1.0 || !min.is_finite() || !max.is_finite() {
            bail!("H3 unit RGB condition must be finite and within [0,1], got [{min},{max}]")
        }
        Ok(Self(f32_tensor))
    }

    pub(crate) fn into_imagenet(self) -> Result<ImageNetPixels> {
        normalize_imagenet(self.0)
    }
}

/// One RGB condition in ComfyUI's public floating-point `[-1,1]` domain.
/// Keeping this distinct from [`UnitRgbPixels`] prevents a silent 0.5 offset.
#[derive(Clone, Debug)]
pub struct SignedRgbPixels(Tensor);

impl SignedRgbPixels {
    pub fn new(tensor: Tensor) -> Result<Self> {
        validate_rgb_condition_shape(&tensor)?;
        if !matches!(
            tensor.dtype(),
            DType::F16 | DType::BF16 | DType::F32 | DType::F64
        ) {
            bail!("H3 signed RGB condition requires a floating-point tensor")
        }
        let f32_tensor = tensor.to_dtype(DType::F32)?;
        let min = f32_tensor.min_all()?.to_scalar::<f32>()?;
        let max = f32_tensor.max_all()?.to_scalar::<f32>()?;
        if min < -1.0 || max > 1.0 || !min.is_finite() || !max.is_finite() {
            bail!("H3 signed RGB condition must be finite and within [-1,1], got [{min},{max}]")
        }
        Ok(Self(f32_tensor))
    }

    pub(crate) fn into_imagenet(self) -> Result<ImageNetPixels> {
        normalize_imagenet(self.0.affine(0.5, 0.5)?)
    }
}

/// RGB already transformed into the VAE's ImageNet-normalized pixel domain.
#[derive(Clone, Debug)]
pub struct ImageNetPixels(Tensor);

impl ImageNetPixels {
    pub(crate) fn into_tensor(self) -> Tensor {
        self.0
    }
}

fn validate_rgb_condition_shape(tensor: &Tensor) -> Result<()> {
    let (batch, channels, frames, height, width) = tensor.dims5()?;
    if batch != 1 || channels != 3 || frames == 0 || height == 0 || width == 0 {
        bail!(
            "H3 visual condition must be [1,3,T,H,W] with positive dimensions, got {:?}",
            tensor.dims()
        )
    }
    Ok(())
}

fn normalize_imagenet(unit: Tensor) -> Result<ImageNetPixels> {
    let device = unit.device();
    let mean = Tensor::from_slice(&H3_IMAGENET_MEAN, (1, 3, 1, 1, 1), device)?;
    let std = Tensor::from_slice(&H3_IMAGENET_STD, (1, 3, 1, 1, 1), device)?;
    Ok(ImageNetPixels(
        unit.broadcast_sub(&mean)?.broadcast_div(&std)?,
    ))
}

#[derive(Clone, Debug)]
pub(crate) struct DiagonalGaussianDistribution {
    mean: Tensor,
    logvar: Tensor,
}

impl DiagonalGaussianDistribution {
    pub(crate) fn from_moments(moments: &Tensor) -> Result<Self> {
        let (_, channels, _, _, _) = moments.dims5()?;
        if channels == 0 || channels % 2 != 0 {
            bail!("H3 posterior moments require an even nonzero channel count, got {channels}")
        }
        let latent_channels = channels / 2;
        let moments = moments.to_dtype(DType::F32)?;
        Ok(Self {
            mean: moments
                .i((.., ..latent_channels, .., .., ..))?
                .contiguous()?,
            logvar: moments
                .i((.., latent_channels.., .., .., ..))?
                .clamp(-30.0, 20.0)?
                .contiguous()?,
        })
    }

    pub(crate) fn condition_latents(
        &self,
        mode: ConditionEncodeMode,
        latents_mean: &[f32],
        latents_std: &[f32],
    ) -> Result<Tensor> {
        let latent = match mode {
            ConditionEncodeMode::OfficialFreshSeed42 => {
                let std = self.logvar.affine(0.5, 0.0)?.exp()?;
                let noise = fresh_seed42_noise(self.mean.shape())?.to_device(self.mean.device())?;
                // The official generator is CPU-owned and independent from
                // request/denoiser noise. The sampled latent is then rounded
                // through FP16 before normalization.
                self.mean
                    .add(&std.mul(&noise)?)?
                    .to_dtype(DType::F16)?
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
            }
            ConditionEncodeMode::ComfyMeanOnly => self.mean.to_device(&Device::Cpu)?,
        };
        normalize_latents(&latent, latents_mean, latents_std)
    }
}

/// A fresh local generator per condition is the authority boundary. Using a
/// process/global Candle RNG here would couple posterior samples to unrelated
/// model noise and violate the official recipe.
fn fresh_seed42_noise(shape: &candle::Shape) -> Result<Tensor> {
    let values = pytorch_cpu_randn_f32(shape.elem_count(), 42);
    Tensor::from_vec(values, shape.clone(), &Device::Cpu)
}

/// PyTorch's CPU `torch.randn(..., generator=torch.Generator().manual_seed(seed))`
/// recipe for a contiguous FP32 tensor. H3 creates that CPU generator afresh
/// for every condition, even when the VAE itself is on a GPU.
fn pytorch_cpu_randn_f32(size: usize, seed: u64) -> Vec<f32> {
    let mut generator = TorchMt19937::new(seed);
    if size < 16 {
        let mut output = Vec::with_capacity(size);
        let mut cached = None;
        while output.len() < size {
            if let Some(value) = cached.take() {
                output.push(value as f32);
                continue;
            }
            let u1 = generator.uniform_f64();
            let u2 = generator.uniform_f64();
            let radius = (-2.0 * (-u2).ln_1p()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u1;
            cached = Some(radius * theta.sin());
            output.push((radius * theta.cos()) as f32);
        }
        return output;
    }

    let mut output = (0..size)
        .map(|_| generator.uniform_f32())
        .collect::<Vec<_>>();
    let mut offset = 0;
    while offset + 16 <= size {
        torch_normal_fill_16(&mut output[offset..offset + 16]);
        offset += 16;
    }
    // PyTorch deliberately regenerates and recomputes the final 16 values for
    // a non-multiple length rather than transforming a partial block.
    if !size.is_multiple_of(16) {
        let offset = size - 16;
        for value in &mut output[offset..] {
            *value = generator.uniform_f32();
        }
        torch_normal_fill_16(&mut output[offset..]);
    }
    output
}

fn torch_normal_fill_16(data: &mut [f32]) {
    debug_assert_eq!(data.len(), 16);
    for pair in 0..8 {
        let u1 = 1.0 - data[pair];
        let u2 = data[pair + 8];
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        data[pair] = radius * theta.cos();
        data[pair + 8] = radius * theta.sin();
    }
}

/// PyTorch's `at::mt19937_engine`; this is intentionally not Rust's
/// `StdRng`, whose seed-42 stream differs from the released conditioner.
struct TorchMt19937 {
    state: [u32; 624],
    left: usize,
    next: usize,
}

impl TorchMt19937 {
    fn new(seed: u64) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed as u32;
        for index in 1..state.len() {
            state[index] = 1_812_433_253u32
                .wrapping_mul(state[index - 1] ^ (state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        Self {
            state,
            left: 1,
            next: 0,
        }
    }

    fn random_u32(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.next_state();
        }
        let mut value = self.state[self.next];
        self.next += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    fn random_u64(&mut self) -> u64 {
        (u64::from(self.random_u32()) << 32) | u64::from(self.random_u32())
    }

    fn uniform_f32(&mut self) -> f32 {
        const MASK: u32 = (1 << 24) - 1;
        (self.random_u32() & MASK) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    fn uniform_f64(&mut self) -> f64 {
        const MASK: u64 = (1 << 53) - 1;
        (self.random_u64() & MASK) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn next_state(&mut self) {
        const N: usize = 624;
        const M: usize = 397;
        for index in 0..(N - M) {
            self.state[index] =
                self.state[index + M] ^ Self::twist(self.state[index], self.state[index + 1]);
        }
        for index in (N - M)..(N - 1) {
            self.state[index] =
                self.state[index + M - N] ^ Self::twist(self.state[index], self.state[index + 1]);
        }
        self.state[N - 1] = self.state[M - 1] ^ Self::twist(self.state[N - 1], self.state[0]);
        self.left = N;
        self.next = 0;
    }

    fn twist(first: u32, second: u32) -> u32 {
        let mixed = (first & 0x8000_0000) | (second & 0x7fff_ffff);
        (mixed >> 1) ^ if second & 1 == 1 { 0x9908_b0df } else { 0 }
    }
}

pub(crate) fn normalize_latents(
    latent: &Tensor,
    latents_mean: &[f32],
    latents_std: &[f32],
) -> Result<Tensor> {
    let (_, channels, _, _, _) = latent.dims5()?;
    if latents_mean.len() != channels || latents_std.len() != channels {
        bail!(
            "H3 latent statistics require {channels} channels, got mean={} std={}",
            latents_mean.len(),
            latents_std.len()
        )
    }
    if latents_std
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        bail!("H3 latent standard deviations must be finite and positive")
    }
    let mean = Tensor::from_slice(latents_mean, (1, channels, 1, 1, 1), latent.device())?;
    let std = Tensor::from_slice(latents_std, (1, channels, 1, 1, 1), latent.device())?;
    latent.broadcast_sub(&mean)?.broadcast_div(&std)
}

pub(crate) fn denormalize_latents(
    latent: &Tensor,
    latents_mean: &[f32],
    latents_std: &[f32],
) -> Result<Tensor> {
    let (_, channels, _, _, _) = latent.dims5()?;
    if latents_mean.len() != channels || latents_std.len() != channels {
        bail!("H3 latent statistics do not match the decode tensor")
    }
    let mean = Tensor::from_slice(latents_mean, (1, channels, 1, 1, 1), latent.device())?;
    let std = Tensor::from_slice(latents_std, (1, channels, 1, 1, 1), latent.device())?;
    latent.broadcast_mul(&std)?.broadcast_add(&mean)
}

pub(crate) fn denormalize_imagenet(pixels: &Tensor) -> Result<Tensor> {
    let device = pixels.device();
    let mean = Tensor::from_slice(&H3_IMAGENET_MEAN, (1, 3, 1, 1, 1), device)?;
    let std = Tensor::from_slice(&H3_IMAGENET_STD, (1, 3, 1, 1, 1), device)?;
    pixels
        .to_dtype(DType::F32)?
        .broadcast_mul(&std)?
        .broadcast_add(&mean)?
        .clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_pixel_domains_are_distinct_but_normalize_identically() {
        let u8_pixels =
            Tensor::from_vec(vec![0u8, 128, 255], (1, 3, 1, 1, 1), &Device::Cpu).unwrap();
        let unit_pixels = Tensor::from_vec(
            vec![0.0f32, 128.0 / 255.0, 1.0],
            (1, 3, 1, 1, 1),
            &Device::Cpu,
        )
        .unwrap();
        let signed_pixels = Tensor::from_vec(
            vec![-1.0f32, 2.0 * (128.0 / 255.0) - 1.0, 1.0],
            (1, 3, 1, 1, 1),
            &Device::Cpu,
        )
        .unwrap();
        let from_u8 = Uint8RgbPixels::new(u8_pixels)
            .unwrap()
            .into_imagenet()
            .unwrap()
            .into_tensor();
        let from_unit = UnitRgbPixels::new(unit_pixels)
            .unwrap()
            .into_imagenet()
            .unwrap()
            .into_tensor();
        let from_signed = SignedRgbPixels::new(signed_pixels)
            .unwrap()
            .into_imagenet()
            .unwrap()
            .into_tensor();
        let delta = from_u8
            .sub(&from_unit)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap();
        assert!(delta.to_scalar::<f32>().unwrap() < 1e-6);
        let signed_delta = from_u8
            .sub(&from_signed)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap();
        assert!(signed_delta.to_scalar::<f32>().unwrap() < 1e-6);
        assert!(Uint8RgbPixels::new(
            Tensor::zeros((1, 3, 1, 1, 1), DType::F32, &Device::Cpu).unwrap()
        )
        .is_err());
        assert!(
            UnitRgbPixels::new(Tensor::full(1.1f32, (1, 3, 1, 1, 1), &Device::Cpu).unwrap())
                .is_err()
        );
        assert!(SignedRgbPixels::new(
            Tensor::full(-1.1f32, (1, 3, 1, 1, 1), &Device::Cpu).unwrap()
        )
        .is_err());
    }

    #[test]
    fn official_posterior_is_fresh_seed42_and_domain_independent() {
        let means = Tensor::zeros((1, 2, 1, 1, 3), DType::F32, &Device::Cpu).unwrap();
        let logvars = Tensor::zeros((1, 2, 1, 1, 3), DType::F32, &Device::Cpu).unwrap();
        let posterior = DiagonalGaussianDistribution::from_moments(
            &Tensor::cat(&[&means, &logvars], 1).unwrap(),
        )
        .unwrap();
        let first = posterior
            .condition_latents(
                ConditionEncodeMode::OfficialFreshSeed42,
                &[0.0, 0.0],
                &[1.0, 1.0],
            )
            .unwrap();
        // Consume unrelated random values. The next condition must still use
        // its own seed-42 stream rather than global/request RNG state.
        let _ = Tensor::randn(0.0f32, 1.0f32, (4096,), &Device::Cpu).unwrap();
        let second = posterior
            .condition_latents(
                ConditionEncodeMode::OfficialFreshSeed42,
                &[0.0, 0.0],
                &[1.0, 1.0],
            )
            .unwrap();
        assert_eq!(
            first.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            second.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_ne!(
            first.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            posterior
                .condition_latents(ConditionEncodeMode::ComfyMeanOnly, &[0.0, 0.0], &[1.0, 1.0])
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn seed42_noise_matches_pytorch_cpu_reference_stream() {
        // `torch.randn(16, generator=torch.Generator().manual_seed(42))`.
        // The scalar math can differ by a few ULP from PyTorch's AVX2
        // approximation, which is immaterial after the required FP16 round trip.
        let expected = [
            1.9269, 1.4873, 0.9007, -2.1055, 0.6784, -1.2345, -0.0431, -1.6047, -0.7521, 1.6487,
            -0.3925, -1.4036, -0.7279, -0.5594, -0.7688, 0.7624,
        ];
        let actual = pytorch_cpu_randn_f32(16, 42);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 5e-4, "{actual} != {expected}");
        }
    }

    #[test]
    fn fp16_round_trip_happens_before_latent_normalization() {
        let mean = Tensor::from_vec(vec![0.33333334f32], (1, 1, 1, 1, 1), &Device::Cpu).unwrap();
        let logvar = Tensor::full(-30.0f32, (1, 1, 1, 1, 1), &Device::Cpu).unwrap();
        let posterior =
            DiagonalGaussianDistribution::from_moments(&Tensor::cat(&[&mean, &logvar], 1).unwrap())
                .unwrap();
        let actual = posterior
            .condition_latents(ConditionEncodeMode::OfficialFreshSeed42, &[0.1], &[0.2])
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        let rounded = Tensor::from_slice(&[0.33333334f32], 1, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!((actual - (rounded - 0.1) / 0.2).abs() < 1e-3);
    }
}
