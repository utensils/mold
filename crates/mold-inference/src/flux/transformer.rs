use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::models::flux;

/// BF16 or quantized (GGUF) FLUX transformer.
#[allow(clippy::large_enum_variant)]
pub(crate) enum FluxTransformer {
    BF16(flux::model::Flux),
    Quantized(flux::quantized_model::Flux),
}

impl FluxTransformer {
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        vec_: &Tensor,
        timesteps: &[f64],
        guidance: f64,
    ) -> Result<Tensor> {
        match self {
            Self::BF16(m) => Ok(flux::sampling::denoise(
                m, img, img_ids, txt, txt_ids, vec_, timesteps, guidance,
            )?),
            Self::Quantized(m) => Ok(flux::sampling::denoise(
                m, img, img_ids, txt, txt_ids, vec_, timesteps, guidance,
            )?),
        }
    }
}
