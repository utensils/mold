use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::models::z_image::ZImageTransformer2DModel;

use super::quantized_transformer::QuantizedZImageTransformer2DModel;

/// BF16 or quantized (GGUF) Z-Image transformer.
#[allow(clippy::large_enum_variant)]
pub(crate) enum ZImageTransformer {
    BF16(ZImageTransformer2DModel),
    Quantized(QuantizedZImageTransformer2DModel),
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
            Self::BF16(m) => Ok(m.forward(x, t, cap_feats, cap_mask)?),
            Self::Quantized(m) => Ok(m.forward(x, t, cap_feats, cap_mask)?),
        }
    }
}
