use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::models::z_image::ZImageTransformer2DModel;

/// Dense Z-Image transformer, regardless of original weight source.
pub(crate) enum ZImageTransformer {
    Dense(ZImageTransformer2DModel),
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
        }
    }
}
