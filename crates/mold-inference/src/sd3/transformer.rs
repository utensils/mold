use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::models::mmdit::model::MMDiT;

use super::quantized_mmdit::QuantizedMMDiT;

/// BF16 or quantized (GGUF) SD3 MMDiT transformer.
#[allow(clippy::large_enum_variant)]
pub(crate) enum SD3Transformer {
    BF16(MMDiT),
    Quantized(QuantizedMMDiT),
}

/// SAFETY: SD3Transformer is only ever accessed from a single thread at a time.
/// The server wraps it in a `tokio::sync::Mutex`, and CLI uses it from `spawn_blocking`.
/// The candle `MMDiT` struct contains `dyn JointBlock` and `nn::Sequential` (with `dyn Module`)
/// which are not `Send + Sync`, but our usage pattern guarantees single-threaded access.
unsafe impl Send for SD3Transformer {}
unsafe impl Sync for SD3Transformer {}

impl SD3Transformer {
    /// Forward pass through the MMDiT transformer.
    ///
    /// - `x`: Latent image tensor [N, C, H, W]
    /// - `t`: Diffusion timesteps [N]
    /// - `y`: Vector conditioning (pooled CLIP embeddings) [N, 2048]
    /// - `context`: Text embeddings (CLIP + T5 concat) [N, seq, 4096]
    /// - `skip_layers`: Optional layer indices to skip (for SLG)
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        y: &Tensor,
        context: &Tensor,
        skip_layers: Option<&[usize]>,
    ) -> Result<Tensor> {
        match self {
            Self::BF16(m) => Ok(m.forward(x, t, y, context, skip_layers)?),
            Self::Quantized(m) => m.forward(x, t, y, context, skip_layers),
        }
    }
}
