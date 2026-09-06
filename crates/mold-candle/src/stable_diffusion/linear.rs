//! Torch Linear rounding for Diffusers model ports. Parameters have already
//! been rounded to the model dtype before being stored as float32 for opmath.
//! This is opt-in; existing Candle SD components keep their numerical policy.
use candle::{DType, Result, Tensor};
use candle_nn::Module;

pub fn forward(layer: &candle_nn::Linear, input: &Tensor) -> Result<Tensor> {
    // Torch 2.5.1 ATen/native/Linear.cpp:94-120 uses addmm for 2D and
    // contiguous ND inputs, but non-contiguous ND takes matmul then bias add.
    // The spatial UNet's BCHW -> B,HW,C view takes the latter path: its half
    // matrix product must round BEFORE bias. Making that view contiguous or
    // folding bias into float opmath changes its subsequent norm1 cache.
    if input.rank() > 2 && !input.is_contiguous() {
        let layer = candle_nn::Linear::new(
            layer.weight().to_dtype(input.dtype())?,
            layer
                .bias()
                .map(|bias| bias.to_dtype(input.dtype()))
                .transpose()?,
        );
        return layer.forward(input);
    }
    layer
        .forward(&input.to_dtype(DType::F32)?)?
        .to_dtype(input.dtype())
}
