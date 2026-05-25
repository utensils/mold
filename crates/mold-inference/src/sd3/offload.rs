use anyhow::Result;
use candle_core::{DType, Device, Module, Shape, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::mmdit::blocks::{
    ContextQkvOnlyJointBlock, FinalLayer, JointBlock, MMDiTJointBlock, MMDiTXJointBlock,
};
use candle_transformers::models::mmdit::embedding::{
    PatchEmbedder, PositionEmbedder, TimestepEmbedder, Unpatchifier, VectorEmbedder,
};
use candle_transformers::models::mmdit::model::Config;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone)]
struct CpuTensorBackend {
    tensors: Arc<HashMap<String, Tensor>>,
}

impl CpuTensorBackend {
    fn new(tensors: Arc<HashMap<String, Tensor>>) -> Self {
        Self { tensors }
    }

    fn load(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::msg(format!("missing SD3 MMDiT tensor {name}")))?;
        tensor.to_device(dev)?.to_dtype(dtype)
    }
}

impl candle_nn::var_builder::SimpleBackend for CpuTensorBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _init: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.load(name, dtype, dev)?;
        if tensor.shape() != &shape {
            return Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: shape,
                got: tensor.shape().clone(),
            });
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.load(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sd3StreamingBlock {
    Joint(usize),
    FinalJoint(usize),
}

pub(crate) fn sd3_streaming_block_plan(cfg: &Config) -> Vec<Sd3StreamingBlock> {
    let mut blocks = Vec::with_capacity(cfg.depth);
    blocks.extend((0..cfg.depth.saturating_sub(1)).map(Sd3StreamingBlock::Joint));
    if cfg.depth > 0 {
        blocks.push(Sd3StreamingBlock::FinalJoint(cfg.depth - 1));
    }
    blocks
}

pub(crate) struct OffloadedMMDiT {
    cfg: Config,
    block_plan: Vec<Sd3StreamingBlock>,
    tensors: Arc<HashMap<String, Tensor>>,
    dtype: DType,
    device: Device,
    patch_embedder: PatchEmbedder,
    pos_embedder: PositionEmbedder,
    timestep_embedder: TimestepEmbedder,
    vector_embedder: VectorEmbedder,
    context_embedder: candle_nn::Linear,
    final_layer: FinalLayer,
    unpatchifier: Unpatchifier,
}

impl OffloadedMMDiT {
    pub(crate) fn new(
        cfg: &Config,
        tensors: Arc<HashMap<String, Tensor>>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let hidden_size = cfg.head_size * cfg.depth;
        let vb = Self::var_builder(tensors.clone(), dtype, device).pp("model.diffusion_model");
        let patch_embedder = PatchEmbedder::new(
            cfg.patch_size,
            cfg.in_channels,
            hidden_size,
            vb.pp("x_embedder"),
        )?;
        let pos_embedder = PositionEmbedder::new(
            hidden_size,
            cfg.patch_size,
            cfg.pos_embed_max_size,
            vb.clone(),
        )?;
        let timestep_embedder = TimestepEmbedder::new(
            hidden_size,
            cfg.frequency_embedding_size,
            vb.pp("t_embedder"),
        )?;
        let vector_embedder =
            VectorEmbedder::new(cfg.adm_in_channels, hidden_size, vb.pp("y_embedder"))?;
        let context_embedder = candle_nn::linear(
            cfg.context_embed_size,
            hidden_size,
            vb.pp("context_embedder"),
        )?;
        let final_layer = FinalLayer::new(
            hidden_size,
            cfg.patch_size,
            cfg.out_channels,
            vb.pp("final_layer"),
        )?;
        let unpatchifier = Unpatchifier::new(cfg.patch_size, cfg.out_channels)?;

        Ok(Self {
            cfg: cfg.clone(),
            block_plan: sd3_streaming_block_plan(cfg),
            tensors,
            dtype,
            device: device.clone(),
            patch_embedder,
            pos_embedder,
            timestep_embedder,
            vector_embedder,
            context_embedder,
            final_layer,
            unpatchifier,
        })
    }

    fn var_builder<'a>(
        tensors: Arc<HashMap<String, Tensor>>,
        dtype: DType,
        device: &Device,
    ) -> VarBuilder<'a> {
        VarBuilder::from_backend(
            Box::new(CpuTensorBackend::new(tensors)),
            dtype,
            device.clone(),
        )
    }

    fn block_var_builder(&self, idx: usize) -> VarBuilder<'_> {
        Self::var_builder(self.tensors.clone(), self.dtype, &self.device)
            .pp("model.diffusion_model")
            .pp(format!("joint_blocks.{idx}"))
    }

    fn joint_block(&self, idx: usize) -> Result<Box<dyn JointBlock>> {
        let hidden_size = self.cfg.head_size * self.cfg.depth;
        let block_vb = self.block_var_builder(idx);
        let block: Box<dyn JointBlock> = if block_vb
            .pp("x_block")
            .pp("attn2")
            .contains_tensor("qkv.weight")
        {
            Box::new(MMDiTXJointBlock::new(
                hidden_size,
                self.cfg.depth,
                false,
                block_vb,
            )?)
        } else {
            Box::new(MMDiTJointBlock::new(
                hidden_size,
                self.cfg.depth,
                false,
                block_vb,
            )?)
        };
        Ok(block)
    }

    fn final_joint_block(&self, idx: usize) -> Result<ContextQkvOnlyJointBlock> {
        Ok(ContextQkvOnlyJointBlock::new(
            self.cfg.head_size * self.cfg.depth,
            self.cfg.depth,
            false,
            self.block_var_builder(idx),
        )?)
    }

    pub(crate) fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        y: &Tensor,
        context: &Tensor,
        skip_layers: Option<&[usize]>,
    ) -> Result<Tensor> {
        let x = x.to_device(&self.device)?;
        let t = t.to_device(&self.device)?;
        let y = y.to_device(&self.device)?;
        let context = context.to_device(&self.device)?;
        let h = x.dim(D::Minus2)?;
        let w = x.dim(D::Minus1)?;
        let cropped_pos_embed = self.pos_embedder.get_cropped_pos_embed(h, w)?;
        let mut x = self
            .patch_embedder
            .forward(&x)?
            .broadcast_add(&cropped_pos_embed)?;
        let c = self.timestep_embedder.forward(&t)?;
        let y = self.vector_embedder.forward(&y)?;
        let c = (c + y)?;
        let mut context = self.context_embedder.forward(&context)?;

        for block in &self.block_plan {
            match *block {
                Sd3StreamingBlock::Joint(idx) => {
                    if skip_layers.is_some_and(|layers| layers.contains(&idx)) {
                        continue;
                    }
                    let block = self.joint_block(idx)?;
                    (context, x) = block.forward(&context, &x, &c)?;
                }
                Sd3StreamingBlock::FinalJoint(idx) => {
                    let block = self.final_joint_block(idx)?;
                    x = block.forward(&context, &x, &c)?;
                }
            }
        }

        let x = self.final_layer.forward(&x, &c)?;
        let x = self.unpatchifier.unpatchify(&x, h, w)?;
        Ok(x.narrow(2, 0, h)?.narrow(3, 0, w)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sd3_streaming_block_plan_preserves_reference_order() {
        let mut cfg = Config::sd3_5_large();
        cfg.depth = 4;

        assert_eq!(
            sd3_streaming_block_plan(&cfg),
            vec![
                Sd3StreamingBlock::Joint(0),
                Sd3StreamingBlock::Joint(1),
                Sd3StreamingBlock::Joint(2),
                Sd3StreamingBlock::FinalJoint(3),
            ]
        );
    }
}
