use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder as DenseVarBuilder;
use candle_transformers::{
    models::z_image::{Config, ZImageTransformer2DModel},
    quantized_var_builder::VarBuilder as QuantizedVarBuilder,
};

fn dequantized_tensor(vb: &QuantizedVarBuilder, name: &str, dtype: DType) -> Result<Tensor> {
    let tensor = vb
        .get_no_shape(name)
        .with_context(|| format!("missing GGUF tensor {name}"))?
        .dequantize(vb.device())
        .with_context(|| format!("failed to dequantize GGUF tensor {name}"))?;
    if tensor.dtype() == dtype {
        Ok(tensor)
    } else {
        tensor
            .to_dtype(dtype)
            .with_context(|| format!("failed to cast GGUF tensor {name} to {dtype:?}"))
    }
}

fn prepare_linear_weight(weight: &Tensor) -> Result<Tensor> {
    if weight.rank() != 2 {
        bail!("expected 2D linear weight, got shape {:?}", weight.shape());
    }
    weight.contiguous().map_err(Into::into)
}

fn split_qkv_weight(weight: &Tensor, dim: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let (rows, _) = weight.dims2()?;
    if rows != dim * 3 {
        bail!(
            "expected fused QKV weight with {} rows, got {}",
            dim * 3,
            rows
        );
    }
    let q = weight.narrow(0, 0, dim)?.contiguous()?;
    let k = weight.narrow(0, dim, dim)?.contiguous()?;
    let v = weight.narrow(0, dim * 2, dim)?.contiguous()?;
    Ok((q, k, v))
}

fn insert_tensor(
    tensors: &mut HashMap<String, Tensor>,
    target_name: impl Into<String>,
    tensor: Tensor,
) {
    tensors.insert(target_name.into(), tensor);
}

fn insert_scalar_or_norm(
    tensors: &mut HashMap<String, Tensor>,
    target_name: impl Into<String>,
    vb: &QuantizedVarBuilder,
    source_name: &str,
    dtype: DType,
) -> Result<()> {
    insert_tensor(
        tensors,
        target_name,
        dequantized_tensor(vb, source_name, dtype)?,
    );
    Ok(())
}

fn insert_linear_weight(
    tensors: &mut HashMap<String, Tensor>,
    target_name: impl Into<String>,
    vb: &QuantizedVarBuilder,
    source_name: &str,
    dtype: DType,
) -> Result<()> {
    let weight = dequantized_tensor(vb, source_name, dtype)?;
    insert_tensor(tensors, target_name, prepare_linear_weight(&weight)?);
    Ok(())
}

fn insert_pad_token(
    tensors: &mut HashMap<String, Tensor>,
    target_name: impl Into<String>,
    vb: &QuantizedVarBuilder,
    source_name: &str,
    dtype: DType,
) -> Result<()> {
    let token = dequantized_tensor(vb, source_name, dtype)?;
    let token = match token.rank() {
        1 => token.unsqueeze(0)?,
        2 => token,
        _ => bail!(
            "expected 1D or 2D pad token for {source_name}, got {:?}",
            token.shape()
        ),
    };
    insert_tensor(tensors, target_name, token);
    Ok(())
}

fn insert_transformer_block(
    tensors: &mut HashMap<String, Tensor>,
    vb: &QuantizedVarBuilder,
    source_prefix: &str,
    target_prefix: &str,
    modulation: bool,
    cfg: &Config,
    dtype: DType,
) -> Result<()> {
    let qkv = dequantized_tensor(vb, &format!("{source_prefix}.attention.qkv.weight"), dtype)?;
    let qkv = prepare_linear_weight(&qkv)?;
    let (q, k, v) = split_qkv_weight(&qkv, cfg.dim)?;
    insert_tensor(tensors, format!("{target_prefix}.attention.to_q.weight"), q);
    insert_tensor(tensors, format!("{target_prefix}.attention.to_k.weight"), k);
    insert_tensor(tensors, format!("{target_prefix}.attention.to_v.weight"), v);

    insert_linear_weight(
        tensors,
        format!("{target_prefix}.attention.to_out.0.weight"),
        vb,
        &format!("{source_prefix}.attention.out.weight"),
        dtype,
    )?;
    insert_scalar_or_norm(
        tensors,
        format!("{target_prefix}.attention.norm_q.weight"),
        vb,
        &format!("{source_prefix}.attention.q_norm.weight"),
        dtype,
    )?;
    insert_scalar_or_norm(
        tensors,
        format!("{target_prefix}.attention.norm_k.weight"),
        vb,
        &format!("{source_prefix}.attention.k_norm.weight"),
        dtype,
    )?;

    for norm_name in [
        "attention_norm1",
        "attention_norm2",
        "ffn_norm1",
        "ffn_norm2",
    ] {
        insert_scalar_or_norm(
            tensors,
            format!("{target_prefix}.{norm_name}.weight"),
            vb,
            &format!("{source_prefix}.{norm_name}.weight"),
            dtype,
        )?;
    }

    for weight_name in ["w1", "w2", "w3"] {
        insert_linear_weight(
            tensors,
            format!("{target_prefix}.feed_forward.{weight_name}.weight"),
            vb,
            &format!("{source_prefix}.feed_forward.{weight_name}.weight"),
            dtype,
        )?;
    }

    if modulation {
        insert_linear_weight(
            tensors,
            format!("{target_prefix}.adaLN_modulation.0.weight"),
            vb,
            &format!("{source_prefix}.adaLN_modulation.0.weight"),
            dtype,
        )?;
        insert_scalar_or_norm(
            tensors,
            format!("{target_prefix}.adaLN_modulation.0.bias"),
            vb,
            &format!("{source_prefix}.adaLN_modulation.0.bias"),
            dtype,
        )?;
    }

    Ok(())
}

pub(crate) fn load_gguf_dense_transformer(
    cfg: &Config,
    dtype: DType,
    vb: QuantizedVarBuilder,
) -> Result<ZImageTransformer2DModel> {
    let device = vb.device().clone();
    let mut tensors = HashMap::new();

    for layer in ["0", "2"] {
        insert_linear_weight(
            &mut tensors,
            format!("t_embedder.mlp.{layer}.weight"),
            &vb,
            &format!("t_embedder.mlp.{layer}.weight"),
            dtype,
        )?;
        insert_scalar_or_norm(
            &mut tensors,
            format!("t_embedder.mlp.{layer}.bias"),
            &vb,
            &format!("t_embedder.mlp.{layer}.bias"),
            dtype,
        )?;
    }

    insert_scalar_or_norm(
        &mut tensors,
        "cap_embedder.0.weight",
        &vb,
        "cap_embedder.0.weight",
        dtype,
    )?;
    insert_linear_weight(
        &mut tensors,
        "cap_embedder.1.weight",
        &vb,
        "cap_embedder.1.weight",
        dtype,
    )?;
    insert_scalar_or_norm(
        &mut tensors,
        "cap_embedder.1.bias",
        &vb,
        "cap_embedder.1.bias",
        dtype,
    )?;

    insert_linear_weight(
        &mut tensors,
        "all_x_embedder.2-1.weight",
        &vb,
        "x_embedder.weight",
        dtype,
    )?;
    insert_scalar_or_norm(
        &mut tensors,
        "all_x_embedder.2-1.bias",
        &vb,
        "x_embedder.bias",
        dtype,
    )?;
    insert_pad_token(&mut tensors, "x_pad_token", &vb, "x_pad_token", dtype)?;
    insert_pad_token(&mut tensors, "cap_pad_token", &vb, "cap_pad_token", dtype)?;

    insert_linear_weight(
        &mut tensors,
        "all_final_layer.2-1.linear.weight",
        &vb,
        "final_layer.linear.weight",
        dtype,
    )?;
    insert_scalar_or_norm(
        &mut tensors,
        "all_final_layer.2-1.linear.bias",
        &vb,
        "final_layer.linear.bias",
        dtype,
    )?;
    insert_linear_weight(
        &mut tensors,
        "all_final_layer.2-1.adaLN_modulation.1.weight",
        &vb,
        "final_layer.adaLN_modulation.1.weight",
        dtype,
    )?;
    insert_scalar_or_norm(
        &mut tensors,
        "all_final_layer.2-1.adaLN_modulation.1.bias",
        &vb,
        "final_layer.adaLN_modulation.1.bias",
        dtype,
    )?;

    for i in 0..cfg.n_refiner_layers {
        insert_transformer_block(
            &mut tensors,
            &vb,
            &format!("noise_refiner.{i}"),
            &format!("noise_refiner.{i}"),
            true,
            cfg,
            dtype,
        )?;
        insert_transformer_block(
            &mut tensors,
            &vb,
            &format!("context_refiner.{i}"),
            &format!("context_refiner.{i}"),
            false,
            cfg,
            dtype,
        )?;
    }

    for i in 0..cfg.n_layers {
        insert_transformer_block(
            &mut tensors,
            &vb,
            &format!("layers.{i}"),
            &format!("layers.{i}"),
            true,
            cfg,
            dtype,
        )?;
    }

    let vb = DenseVarBuilder::from_tensors(tensors, dtype, &device);
    ZImageTransformer2DModel::new(cfg, vb)
        .context("failed to build dense Z-Image transformer from GGUF weights")
}

#[cfg(test)]
mod tests {
    use super::{prepare_linear_weight, split_qkv_weight};
    use candle_core::{Device, Tensor};

    #[test]
    fn prepare_linear_weight_keeps_dense_layout() {
        let weight =
            Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (2, 3), &Device::Cpu).expect("weight");
        let prepared = prepare_linear_weight(&weight).expect("prepare");
        let values = prepared.to_vec2::<f32>().expect("values");
        assert_eq!(values, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }

    #[test]
    fn split_qkv_weight_slices_along_output_rows() {
        let weight = Tensor::from_vec(
            (0..24).map(|v| v as f32).collect::<Vec<_>>(),
            (6, 4),
            &Device::Cpu,
        )
        .expect("weight");
        let (q, k, v) = split_qkv_weight(&weight, 2).expect("split");
        assert_eq!(
            q.to_vec2::<f32>().expect("q"),
            vec![vec![0.0, 1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0, 7.0]]
        );
        assert_eq!(
            k.to_vec2::<f32>().expect("k"),
            vec![vec![8.0, 9.0, 10.0, 11.0], vec![12.0, 13.0, 14.0, 15.0]]
        );
        assert_eq!(
            v.to_vec2::<f32>().expect("v"),
            vec![vec![16.0, 17.0, 18.0, 19.0], vec![20.0, 21.0, 22.0, 23.0]]
        );
    }
}
