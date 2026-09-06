//! Tencent paint position-map pyramid, modules.py:203-275 at 82920d64.
//! Half rounding and the dtype-dependent mutation between scales are part of
//! the checkpoint's conditioning protocol. Caller-owned maps remain unchanged.
use candle_core::{DType, Result, Tensor};
use std::collections::HashMap;

pub fn position_pyramid(maps: &Tensor, latent_size: usize) -> Result<HashMap<usize, Tensor>> {
    let dims = maps.dims();
    if dims.len() != 5
        || dims[0] == 0
        || dims[1] == 0
        || dims[2] != 3
        || !(8..=64).contains(&latent_size)
        || !latent_size.is_power_of_two()
        || dims[3] == 0
        || dims[4] == 0
        || dims[3] > 2048
        || dims[4] > 2048
        || !dims[3].is_multiple_of(latent_size)
        || !dims[4].is_multiple_of(latent_size)
        || !matches!(maps.dtype(), DType::F16 | DType::F32)
    {
        candle_core::bail!("invalid paint position-map dimensions or dtype")
    }
    let (batch, views, height, width) = (dims[0], dims[1], dims[3], dims[4]);
    let mut working = maps.to_dtype(DType::F16)?;
    let mut pyramid = HashMap::new();
    for grid in [
        latent_size,
        latent_size / 2,
        latent_size / 4,
        latent_size / 8,
    ] {
        // .half() copies F32 maps afresh each scale, but aliases F16 maps.
        if maps.dtype() == DType::F32 {
            working = maps.to_dtype(DType::F16)?;
        }
        let valid = working.ne(1.)?.to_dtype(DType::F32)?.min_keepdim(2)?;
        working = working
            .to_dtype(DType::F32)?
            .broadcast_mul(&valid)?
            .to_dtype(DType::F16)?;
        let (patch_h, patch_w) = (height / grid, width / grid);
        let reduce = |tensor: &Tensor, channels: usize| -> Result<Tensor> {
            tensor
                .reshape(&[batch, views, channels, grid, patch_h, grid, patch_w][..])?
                .permute(&[0, 1, 3, 5, 2, 4, 6][..])?
                .contiguous()?
                .reshape((batch, views, grid, grid, channels, patch_h * patch_w))?
                .sum(5)
        };
        let sum = reduce(&working.to_dtype(DType::F32)?, 3)?
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?;
        let counts = reduce(&valid, 1)?;
        // TensorIterator promotes the int64 count to the half numerator's dtype.
        let denominator = counts
            .clamp(1., f32::MAX as f64)?
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?;
        let mean = sum.broadcast_div(&denominator)?.to_dtype(DType::F16)?;
        let enough = counts
            .ge((patch_h * patch_w / 16) as f64)?
            .broadcast_as(mean.shape())?;
        let mean = enough
            .where_cond(&mean, &mean.zeros_like()?)?
            .to_dtype(DType::F32)?
            .clamp(0., 1.)?;
        let scaled = mean
            .affine((grid * 8 - 1) as f64, 0.)?
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?;
        // Candle's generic round has backend-dependent tie behavior; explicitly
        // use IEEE ties-to-even for these small cached integer coordinate tables.
        let values = scaled.flatten_all()?.to_vec1::<f32>()?;
        if values.iter().any(|v| !v.is_finite()) {
            candle_core::bail!("nonfinite paint position map")
        }
        let indices = values
            .into_iter()
            .map(|v| v.round_ties_even() as i64)
            .collect::<Vec<_>>();
        pyramid.insert(
            views * grid * grid,
            Tensor::from_vec(indices, (batch, views * grid * grid, 3), maps.device())?,
        );
    }
    Ok(pyramid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn paint_position_fixture_matches_both_dtypes() -> Result<()> {
        let fixture = candle_core::safetensors::load_buffer(
            include_bytes!("../../../../tests/fixtures/hunyuan3d/paint-positions.safetensors"),
            &Device::Cpu,
        )?;
        for dtype in ["f32", "f16"] {
            let maps = &fixture[&format!("{dtype}.input.position_maps")];
            let pyramid = position_pyramid(maps, 8)?;
            for (tokens, value) in pyramid {
                let expected = &fixture[&format!("{dtype}.cache.positions.{tokens}")];
                assert_eq!(
                    value.flatten_all()?.to_vec1::<i64>()?,
                    expected.flatten_all()?.to_vec1::<i64>()?,
                    "{dtype} tokens={tokens}"
                );
            }
        }
        Ok(())
    }
    #[test]
    fn paint_position_count_rounds_to_half_before_division() -> Result<()> {
        let mut pixels = vec![1f32; 3 * 64 * 64];
        for channel in 0..3 {
            pixels[channel * 4096..channel * 4096 + 2048].fill(0.5);
            pixels[channel * 4096 + 2048] = 0.;
        }
        let maps = Tensor::from_vec(pixels, (1, 1, 3, 64, 64), &Device::Cpu)?;
        let pyramid = position_pyramid(&maps, 8)?;
        // Torch2.5.1: sum1024 / half(count2049)=.5; round(.5*7)=4.
        assert_eq!(pyramid[&1].flatten_all()?.to_vec1::<i64>()?, vec![4, 4, 4]);
        Ok(())
    }
    #[test]
    #[ignore = "requires retained Tencent position-map capture"]
    fn paint_position_pyramid_matches_tencent() -> Result<()> {
        let root = std::env::var("MOLD_PAINT_UNET_ORACLE").expect("oracle directory");
        let device = if std::env::var_os("MOLD_PAINT_POSITIONS_CUDA").is_some() {
            Device::new_cuda(0)?
        } else {
            Device::Cpu
        };
        let fixture = candle_core::safetensors::load(
            std::path::Path::new(&root).join("paint-unet.safetensors"),
            &device,
        )?;
        let maps = &fixture["input.position_maps"];
        let before = maps.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let pyramid = position_pyramid(maps, fixture["input.sample"].dim(4)?)?;
        assert_eq!(pyramid.len(), 4);
        for (tokens, value) in pyramid {
            let expected = &fixture[&format!("cache.positions.{tokens}")];
            assert_eq!(value.dims(), expected.dims());
            let actual = value.flatten_all()?.to_vec1::<i64>()?;
            let expected = expected.flatten_all()?.to_vec1::<i64>()?;
            let differences = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            eprintln!(
                "position {tokens}: mismatches={differences}/{}",
                actual.len()
            );
            assert_eq!(differences, 0);
        }
        assert_eq!(
            before,
            maps.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }
}
