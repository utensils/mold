//! Named Hunyuan3D 2mv conditioning.
//!
//! Tencent reference: `hy3dgen/shapegen/models/conditioner.py` at
//! `f8db63096c8282cb27354314d896feba5ba6ff8a`. Camera slots are semantic:
//! missing inputs retain their original position index and present inputs are
//! concatenated in front, left, back, right order.

use candle_core::{Result, Tensor};
use mold_core::GenerationImageReferenceRole as Role;

const VIEW_COUNT: usize = 4;
const POSITION_BASE: f64 = 10_000.0;

pub(crate) const fn view_slot(role: Role) -> usize {
    match role {
        Role::Front => 0,
        Role::Left => 1,
        Role::Back => 2,
        Role::Right => 3,
    }
}

fn position_embedding(width: usize, role: Role) -> Result<Vec<f32>> {
    if width == 0 || !width.is_multiple_of(2) {
        candle_core::bail!("Hunyuan3D 2mv context width must be positive and even")
    }
    let half = width / 2;
    let position = view_slot(role) as f64;
    let mut embedding = Vec::with_capacity(width);
    for index in 0..half {
        let omega = 1.0 / POSITION_BASE.powf(index as f64 / half as f64);
        embedding.push((position * omega).sin() as f32);
    }
    for index in 0..half {
        let omega = 1.0 / POSITION_BASE.powf(index as f64 / half as f64);
        embedding.push((position * omega).cos() as f32);
    }
    Ok(embedding)
}

/// Add each semantic camera embedding and concatenate the present views.
///
/// The caller may supply views in request order. This function canonicalizes
/// them before concatenation and refuses ambiguity again at the inference
/// boundary. Each encoded input is `[batch, tokens, width]`.
pub(crate) fn compose_view_conditioning(mut views: Vec<(Role, Tensor)>) -> Result<Tensor> {
    if views.is_empty() || views.len() > VIEW_COUNT {
        candle_core::bail!("Hunyuan3D 2mv requires one to four named views")
    }
    views.sort_by_key(|(role, _)| view_slot(*role));
    if views.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        candle_core::bail!("Hunyuan3D 2mv received a duplicate camera view")
    }

    let (batch, _, width) = views[0].1.dims3()?;
    let dtype = views[0].1.dtype();
    let device = views[0].1.device().clone();
    let mut conditioned = Vec::with_capacity(views.len());
    for (role, tokens) in views {
        let (view_batch, _, view_width) = tokens.dims3()?;
        if view_batch != batch
            || view_width != width
            || tokens.dtype() != dtype
            || !tokens.device().same_device(&device)
        {
            candle_core::bail!(
                "Hunyuan3D 2mv encoded views must share batch, width, dtype and device"
            )
        }
        let position = Tensor::from_vec(position_embedding(width, role)?, (1, 1, width), &device)?
            .to_dtype(dtype)?;
        conditioned.push(tokens.broadcast_add(&position)?);
    }
    Tensor::cat(&conditioned.iter().collect::<Vec<_>>(), 1)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};
    use mold_core::GenerationImageReferenceRole as Role;

    use super::compose_view_conditioning;

    fn tokens(value: f32) -> Tensor {
        Tensor::new(&[[[value, value, value, value]]], &Device::Cpu).unwrap()
    }

    #[test]
    fn every_nonempty_subset_keeps_fixed_slot_indices_and_canonical_order() {
        let roles = [Role::Front, Role::Left, Role::Back, Role::Right];
        for mask in 1_u8..16 {
            let mut selected = roles
                .iter()
                .enumerate()
                .filter(|(index, _)| mask & (1 << index) != 0)
                .map(|(index, role)| (*role, tokens(index as f32 * 10.0)))
                .collect::<Vec<_>>();
            selected.reverse();

            let output = compose_view_conditioning(selected).unwrap();
            assert_eq!(output.dims(), &[1, mask.count_ones() as usize, 4]);
            let output = output.to_vec3::<f32>().unwrap();
            let mut output_index = 0;
            for (slot, _) in roles.iter().enumerate() {
                if mask & (1 << slot) == 0 {
                    continue;
                }
                let row = &output[0][output_index];
                let base = slot as f32 * 10.0;
                assert!((row[0] - (base + (slot as f32).sin())).abs() < 1e-6);
                assert!((row[1] - (base + (slot as f32 * 0.01).sin())).abs() < 1e-6);
                assert!((row[2] - (base + (slot as f32).cos())).abs() < 1e-6);
                assert!((row[3] - (base + (slot as f32 * 0.01).cos())).abs() < 1e-6);
                output_index += 1;
            }
        }
    }

    #[test]
    fn duplicate_or_empty_view_sets_are_refused_at_the_engine_boundary() {
        assert!(compose_view_conditioning(Vec::new()).is_err());
        assert!(compose_view_conditioning(vec![
            (Role::Front, tokens(1.0)),
            (Role::Front, tokens(2.0)),
        ])
        .is_err());
    }
}
