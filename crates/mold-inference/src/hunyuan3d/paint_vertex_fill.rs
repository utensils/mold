//! Mesh-connectivity propagation before paint's pixel-space inpainting.

use anyhow::{ensure, Result};

/// Separate position and UV indices preserve split seams from the prepared mesh.
pub struct VertexFillInput<'a> {
    pub width: usize,
    pub height: usize,
    pub texture: &'a [[f32; 3]],
    pub mask: &'a [u8],
    pub positions: &'a [[f32; 3]],
    pub uv: &'a [[f32; 2]],
    pub faces: &'a [[u32; 3]],
    pub uv_faces: &'a [[u32; 3]],
}

/// Port of Tencent 82920d6 mesh_inpaint_processor.cpp:57-163, 213-302.
/// Directed face edges, repeated unseen corners, and in-place smoothing order
/// are part of the reference. Outputs are private until successful completion.
pub fn fill_vertices(
    input: &VertexFillInput<'_>,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<(Vec<[f32; 3]>, Vec<u8>)> {
    checkpoint()?;
    ensure!(
        (1..=4096).contains(&input.width) && (1..=4096).contains(&input.height),
        "invalid vertex fill dimensions"
    );
    ensure!(
        input.texture.len() == input.width * input.height
            && input.mask.len() == input.texture.len(),
        "invalid vertex fill image lengths"
    );
    ensure!(
        input.faces.len() == input.uv_faces.len(),
        "vertex fill face counts differ"
    );
    for values in [input.positions, input.texture] {
        for chunk in values.chunks(4096) {
            checkpoint()?;
            ensure!(
                chunk.iter().flatten().all(|v| v.is_finite()),
                "nonfinite vertex fill input"
            );
        }
    }
    for chunk in input.uv.chunks(4096) {
        checkpoint()?;
        ensure!(
            chunk
                .iter()
                .flatten()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
            "invalid vertex fill UV"
        );
    }
    for (ordinal, (face, uv_face)) in input.faces.iter().zip(input.uv_faces).enumerate() {
        if ordinal % 4096 == 0 {
            checkpoint()?;
        }
        ensure!(
            face.iter().all(|i| (*i as usize) < input.positions.len())
                && uv_face.iter().all(|i| (*i as usize) < input.uv.len()),
            "invalid vertex fill index"
        );
    }
    let texel = |index: u32| {
        let [u, v] = input.uv[index as usize];
        // C++ u multiplication is float; 1.0-v promotes the v path to double.
        let x = (u * (input.width - 1) as f32).round() as usize;
        let y = ((1.0 - f64::from(v)) * (input.height - 1) as f64).round() as usize;
        y * input.width + x
    };
    let mut known = vec![false; input.positions.len()];
    let mut colors = vec![[0f32; 3]; input.positions.len()];
    let mut unseen = Vec::new();
    let mut edges = vec![Vec::new(); input.positions.len()];
    for (face, uv_face) in input.faces.iter().zip(input.uv_faces) {
        checkpoint()?;
        for k in 0..3 {
            let vertex = face[k] as usize;
            let pixel = texel(uv_face[k]);
            if input.mask[pixel] > 0 {
                known[vertex] = true;
                colors[vertex] = input.texture[pixel];
            } else {
                unseen.push(vertex);
            }
            edges[vertex].push(face[(k + 1) % 3] as usize);
        }
    }
    let mut smooth_count = 2;
    let mut last_uncolored = 0;
    while smooth_count > 0 {
        checkpoint()?;
        let mut uncolored = 0;
        for (ordinal, &vertex) in unseen.iter().enumerate() {
            if ordinal % 4096 == 0 {
                checkpoint()?;
            }
            let mut sum = [0f32; 3];
            let mut total = 0f32;
            for (edge, &neighbor) in edges[vertex].iter().enumerate() {
                if edge % 4096 == 0 {
                    checkpoint()?;
                }
                if known[neighbor] {
                    let delta = std::array::from_fn::<_, 3, _>(|c| {
                        f64::from(input.positions[vertex][c] - input.positions[neighbor][c])
                    });
                    let distance =
                        (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
                    let inverse = (1.0 / distance.max(1e-4)) as f32;
                    let weight = inverse * inverse;
                    for (channel, value) in sum.iter_mut().enumerate() {
                        *value += colors[neighbor][channel] * weight;
                    }
                    total += weight;
                }
            }
            if total > 0.0 {
                colors[vertex] = sum.map(|value| value / total);
                ensure!(
                    colors[vertex].iter().all(|value| value.is_finite()),
                    "nonfinite vertex fill result"
                );
                known[vertex] = true;
            } else {
                uncolored += 1;
            }
        }
        if last_uncolored == uncolored {
            smooth_count -= 1;
        } else {
            smooth_count += 1;
        }
        last_uncolored = uncolored;
    }
    checkpoint()?;
    let mut texture = input.texture.to_vec();
    let mut mask = input.mask.to_vec();
    for (face, uv_face) in input.faces.iter().zip(input.uv_faces) {
        checkpoint()?;
        for k in 0..3 {
            let vertex = face[k] as usize;
            if known[vertex] {
                let pixel = texel(uv_face[k]);
                texture[pixel] = colors[vertex];
                mask[pixel] = 255;
            }
        }
    }
    checkpoint()?;
    Ok((texture, mask))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    struct Case {
        name: String,
        width: usize,
        height: usize,
        texture: Vec<[f32; 3]>,
        mask: Vec<u8>,
        positions: Vec<[f32; 3]>,
        uv: Vec<[f32; 2]>,
        faces: Vec<[u32; 3]>,
        uv_faces: Vec<[u32; 3]>,
        expected: Vec<[f32; 3]>,
        expected_mask: Vec<u8>,
    }
    #[test]
    fn vertex_fill_refuses_overflow_and_cancels_duplicate_edges() {
        let texture = [[f32::MAX; 3], [0.; 3]];
        let positions = [[0.; 3]; 3];
        let uv = [[0., 0.], [1., 0.], [1., 0.]];
        let faces = vec![[0, 1, 2]; 4097];
        let mut input = VertexFillInput {
            width: 2,
            height: 1,
            texture: &texture,
            mask: &[255, 0],
            positions: &positions,
            uv: &uv,
            faces: &faces,
            uv_faces: &faces,
        };
        let error = fill_vertices(&input, &mut || Ok(())).unwrap_err();
        assert!(error.to_string().contains("nonfinite vertex fill result"));
        input.mask = &[0, 0];
        let mut calls = 0;
        // Cancel during high-valence traversal, after validation and graph build.
        let error = fill_vertices(&input, &mut || {
            calls += 1;
            anyhow::ensure!(calls < 4110, "cancelled duplicate traversal");
            Ok(())
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "cancelled duplicate traversal");
    }

    #[test]
    fn vertex_fill_matches_tencent_smoothing() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../../tests/fixtures/hunyuan3d/vertex-fill.json"
        ))
        .unwrap();
        for case in fixture.cases {
            let input = VertexFillInput {
                width: case.width,
                height: case.height,
                texture: &case.texture,
                mask: &case.mask,
                positions: &case.positions,
                uv: &case.uv,
                faces: &case.faces,
                uv_faces: &case.uv_faces,
            };
            let mut checkpoints = 0;
            let (texture, mask) = fill_vertices(&input, &mut || {
                checkpoints += 1;
                Ok(())
            })
            .unwrap();
            for cancel_at in 1..=checkpoints {
                let mut calls = 0;
                let error = fill_vertices(&input, &mut || {
                    calls += 1;
                    anyhow::ensure!(calls != cancel_at, "cancelled");
                    Ok(())
                })
                .unwrap_err();
                assert_eq!(error.to_string(), "cancelled");
            }
            assert_eq!(mask, case.expected_mask, "{}", case.name);
            assert!(texture.iter().flatten().all(|v| v.is_finite()));
            let max = texture
                .iter()
                .flatten()
                .zip(case.expected.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max <= 1e-7, "{} max {max}", case.name);
        }
    }
}
