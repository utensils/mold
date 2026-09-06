//! Material projection, streaming bake and texture-fill composition.

use super::{
    paint_back_sample::{back_sample, BackSampleView},
    paint_bake::{BakedTexture, TextureBaker},
    paint_camera::CameraGeometry,
    paint_fill::fill_texture,
    paint_pixels::PaintMaterials,
    paint_raster::PreparedMesh,
    paint_uv::UvGeometry,
    paint_vertex_fill::VertexFillInput,
    paint_views::PaintView,
};
use anyhow::{ensure, Result};
use image::RgbImage;

pub struct BakedMaterials {
    pub albedo: BakedTexture,
    pub metallic_roughness: BakedTexture,
}

pub struct MaterialTextures {
    pub albedo: RgbImage,
    pub metallic_roughness: RgbImage,
}

/// Share camera geometry across both streams, but accumulate each material
/// independently in the exact view order. Inputs are already upscaled to 2048.
pub fn bake_materials(
    mesh: &PreparedMesh,
    materials: &PaintMaterials,
    views: &[PaintView],
    size: u32,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<BakedMaterials> {
    checkpoint()?;
    ensure!(
        [1024, 2048, 4096].contains(&size),
        "invalid material texture size"
    );
    ensure!(
        (1..=6).contains(&views.len())
            && materials.albedo.len() == views.len()
            && materials.metallic_roughness.len() == views.len(),
        "material view counts differ"
    );
    ensure!(
        materials
            .albedo
            .iter()
            .chain(&materials.metallic_roughness)
            .all(|image| image.dimensions() == (2048, 2048)),
        "material baking requires 2048x2048 views"
    );
    ensure!(
        views.iter().all(|v| v.elevation.is_finite()
            && v.elevation.abs() <= 90.
            && v.azimuth.is_finite()
            && v.weight.is_finite()
            && v.weight >= 0.),
        "invalid material camera"
    );
    let mut uv = UvGeometry::extract(mesh, size, checkpoint)?;
    // Face normals are not consumed by the qualified back-sample path.
    uv.normals = Vec::new();
    let mut albedo = TextureBaker::new(size, checkpoint)?;
    let mut mr = TextureBaker::new(size, checkpoint)?;
    for (index, view) in views.iter().enumerate() {
        checkpoint()?;
        let geometry =
            CameraGeometry::render(mesh, view.elevation, view.azimuth, 2048, checkpoint)?;
        let reliability = geometry.reliability(checkpoint)?;
        let projected = uv.project(view.elevation, view.azimuth, checkpoint)?;
        for (image, baker) in [
            (&materials.albedo[index], &mut albedo),
            (&materials.metallic_roughness[index], &mut mr),
        ] {
            let mut colors = Vec::with_capacity(2048 * 2048);
            for (pixel, rgb) in image.pixels().enumerate() {
                if pixel.is_multiple_of(4096) {
                    checkpoint()?;
                }
                colors.push(rgb.0.map(|v| f32::from(v) / 255.));
            }
            let sampled = back_sample(
                &BackSampleView {
                    size: 2048,
                    colors: &colors,
                    depth: &geometry.depth,
                    visible: &reliability.visible,
                    cosine: &reliability.cosine,
                    boundary: &reliability.boundary,
                },
                &projected,
                &uv.texels,
                size,
                checkpoint,
            )?;
            baker.add_view(&sampled.colors, &sampled.cosine, view.weight, checkpoint)?;
        }
    }
    Ok(BakedMaterials {
        albedo: albedo.finish(checkpoint)?,
        metallic_roughness: mr.finish(checkpoint)?,
    })
}

type FillGeometry = (Vec<[f32; 3]>, Vec<[f32; 2]>);
fn fill_geometry(
    mesh: &PreparedMesh,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<FillGeometry> {
    checkpoint()?;
    mesh.mesh.validate()?;
    let source_uv = mesh
        .mesh
        .uvs
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("paint mesh has no UVs"))?;
    let mut positions = Vec::with_capacity(mesh.mesh.vertices.len());
    let mut uv = Vec::with_capacity(source_uv.len());
    // MeshRender.get_mesh(normalize=True): invert axes, keep normalized scale.
    for (index, p) in mesh.mesh.vertices.iter().enumerate() {
        if index.is_multiple_of(4096) {
            checkpoint()?;
        }
        positions.push([-p[0], -p[2], p[1]]);
    }
    for (index, p) in source_uv.iter().enumerate() {
        if index.is_multiple_of(4096) {
            checkpoint()?;
        }
        uv.push([p[0], 1. - p[1]]);
    }
    Ok((positions, uv))
}

pub fn finish_materials(
    mesh: &PreparedMesh,
    baked: BakedMaterials,
    checkpoint: &mut dyn FnMut() -> Result<()>,
) -> Result<MaterialTextures> {
    ensure!(
        baked.albedo.size == baked.metallic_roughness.size,
        "material texture sizes differ"
    );
    let (positions, uv) = fill_geometry(mesh, checkpoint)?;
    let finish =
        |baked: BakedTexture, checkpoint: &mut dyn FnMut() -> Result<()>| -> Result<RgbImage> {
            let mut mask = Vec::with_capacity(baked.trusted.len());
            for (index, known) in baked.trusted.iter().enumerate() {
                if index.is_multiple_of(4096) {
                    checkpoint()?;
                }
                mask.push(if *known { 255 } else { 0 });
            }
            fill_texture(
                &VertexFillInput {
                    width: baked.size as usize,
                    height: baked.size as usize,
                    texture: &baked.colors,
                    mask: &mask,
                    positions: &positions,
                    uv: &uv,
                    faces: &mesh.mesh.faces,
                    uv_faces: &mesh.mesh.faces,
                },
                checkpoint,
            )
        };
    let albedo = finish(baked.albedo, checkpoint)?;
    let metallic_roughness = finish(baked.metallic_roughness, checkpoint)?;
    checkpoint()?;
    Ok(MaterialTextures {
        albedo,
        metallic_roughness,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{mesh::Mesh, paint_raster::PreparedMesh};
    use super::*;

    #[test]
    fn material_bake_refuses_invalid_views_before_rasterization() {
        let mesh = PreparedMesh {
            mesh: Mesh::default(),
            center: [0.; 3],
            scale: 1.,
        };
        let materials = PaintMaterials {
            albedo: vec![],
            metallic_roughness: vec![],
        };
        let views = super::super::paint_views::candidate_views();
        let error = bake_materials(&mesh, &materials, &views[..6], 1024, &mut || Ok(()))
            .err()
            .unwrap();
        assert!(error.to_string().contains("view counts"));
        let error = bake_materials(&mesh, &materials, &views[..6], 1024, &mut || {
            anyhow::bail!("cancelled")
        })
        .err()
        .unwrap();
        assert_eq!(error.to_string(), "cancelled");
    }

    #[test]
    fn material_finish_cancels_between_streams_and_rejects_mismatched_sizes() {
        let mesh = PreparedMesh {
            mesh: Mesh {
                uvs: Some(vec![]),
                ..Mesh::default()
            },
            center: [0.; 3],
            scale: 1.,
        };
        let make = || BakedMaterials {
            albedo: BakedTexture {
                size: 2,
                colors: vec![[0.5; 3]; 4],
                trusted: vec![true; 4],
            },
            metallic_roughness: BakedTexture {
                size: 2,
                colors: vec![[0.25; 3]; 4],
                trusted: vec![true; 4],
            },
        };
        let mut calls = 0;
        let result = finish_materials(&mesh, make(), &mut || {
            calls += 1;
            Ok(())
        })
        .unwrap();
        assert!(result.albedo.pixels().all(|p| p.0 == [127; 3]));
        assert!(result.metallic_roughness.pixels().all(|p| p.0 == [63; 3]));
        for stop in 1..=calls {
            let mut seen = 0;
            let error = finish_materials(&mesh, make(), &mut || {
                seen += 1;
                anyhow::ensure!(seen != stop, "cancelled");
                Ok(())
            })
            .err()
            .unwrap();
            assert_eq!(error.to_string(), "cancelled");
        }
        let mut wrong = make();
        wrong.metallic_roughness.size = 3;
        assert!(finish_materials(&mesh, wrong, &mut || Ok(()))
            .err()
            .unwrap()
            .to_string()
            .contains("sizes differ"));
    }

    #[test]
    #[ignore = "requires full material bake oracle and new output directory"]
    fn full_material_bake_matches_tencent() -> Result<()> {
        use candle_core::{Device, Tensor};
        use std::{collections::HashMap, path::PathBuf};
        let oracle = PathBuf::from(std::env::var("MOLD_MATERIAL_BAKE_ORACLE")?);
        let output = PathBuf::from(std::env::var("MOLD_MATERIAL_BAKE_OUTPUT")?);
        std::fs::create_dir(&output)?;
        let info: serde_json::Value =
            serde_json::from_slice(&std::fs::read(oracle.join("completed.json"))?)?;
        ensure!(
            info.get("diagnostic_bake_input")
                .is_none_or(serde_json::Value::is_null),
            "diagnostic bake input cannot qualify full material parity"
        );
        let size = info["size"].as_u64().unwrap() as u32;
        let tensors =
            candle_core::safetensors::load(oracle.join("mesh.safetensors"), &Device::Cpu)?;
        let mesh = Mesh {
            vertices: tensors["vertices"]
                .to_vec2::<f32>()?
                .into_iter()
                .map(|v| [v[0], v[1], v[2]])
                .collect(),
            faces: tensors["faces"]
                .to_vec2::<i32>()?
                .into_iter()
                .map(|v| [v[0] as u32, v[1] as u32, v[2] as u32])
                .collect(),
            uvs: Some(
                tensors["uv"]
                    .to_vec2::<f32>()?
                    .into_iter()
                    .map(|v| [v[0], v[1]])
                    .collect(),
            ),
            normals: None,
            vertex_colors: None,
        };
        let mesh = super::super::paint_raster::prepare_mesh(&mesh)?;
        let load_images = |stream: usize| -> Result<Vec<RgbImage>> {
            info["records"][stream]["images"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| Ok(image::open(item["path"].as_str().unwrap())?.to_rgb8()))
                .collect()
        };
        let materials = PaintMaterials {
            albedo: load_images(0)?,
            metallic_roughness: load_images(1)?,
        };
        let views = super::super::paint_views::candidate_views();
        let start = std::time::Instant::now();
        let baked = bake_materials(&mesh, &materials, &views[..6], size, &mut || Ok(()))?;
        let mut reports = Vec::new();
        let mut failed = false;
        for (name, actual) in [("albedo", &baked.albedo), ("mr", &baked.metallic_roughness)] {
            let colors = Tensor::from_vec(
                actual.colors.iter().flatten().copied().collect::<Vec<_>>(),
                (size as usize, size as usize, 3),
                &Device::Cpu,
            )?;
            let trust = Tensor::from_vec(
                actual
                    .trusted
                    .iter()
                    .map(|v| u8::from(*v))
                    .collect::<Vec<_>>(),
                (size as usize, size as usize, 1),
                &Device::Cpu,
            )?;
            candle_core::safetensors::save(
                &HashMap::from([("colors".to_string(), colors), ("trust".to_string(), trust)]),
                output.join(format!("{name}-bake.safetensors")),
            )?;
            let expected = candle_core::safetensors::load(
                oracle.join(format!("{name}-bake.safetensors")),
                &Device::Cpu,
            )?;
            ensure!(
                expected["colors"].dims() == [size as usize, size as usize, 3]
                    && expected["trust"].elem_count() == actual.trusted.len(),
                "oracle bake shape differs"
            );
            let reference = expected["colors"].flatten_all()?.to_vec1::<f32>()?;
            let max = actual
                .colors
                .iter()
                .flatten()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let trusted = expected["trust"].flatten_all()?.to_vec1::<u8>()?;
            let mask_diff = actual
                .trusted
                .iter()
                .zip(&trusted)
                .filter(|(a, b)| u8::from(**a) != **b)
                .count();
            let finite = actual.colors.iter().flatten().all(|v| v.is_finite())
                && reference.iter().all(|v| v.is_finite());
            failed |= !finite || max > 2e-6 || mask_diff != 0;
            reports.push(serde_json::json!({"stream":name,"max_float":max,"different_trust":mask_diff,"finite":finite}));
        }
        let textures = finish_materials(&mesh, baked, &mut || Ok(()))?;
        for (index, (name, actual)) in [
            ("albedo", textures.albedo),
            ("mr", textures.metallic_roughness),
        ]
        .into_iter()
        .enumerate()
        {
            actual.save(output.join(format!("{name}-filled.png")))?;
            let expected = image::open(oracle.join(format!("{name}-filled.png")))?.to_rgb8();
            ensure!(
                actual.dimensions() == expected.dimensions(),
                "oracle fill shape differs"
            );
            let max = actual
                .as_raw()
                .iter()
                .zip(expected.as_raw())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            reports[index]["max_byte"] = serde_json::json!(max);
            failed |= max != 0;
        }
        std::fs::write(
            output.join("comparison.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"seconds":start.elapsed().as_secs_f64(),"results":reports}),
            )?,
        )?;
        ensure!(!failed, "material bake or fill differs from oracle");
        Ok(())
    }

    #[test]
    fn fill_mesh_uses_normalized_inverse_axes_and_restored_uvs() {
        let mesh = PreparedMesh {
            mesh: Mesh {
                vertices: vec![[0.25, 0.5, -0.125]],
                faces: vec![],
                normals: None,
                vertex_colors: None,
                uvs: Some(vec![[0.25, 0.75]]),
            },
            center: [10.; 3],
            scale: 9.,
        };
        let (positions, uv) = fill_geometry(&mesh, &mut || Ok(())).unwrap();
        assert_eq!(positions, vec![[-0.25, 0.125, 0.5]]);
        assert_eq!(uv, vec![[0.25, 0.25]]);
    }
}
