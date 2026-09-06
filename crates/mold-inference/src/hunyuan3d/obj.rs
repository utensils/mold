//! Geometry-only Wavefront OBJ ingestion. Material libraries are never opened.
use super::mesh::Mesh;
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;

const MAX_ELEMENTS: usize = 8_000_000;
const MAX_CORNERS: usize = MAX_ELEMENTS * 3;

/// Read static polygon geometry, preserving independent corner attributes.
/// Positive OBJ indices are one-based; negative indices address the arrays
/// available at that face. Unlike a triangle fan, ear clipping also handles
/// concave polygons. Freeform curves and incomplete attributes are refused.
pub fn read_obj(text: &str) -> Result<Mesh> {
    ensure!(
        text.len() <= 256 * 1024 * 1024,
        "OBJ exceeds the 256 MiB input budget"
    );
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut colors = Vec::new();
    let mut mesh = Mesh::default();
    let mut corners = HashMap::new();
    let mut attribute_mode = None;
    for (line_index, line) in text.lines().enumerate() {
        ensure!(
            line.len() <= 65536,
            "OBJ line {} exceeds the line budget",
            line_index + 1
        );
        let line = line.split('#').next().unwrap_or("");
        let mut tokens = line.split_whitespace();
        let Some(kind) = tokens.next() else { continue };
        let values: Vec<_> = tokens.collect();
        let result = (|| -> Result<()> {
            match kind {
                "v" => {
                    ensure!(positions.len() < MAX_ELEMENTS, "too many positions");
                    ensure!(
                        matches!(values.len(), 3 | 4 | 6),
                        "expected XYZ, XYZW or XYZRGB position"
                    );
                    let mut point = vector::<3>(&values[..3])?;
                    if values.len() == 4 {
                        let w = number(values[3])?;
                        ensure!(w != 0., "zero homogeneous position weight");
                        point = point.map(|v| v / w);
                        ensure!(point.iter().all(|v| v.is_finite()), "nonfinite position");
                    }
                    positions.push(point);
                    colors.push(if values.len() == 6 {
                        Some(vector::<3>(&values[3..])?)
                    } else {
                        None
                    });
                }
                "vn" => {
                    ensure!(normals.len() < MAX_ELEMENTS, "too many normals");
                    normals.push(vector::<3>(&values)?);
                }
                "vt" => {
                    ensure!(uvs.len() < MAX_ELEMENTS, "too many UVs");
                    ensure!(
                        (1..=3).contains(&values.len()),
                        "expected one to three texture coordinates"
                    );
                    let u = number(values[0])?;
                    let v = values.get(1).map(|v| number(v)).transpose()?.unwrap_or(0.);
                    if let Some(w) = values.get(2) {
                        ensure!(number(w)? == 0., "3D texture coordinates are unsupported");
                    }
                    uvs.push([u, v]);
                }
                "f" => {
                    ensure!(
                        (3..=256).contains(&values.len()),
                        "faces require 3 to 256 corners"
                    );
                    ensure!(
                        mesh.faces.len() + values.len() - 2 <= MAX_ELEMENTS,
                        "too many triangles"
                    );
                    let mut face = Vec::with_capacity(values.len());
                    for value in values {
                        let fields: Vec<_> = value.split('/').collect();
                        ensure!((1..=3).contains(&fields.len()), "invalid face corner");
                        let p = resolve(fields[0], positions.len())?;
                        let uv = fields
                            .get(1)
                            .filter(|s| !s.is_empty())
                            .map(|s| resolve(s, uvs.len()))
                            .transpose()?;
                        let normal = fields
                            .get(2)
                            .filter(|s| !s.is_empty())
                            .map(|s| resolve(s, normals.len()))
                            .transpose()?;
                        let mode = (uv.is_some(), normal.is_some(), colors[p].is_some());
                        ensure!(
                            attribute_mode.is_none_or(|expected| expected == mode),
                            "incomplete per-corner attributes"
                        );
                        attribute_mode = Some(mode);
                        let key = (p, uv, normal);
                        let next = corners.len();
                        ensure!(next < MAX_CORNERS, "too many distinct corners");
                        let index = *corners.entry(key).or_insert_with(|| {
                            mesh.vertices.push(positions[p]);
                            if let Some(i) = uv {
                                mesh.uvs.get_or_insert_with(Vec::new).push(uvs[i]);
                            }
                            if let Some(i) = normal {
                                mesh.normals.get_or_insert_with(Vec::new).push(normals[i]);
                            }
                            if let Some(color) = colors[p] {
                                mesh.vertex_colors.get_or_insert_with(Vec::new).push(color);
                            }
                            next as u32
                        });
                        face.push(index);
                    }
                    mesh.faces.extend(triangulate(&face, &mesh.vertices)?);
                }
                "o" | "g" | "s" | "usemtl" | "mtllib" => {}
                other => bail!("unsupported OBJ record {other}"),
            }
            Ok(())
        })();
        result.with_context(|| format!("OBJ line {}", line_index + 1))?;
    }
    ensure!(!mesh.is_empty(), "OBJ has no triangle geometry");
    mesh.validate()?;
    Ok(mesh)
}

fn number(value: &str) -> Result<f32> {
    let value: f32 = value.parse().context("invalid numeric component")?;
    ensure!(value.is_finite(), "nonfinite numeric component");
    Ok(value)
}
fn vector<const N: usize>(values: &[&str]) -> Result<[f32; N]> {
    ensure!(values.len() == N, "expected {N} components");
    let mut result = [0.; N];
    for (slot, value) in result.iter_mut().zip(values) {
        *slot = number(value)?;
    }
    Ok(result)
}
fn resolve(value: &str, count: usize) -> Result<usize> {
    let index: i64 = value.parse().context("invalid OBJ index")?;
    let index = if index > 0 {
        index - 1
    } else if index < 0 {
        (count as i64)
            .checked_add(index)
            .context("index overflow")?
    } else {
        bail!("OBJ indices cannot be zero")
    };
    ensure!(
        index >= 0 && index < count as i64,
        "OBJ index is out of range"
    );
    Ok(index as usize)
}

fn triangulate(face: &[u32], vertices: &[[f32; 3]]) -> Result<Vec<[u32; 3]>> {
    let points: Vec<_> = face
        .iter()
        .map(|&i| vertices[i as usize].map(f64::from))
        .collect();
    let mut normal = [0.; 3];
    for (a, b) in points.iter().zip(points.iter().cycle().skip(1)) {
        normal[0] += (a[1] - b[1]) * (a[2] + b[2]);
        normal[1] += (a[2] - b[2]) * (a[0] + b[0]);
        normal[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let length = normal.iter().map(|x| x * x).sum::<f64>().sqrt();
    ensure!(length > 0., "degenerate polygon");
    let origin = points[0];
    let extent = points
        .iter()
        .flat_map(|p| (0..3).map(move |i| (p[i] - origin[i]).abs()))
        .fold(0., f64::max);
    for p in &points {
        let distance = (0..3)
            .map(|i| (p[i] - points[0][i]) * normal[i])
            .sum::<f64>()
            .abs()
            / length;
        ensure!(distance <= extent * 1e-5, "nonplanar polygon");
    }
    let dropped = (0..3)
        .max_by(|&a, &b| normal[a].abs().total_cmp(&normal[b].abs()))
        .unwrap();
    let axes: Vec<_> = (0..3).filter(|&i| i != dropped).collect();
    let p: Vec<_> = points.iter().map(|p| [p[axes[0]], p[axes[1]]]).collect();
    let cross = |a: usize, b: usize, c: usize| {
        (p[b][0] - p[a][0]) * (p[c][1] - p[a][1]) - (p[b][1] - p[a][1]) * (p[c][0] - p[a][0])
    };
    let epsilon = extent * extent * 1e-12;
    for a in 0..p.len() {
        let b = (a + 1) % p.len();
        ensure!(p[a] != p[b], "repeated polygon corner");
        for c in a + 1..p.len() {
            let d = (c + 1) % p.len();
            if a == d || b == c {
                continue;
            }
            // Nonadjacent edges may neither cross nor touch.
            let intersects = cross(a, b, c) * cross(a, b, d) <= 0.
                && cross(c, d, a) * cross(c, d, b) <= 0.
                && (0..2).all(|i| {
                    p[a][i].min(p[b][i]) <= p[c][i].max(p[d][i])
                        && p[c][i].min(p[d][i]) <= p[a][i].max(p[b][i])
                });
            ensure!(!intersects, "self-intersecting polygon");
        }
    }
    let signed_area: f64 = (0..p.len())
        .map(|i| {
            let j = (i + 1) % p.len();
            p[i][0] * p[j][1] - p[j][0] * p[i][1]
        })
        .sum();
    let sign = signed_area.signum();
    let mut remaining: Vec<_> = (0..p.len()).collect();
    let mut triangles = Vec::with_capacity(p.len() - 2);
    while remaining.len() > 3 {
        let mut ear = None;
        for i in 0..remaining.len() {
            let a = remaining[(i + remaining.len() - 1) % remaining.len()];
            let b = remaining[i];
            let c = remaining[(i + 1) % remaining.len()];
            if cross(a, b, c) * sign <= epsilon {
                continue;
            }
            let contains = remaining.iter().any(|&q| {
                q != a
                    && q != b
                    && q != c
                    && cross(a, b, q) * sign >= -epsilon
                    && cross(b, c, q) * sign >= -epsilon
                    && cross(c, a, q) * sign >= -epsilon
            });
            if !contains {
                ear = Some((i, [face[a], face[b], face[c]]));
                break;
            }
        }
        let (i, triangle) = ear.context("polygon cannot be triangulated")?;
        triangles.push(triangle);
        remaining.remove(i);
    }
    ensure!(
        cross(remaining[0], remaining[1], remaining[2]) * sign > epsilon,
        "degenerate final triangle"
    );
    triangles.push([face[remaining[0]], face[remaining[1]], face[remaining[2]]]);
    Ok(triangles)
}

#[cfg(test)]
mod tests {
    use super::*;
    const QUAD: &str = "v 0 0 0\nv 2 0 0\nv 2 2 0\nv 0 2 0\n";

    #[test]
    fn negative_indices_resolve_at_the_face_not_at_end_of_file() {
        let mesh = read_obj(&format!("{QUAD}f -4 -3 -2 -1\nv 999 999 999\n")).unwrap();
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.faces.len(), 2);
        assert_eq!(mesh.bounds(), ([0., 0., 0.], [2., 2., 0.]));
    }

    #[test]
    fn seams_keep_independent_position_uv_and_normal_indices() {
        let mesh = read_obj(&format!("{QUAD}vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1\nf 1/4/1 3/3/1 4/1/1\n")).unwrap();
        assert_eq!(mesh.vertices.len(), 5);
        assert_eq!(mesh.uvs.as_ref().unwrap().len(), 5);
        assert_eq!(mesh.normals.as_ref().unwrap(), &vec![[0., 0., 1.]; 5]);
    }

    #[test]
    fn concave_polygon_is_triangulated_without_covering_the_notch() {
        let mesh = read_obj("v 0 0 0\nv 3 0 0\nv 3 3 0\nv 2 1 0\nv 0 3 0\nf 1 2 3 4 5\n").unwrap();
        assert_eq!(mesh.faces.len(), 3);
        let area: f32 = mesh
            .faces
            .iter()
            .map(|face| {
                let [a, b, c] = face.map(|i| mesh.vertices[i as usize]);
                ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * 0.5
            })
            .sum();
        assert!((area - 6.).abs() < 1e-6);
    }

    #[test]
    fn material_names_are_metadata_not_filesystem_requests() {
        let mesh = read_obj(&format!(
            "mtllib ../../secret.mtl\n{QUAD}usemtl https://example.invalid/a\nf 1 2 3\n"
        ))
        .unwrap();
        assert_eq!(mesh.faces.len(), 1);
    }

    #[test]
    fn truncated_and_mutated_inputs_never_panic() {
        let source = format!("{QUAD}vt 0 0\nvn 0 0 1\nf 1/1/1 2/1/1 3/1/1 4/1/1\n");
        for end in 0..=source.len() {
            let _ = read_obj(&source[..end]);
        }
        for value in [
            "9223372036854775807",
            "-9223372036854775808",
            "1e999",
            "inf",
            "NaN",
            "-0",
            "1//2/3",
        ] {
            assert!(read_obj(&format!("{QUAD}f {value} 2 3\n")).is_err());
        }
        assert!(read_obj(&format!("{QUAD}vt 0 0\nf 1/1 2 3\n")).is_err());
    }

    #[test]
    fn clockwise_faces_keep_their_winding() {
        let mesh = read_obj(&format!("{QUAD}f 4 3 2 1\n")).unwrap();
        for face in mesh.faces {
            let [a, b, c] = face.map(|i| mesh.vertices[i as usize]);
            assert!((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0.);
        }
    }

    #[test]
    fn malformed_or_unsupported_geometry_is_refused() {
        for tail in [
            "f 0 2 3",
            "f -5 2 3",
            "f 1 2 8",
            "f 1 1 1",
            "f 1 2",
            "f 1 3 2 4",
            "curv 0 1 1 2",
            "f 1/7 2/7 3/7",
        ] {
            assert!(read_obj(&format!("{QUAD}{tail}\n")).is_err(), "{tail}");
        }
        assert!(read_obj("v NaN 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3").is_err());
    }
}
