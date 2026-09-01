//! Gallery poster for an extracted mesh.
//!
//! mold's gallery grid shows one image per print, and no browser tile can
//! render a `.glb`. Every mesh print therefore ships a poster PNG written at
//! save time; without it the tile falls back to `MESH_PLACEHOLDER_SVG`
//! (`crates/mold-server/src/thumbnails.rs`), a generic wireframe cube that is
//! identical for every mesh ever generated.
//!
//! The poster is deliberately styled to sit beside that placeholder rather than
//! replace its look: same slate background ramp (`#1e293b` -> `#0f172a`), same
//! near-white geometry (`#e2e8f0`). A grid mixing the two should read as one
//! set, so a mesh whose poster failed does not look like a different feature.
//!
//! A turntable is this poster set spinning: [`turntable_cameras`] sweeps the
//! azimuth from the poster view and [`render_frame_rgb`] gives one frame per
//! camera; `super::turntable` stacks them into a GIF, APNG or WebP.
//!
//! Pure CPU on top of [`super::raster`]; see that module for the camera and
//! G-buffer contract.

use anyhow::{bail, Context};
use image::{ImageFormat, RgbImage};

use crate::hunyuan3d::mesh::Mesh;
use crate::hunyuan3d::raster::{render_gbuffers, Camera, GBuffers};

/// A three-quarter view: a straight-on render of a symmetric object shows one
/// flat face and reads as a rectangle, which is exactly the failure the poster
/// exists to avoid.
pub const POSTER_AZIMUTH_DEG: f32 = 30.0;
pub const POSTER_ELEVATION_DEG: f32 = 20.0;

/// Supersampling factor. The rasterizer's coverage is a hard in/out test, so
/// silhouette edges alias badly at 1x; a 2x render box-filtered down is the
/// cheapest fix that does not need coverage-aware blending in the inner loop.
const SUPERSAMPLE: u32 = 2;

/// Upper bound on the requested edge. The G-buffers cost ~29 bytes per
/// supersampled pixel (depth + normal + position + mask), so 2048 already
/// reserves ~490 MB and anything larger is a memory incident rather than a
/// thumbnail.
pub const MAX_POSTER_SIZE: u32 = 2048;

/// sRGB background ramp, top to bottom. Same stops as `MESH_PLACEHOLDER_SVG`.
const BG_TOP: [u8; 3] = [0x1e, 0x29, 0x3b];
const BG_BOTTOM: [u8; 3] = [0x0f, 0x17, 0x2a];

/// Surface colour, sRGB. `#e2e8f0`, the placeholder's stroke colour.
const ALBEDO_SRGB: [f32; 3] = [0.886, 0.910, 0.941];

/// The camera the poster renders from. Public so a caller rendering its own
/// variant (a turntable, say) starts from the framing the gallery uses.
pub fn poster_camera() -> Camera {
    Camera::orthographic(POSTER_AZIMUTH_DEG, POSTER_ELEVATION_DEG).with_margin(0.08)
}

/// Render `mesh` to a `size x size` PNG.
///
/// Fails on an empty or unrenderable mesh instead of returning a blank tile:
/// the caller's fallback is the placeholder SVG, and a flat slate square would
/// be indistinguishable from a successful render of nothing.
pub fn render_poster(mesh: &Mesh, size: u32) -> anyhow::Result<Vec<u8>> {
    render_poster_from(mesh, &poster_camera(), size)
}

/// [`render_poster`] from an arbitrary view.
pub fn render_poster_from(mesh: &Mesh, camera: &Camera, size: u32) -> anyhow::Result<Vec<u8>> {
    let img = render_frame_rgb(mesh, camera, size)?;
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), ImageFormat::Png)
        .context("encode mesh poster as PNG")?;
    Ok(png)
}

/// The poster's pixels before any container: a `size x size` RGB frame
/// shaded exactly as [`render_poster_from`] would encode it.
///
/// This is the unit a turntable stacks — one call per camera from
/// [`turntable_cameras`], handed to the animation encoders in
/// `ltx_video::video_enc` — so the first frame of a looping GIF is
/// pixel-identical to the gallery poster.
pub fn render_frame_rgb(mesh: &Mesh, camera: &Camera, size: u32) -> anyhow::Result<RgbImage> {
    if mesh.is_empty() {
        bail!("cannot render a poster: the mesh has no geometry");
    }
    if size == 0 || size > MAX_POSTER_SIZE {
        bail!("poster size {size} is outside 1..={MAX_POSTER_SIZE}");
    }

    let ss = size * SUPERSAMPLE;
    let gb = render_gbuffers(mesh, camera, ss, ss);
    if gb.covered_pixels() == 0 {
        bail!("cannot render a poster: the mesh projects to nothing from this view");
    }

    let shaded = shade(&gb, camera);
    Ok(downsample(&shaded, ss, size))
}

/// The cameras of a `frames`-long turntable, starting at [`poster_camera`].
///
/// Every frame keeps the poster's elevation, margin and projection and only
/// the azimuth moves, so frame 0 IS the gallery poster and the animation
/// reads as that poster set spinning.
///
/// Two sweeps, chosen to match how the animation encoders play them back:
///
/// * **Loop** (`bounce = false`): one full turn in steps of `360 / frames`,
///   so the last frame stops one step SHORT of the first. A player that
///   wraps from the last frame to the first then takes a step like any
///   other; rendering the full 360° as its own frame would hold the poster
///   twice at every loop point.
/// * **Bounce** (`bounce = true`): a half turn, first frame to last
///   inclusive, in steps of `180 / (frames - 1)`. The GIF encoder's bounce
///   (`encode_gif_with_options`) appends the interior frames in reverse, so
///   the playback swings 0° -> 180° -> 0°: the far side is seen once on the
///   way out, and the reversal reads as a deliberate to-and-fro. A full
///   turn played forward then backward would show the object snap into
///   reverse at the very frame it had come round to the front again.
pub fn turntable_cameras(frames: usize, bounce: bool) -> Vec<Camera> {
    let start = poster_camera();
    if frames <= 1 {
        return vec![start; frames];
    }
    let step = if bounce {
        180.0 / (frames - 1) as f32
    } else {
        360.0 / frames as f32
    };
    (0..frames)
        .map(|index| Camera {
            azimuth_deg: start.azimuth_deg + step * index as f32,
            ..start
        })
        .collect()
}

/// Shade the G-buffers into a supersampled sRGB image.
fn shade(gb: &GBuffers, camera: &Camera) -> Vec<[u8; 3]> {
    let (right, up, to_eye) = camera.basis();
    // Lights are defined in the camera's frame, not the world's, so every view
    // is lit the same way: a key over the viewer's left shoulder and a dim fill
    // from the lower right that keeps the shadow side from going to background
    // black and losing its silhouette.
    let key = unit(combine(&[(right, -0.50), (up, 0.68), (to_eye, 0.55)]));
    let fill = unit(combine(&[(right, 0.72), (up, -0.30), (to_eye, 0.35)]));
    // Tuned so a fully key-lit normal lands just under the albedo rather than
    // clipping: a sphere shaded with a key strong enough to saturate has no
    // terminator left and reads as a flat white disc.
    const KEY: f32 = 0.85;
    const FILL: f32 = 0.18;
    const AMBIENT: f32 = 0.06;

    let ramp = surface_ramp();
    let ao_radius = (gb.height / 128).max(1) as i32;

    let mut out = Vec::with_capacity(gb.len());
    for y in 0..gb.height {
        let bg = background(y, gb.height);
        for x in 0..gb.width {
            let i = y as usize * gb.width as usize + x as usize;
            if !gb.mask[i] {
                out.push(bg);
                continue;
            }
            // Surface nets can emit an inward-facing triangle, and a two-sided
            // flip costs one dot product versus a black hole in the poster.
            let mut n = gb.normal[i];
            if dot(n, to_eye) < 0.0 {
                n = [-n[0], -n[1], -n[2]];
            }
            let lit = AMBIENT + KEY * dot(n, key).max(0.0) + FILL * dot(n, fill).max(0.0);
            let l = lit * occlusion(gb, x, y, ao_radius);
            out.push(ramp[ramp_index(l)]);
        }
    }
    out
}

/// Precomputed `albedo * intensity -> sRGB` ramp.
///
/// The sRGB transfer function needs a `powf` per channel, and at 1024x1024
/// supersampled that is three million of them — measurably more than the
/// rasterization it is shading. The albedo is a constant, so the whole
/// pipeline collapses to a one-dimensional function of the light intensity.
fn surface_ramp() -> Vec<[u8; 3]> {
    let albedo = [
        srgb_to_linear(ALBEDO_SRGB[0]),
        srgb_to_linear(ALBEDO_SRGB[1]),
        srgb_to_linear(ALBEDO_SRGB[2]),
    ];
    (0..RAMP_LEN)
        .map(|i| {
            let l = RAMP_MAX * i as f32 / (RAMP_LEN - 1) as f32;
            [
                to_u8(albedo[0] * l),
                to_u8(albedo[1] * l),
                to_u8(albedo[2] * l),
            ]
        })
        .collect()
}

/// 2048 buckets put the ramp's step well under one 8-bit code everywhere, so
/// quantizing the intensity is invisible in the output.
const RAMP_LEN: usize = 2048;
/// The brightest albedo channel is ~0.88 in linear light, so intensities past
/// this all clip to white and need no resolution.
const RAMP_MAX: f32 = 1.4;

fn ramp_index(l: f32) -> usize {
    let t = (l.clamp(0.0, RAMP_MAX) / RAMP_MAX) * (RAMP_LEN - 1) as f32;
    (t as usize).min(RAMP_LEN - 1)
}

/// Screen-space crease darkening: the fraction of a small neighbourhood that is
/// *nearer* than this pixel.
///
/// Eight taps, no accumulation buffer, no second pass — enough to make concave
/// seams read on a matte surface without turning the poster into a renderer.
fn occlusion(gb: &GBuffers, x: u32, y: u32, radius: i32) -> f32 {
    const TAPS: [(i32, i32); 8] = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];
    let i = y as usize * gb.width as usize + x as usize;
    let center = gb.depth[i];
    // Scaled to the mesh's own depth range so the term is resolution- and
    // scale-independent; a fixed world-space epsilon would vanish on a small
    // mesh and swallow a large one.
    let bias = center.abs() * 1e-3 + 1e-4;
    let mut occluded = 0.0f32;
    for (dx, dy) in TAPS {
        let sx = x as i32 + dx * radius;
        let sy = y as i32 + dy * radius;
        if sx < 0 || sy < 0 {
            continue;
        }
        let Some(j) = gb.index(sx as u32, sy as u32) else {
            continue;
        };
        if gb.mask[j] && gb.depth[j] < center - bias {
            occluded += 1.0;
        }
    }
    1.0 - 0.35 * (occluded / TAPS.len() as f32)
}

/// Vertical sRGB ramp between the placeholder's two stops.
fn background(y: u32, height: u32) -> [u8; 3] {
    let t = if height > 1 {
        y as f32 / (height - 1) as f32
    } else {
        0.0
    };
    let mut c = [0u8; 3];
    for k in 0..3 {
        let a = BG_TOP[k] as f32;
        let b = BG_BOTTOM[k] as f32;
        c[k] = (a + (b - a) * t).round().clamp(0.0, 255.0) as u8;
    }
    c
}

/// Box-filter `src` (`ss x ss`) down to `size x size`.
///
/// `ss` is always an exact multiple of `size` ([`SUPERSAMPLE`] is the ratio),
/// so this is a plain block average with no resampling weights to get wrong.
fn downsample(src: &[[u8; 3]], ss: u32, size: u32) -> RgbImage {
    let factor = (ss / size) as usize;
    let n = (factor * factor) as u32;
    RgbImage::from_fn(size, size, |x, y| {
        let mut acc = [0u32; 3];
        for sy in 0..factor {
            let row = (y as usize * factor + sy) * ss as usize;
            for sx in 0..factor {
                let p = src[row + x as usize * factor + sx];
                for k in 0..3 {
                    acc[k] += p[k] as u32;
                }
            }
        }
        image::Rgb([(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8])
    })
}

fn combine(terms: &[([f32; 3], f32)]) -> [f32; 3] {
    let mut v = [0.0f32; 3];
    for (dir, w) in terms {
        for k in 0..3 {
            v[k] += dir[k] * w;
        }
    }
    v
}

fn unit(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    if len > 1e-12 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Lighting is summed in linear light; the palette stops are sRGB. Skipping the
/// conversion makes every mid-tone too bright and flattens the shading, which
/// is the whole point of drawing a poster instead of a silhouette.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn to_u8(linear: f32) -> u8 {
    let v = linear_to_srgb(linear.clamp(0.0, 1.0));
    (v * 255.0).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hunyuan3d::mesh::compute_smooth_normals;

    fn cube(half: f32) -> Mesh {
        let v = [
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
        ];
        let quads = [
            [0, 1, 2, 3],
            [5, 4, 7, 6],
            [1, 5, 6, 2],
            [4, 0, 3, 7],
            [3, 2, 6, 7],
            [4, 5, 1, 0],
        ];
        let mut faces = Vec::new();
        for q in quads {
            faces.push([q[0] as u32, q[1] as u32, q[2] as u32]);
            faces.push([q[0] as u32, q[2] as u32, q[3] as u32]);
        }
        Mesh {
            vertices: v.to_vec(),
            faces,
            ..Default::default()
        }
    }

    fn decode(png: &[u8]) -> image::RgbImage {
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "not a PNG");
        image::load_from_memory_with_format(png, ImageFormat::Png)
            .expect("decode poster")
            .to_rgb8()
    }

    #[test]
    fn poster_is_a_square_png_with_real_shading() {
        let png = render_poster(&cube(0.5), 128).expect("render poster");
        let img = decode(&png);
        assert_eq!((img.width(), img.height()), (128, 128));

        let colors: std::collections::HashSet<[u8; 3]> =
            img.pixels().map(|p| [p[0], p[1], p[2]]).collect();
        assert!(
            colors.len() > 8,
            "poster is nearly flat: only {} distinct colours",
            colors.len()
        );

        // The brightest background stop is #1e293b; the lit surface must clear
        // it by a wide margin, otherwise "shading" is just the gradient.
        let brightest = img
            .pixels()
            .map(|p| p[0] as u32 + p[1] as u32 + p[2] as u32)
            .max();
        assert!(
            brightest.is_some_and(|b| b > 400),
            "nothing on the poster is lit: brightest sum {brightest:?}"
        );

        // Three visible faces at three angles to the key light means at least
        // three clearly separated surface tones.
        let mut tones: Vec<u32> = img
            .pixels()
            .map(|p| p[0] as u32 + p[1] as u32 + p[2] as u32)
            .filter(|s| *s > 200)
            .collect();
        tones.sort_unstable();
        tones.dedup();
        assert!(
            tones.len() >= 3,
            "expected distinct face tones, got {tones:?}"
        );
    }

    #[test]
    fn poster_uses_the_placeholder_background_palette() {
        let png = render_poster(&cube(0.05), 64).expect("render poster");
        let img = decode(&png);
        // A tiny mesh still auto-fits, so sample the extreme corners, which the
        // 8% margin guarantees are background.
        let top = img.get_pixel(0, 0);
        let bottom = img.get_pixel(0, 63);
        assert_eq!([top[0], top[1], top[2]], BG_TOP);
        assert_eq!([bottom[0], bottom[1], bottom[2]], BG_BOTTOM);
    }

    #[test]
    fn smooth_normals_are_honoured() {
        // Flat-shaded and smooth-shaded posters of the same cube must differ,
        // or the poster is silently ignoring `mesh.normals`.
        let flat = render_poster(&cube(0.5), 64).expect("flat poster");
        let mut smooth_mesh = cube(0.5);
        compute_smooth_normals(&mut smooth_mesh);
        let smooth = render_poster(&smooth_mesh, 64).expect("smooth poster");
        assert_ne!(flat, smooth);
    }

    #[test]
    fn empty_and_out_of_range_inputs_error() {
        assert!(render_poster(&Mesh::default(), 128).is_err());
        // Vertices but no faces is still nothing to draw.
        let vertices_only = Mesh {
            vertices: vec![[0.0; 3]; 3],
            ..Default::default()
        };
        assert!(render_poster(&vertices_only, 128).is_err());
        assert!(render_poster(&cube(0.5), 0).is_err());
        assert!(render_poster(&cube(0.5), MAX_POSTER_SIZE + 1).is_err());

        // A mesh with geometry that projects to nothing (all collinear) is an
        // error too, not a slate square.
        let collinear = Mesh {
            vertices: vec![[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            faces: vec![[0, 1, 2]],
            ..Default::default()
        };
        assert!(render_poster(&collinear, 64).is_err());
    }

    /// The RGB frame is the poster before PNG encoding: the same pixels a
    /// turntable stacks into an animation, so a GIF's first frame IS the
    /// gallery poster.
    #[test]
    fn render_frame_rgb_is_the_decoded_poster() {
        let mesh = cube(0.5);
        let camera = poster_camera();
        let frame = render_frame_rgb(&mesh, &camera, 96).expect("render frame");
        assert_eq!((frame.width(), frame.height()), (96, 96));
        let png = render_poster_from(&mesh, &camera, 96).expect("render poster");
        assert_eq!(decode(&png), frame);
        assert!(render_frame_rgb(&Mesh::default(), &camera, 96).is_err());
        assert!(render_frame_rgb(&mesh, &camera, MAX_POSTER_SIZE + 1).is_err());
    }

    /// A looping turntable is ONE full turn whose last frame stops one step
    /// short of the first, so the loop point is a step like any other rather
    /// than a held duplicate. Every frame keeps the poster's elevation and
    /// margin, and frame 0 is the poster itself.
    #[test]
    fn turntable_cameras_loop_is_a_seamless_full_turn() {
        let cameras = turntable_cameras(36, false);
        assert_eq!(cameras.len(), 36);
        let poster = poster_camera();
        assert_eq!(cameras[0], poster);
        for (index, camera) in cameras.iter().enumerate() {
            let expected = POSTER_AZIMUTH_DEG + 10.0 * index as f32;
            assert!(
                (camera.azimuth_deg - expected).abs() < 1e-3,
                "frame {index}: azimuth {} != {expected}",
                camera.azimuth_deg
            );
            assert_eq!(camera.elevation_deg, poster.elevation_deg);
            assert_eq!(camera.margin, poster.margin);
            assert_eq!(camera.projection, poster.projection);
        }
        let last = cameras.last().unwrap();
        assert!(
            (last.azimuth_deg - (POSTER_AZIMUTH_DEG + 350.0)).abs() < 1e-3,
            "the last frame must stop one step short of a full turn, got {}",
            last.azimuth_deg
        );
        assert_eq!(turntable_cameras(1, false), vec![poster]);
        assert!(turntable_cameras(0, false).is_empty());
    }

    /// A bouncing turntable sweeps a half turn, first frame to last
    /// inclusive. The encoder's bounce plays the interior frames back in
    /// reverse, so the reversal reads as a deliberate to-and-fro rather than
    /// a full turn snapping back on itself.
    #[test]
    fn turntable_cameras_bounce_is_a_half_turn_inclusive() {
        let cameras = turntable_cameras(9, true);
        assert_eq!(cameras.len(), 9);
        assert_eq!(cameras[0], poster_camera());
        for (index, camera) in cameras.iter().enumerate() {
            let expected = POSTER_AZIMUTH_DEG + 22.5 * index as f32;
            assert!(
                (camera.azimuth_deg - expected).abs() < 1e-3,
                "frame {index}: azimuth {} != {expected}",
                camera.azimuth_deg
            );
        }
        let last = cameras.last().unwrap();
        assert!(
            (last.azimuth_deg - (POSTER_AZIMUTH_DEG + 180.0)).abs() < 1e-3,
            "a bounce ends exactly half a turn from the poster, got {}",
            last.azimuth_deg
        );
        assert_eq!(turntable_cameras(1, true), vec![poster_camera()]);
    }

    /// Guards the save path: the poster runs inline when a mesh print is
    /// written, so a slow rasterizer stalls the response. Ignored by default
    /// because it is a timing measurement, not an invariant.
    #[test]
    #[ignore = "timing measurement, run with --ignored"]
    fn poster_render_time_for_a_large_mesh() {
        let mesh = big_mesh(200_000);
        assert!(mesh.face_count() >= 200_000, "{}", mesh.face_count());
        let start = std::time::Instant::now();
        let png = render_poster(&mesh, 512).expect("render poster");
        let elapsed = start.elapsed();
        println!(
            "poster: {} faces, {} verts, 512px (1024px supersampled) -> {} bytes in {:?}",
            mesh.face_count(),
            mesh.vertex_count(),
            png.len(),
            elapsed
        );
    }

    /// A closed UV sphere with at least `target` triangles, standing in for a
    /// surface-net extraction at a realistic density.
    #[cfg(test)]
    fn big_mesh(target: usize) -> Mesh {
        let rings = ((target as f32 / 2.0).sqrt().ceil() as usize).max(2);
        let segs = rings;
        let mut vertices = Vec::with_capacity((rings + 1) * (segs + 1));
        for r in 0..=rings {
            let theta = std::f32::consts::PI * r as f32 / rings as f32;
            for s in 0..=segs {
                let phi = std::f32::consts::TAU * s as f32 / segs as f32;
                vertices.push([
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                ]);
            }
        }
        let mut faces = Vec::with_capacity(rings * segs * 2);
        let stride = segs + 1;
        for r in 0..rings {
            for s in 0..segs {
                let a = (r * stride + s) as u32;
                let b = a + 1;
                let c = a + stride as u32;
                let d = c + 1;
                faces.push([a, c, b]);
                faces.push([b, c, d]);
            }
        }
        Mesh {
            vertices,
            faces,
            ..Default::default()
        }
    }
}
