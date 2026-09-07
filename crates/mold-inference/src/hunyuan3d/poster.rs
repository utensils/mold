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
use image::{ImageFormat, RgbImage, RgbaImage};

use crate::hunyuan3d::mesh::Mesh;
use crate::hunyuan3d::raster::{render_gbuffers, sweep_fit_for, Camera, GBuffers};

/// A three-quarter view: a straight-on render of a symmetric object shows one
/// flat face and reads as a rectangle, which is exactly the failure the poster
/// exists to avoid.
pub const POSTER_AZIMUTH_DEG: f32 = 30.0;
pub const POSTER_ELEVATION_DEG: f32 = 20.0;
/// Fraction of the frame left empty around the mesh's swept extent.
///
/// Wider than the rasterizer's own
/// [`DEFAULT_MARGIN`](crate::hunyuan3d::raster::DEFAULT_MARGIN) because the
/// poster's fit is the rotation-invariant sweep bound: the silhouette that
/// actually touches this margin is the one at the widest azimuth, not this
/// frame's.
pub const POSTER_MARGIN: f32 = 0.08;

/// Direction the turntable's azimuth steps, per frame.
///
/// The object spins the way a rightward drag turns it in `MeshViewer.vue` and
/// the way auto-rotate tours it; negative orbits the eye toward `-X`. The
/// viewer rotates the MODEL by `yaw` where the server orbits the EYE by
/// `azimuth` (`yaw = -azimuth`), so a positive step here would play the GIF
/// backwards relative to every interactive surface. Mirrored in
/// `studio/lib/meshViewerCamera.ts` and pinned by
/// `the_viewer_mirrors_the_poster_camera`.
pub const TURNTABLE_AZIMUTH_STEP_SIGN: f32 = -1.0;

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

/// How a texture coordinate outside the unit square is resolved.
///
/// Read from the material's sampler rather than fixed, because mold's own
/// paint and a foreign file disagree: `write_glb` writes `CLAMP_TO_EDGE`
/// and glTF's default for a file that names no sampler is `REPEAT`.
/// `MIRRORED_REPEAT` and any unknown value resolve as `Repeat` — glTF's
/// default, and what this renderer did for every file before it read the
/// sampler at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextureWrap {
    /// glTF's DEFAULT sampler, and what a file that names none asks for.
    #[default]
    Repeat,
    /// `CLAMP_TO_EDGE`, which is what [`write_glb`] stamps onto mold's own
    /// paint and what `MeshViewer.vue` binds the same texture with.
    ///
    /// [`write_glb`]: crate::hunyuan3d::glb::write_glb
    ClampToEdge,
}

impl TextureWrap {
    /// Resolve one texel index against this mode. A zero extent has no texel
    /// to land on; no decoder produces one, and answering 0 is cheaper than
    /// making every caller prove it.
    fn resolve(self, value: i64, extent: u32) -> u32 {
        if extent == 0 {
            return 0;
        }
        let last = i64::from(extent) - 1;
        match self {
            Self::ClampToEdge => value.clamp(0, last) as u32,
            Self::Repeat => {
                let extent = i64::from(extent);
                (((value % extent) + extent) % extent) as u32
            }
        }
    }
}

/// The surface colour a poster paints onto the geometry.
///
/// Bare geometry has none, and deliberately keeps the placeholder's near-white
/// [`ALBEDO_SRGB`] so a grid mixing rendered posters with
/// `MESH_PLACEHOLDER_SVG` fallbacks reads as one set. A PAINTED mesh is a
/// different object: its colours are the point of having painted it, and a
/// purple octopus shown as a grey one is not a thumbnail of that print.
///
/// [`crate::hunyuan3d::glb::read_glb_scene`] is where a stored `.glb` produces
/// one of these.
#[derive(Clone, Debug)]
pub struct Appearance {
    /// `baseColorTexture`, sRGB, sampled through `mesh.uvs`.
    pub base_color_texture: Option<RgbImage>,
    /// Linear `baseColorFactor` RGB, multiplying the texture and the mesh's
    /// vertex colours.
    ///
    /// Read only when there IS one of those to multiply — see
    /// [`crate::hunyuan3d::glb::GlbScene::base_color_factor`].
    pub base_color_factor: [f32; 3],
    /// The `baseColorTexture` sampler's `wrapS` and `wrapT`.
    ///
    /// Read from the file rather than assumed, because the two answers
    /// disagree at a chart border: mold's own paint declares
    /// `CLAMP_TO_EDGE`, so wrapping it blends the atlas's opposite edge into
    /// the poster and the turntable while the viewer — which binds
    /// `CLAMP_TO_EDGE` — shows neither. That is the one seam where "the
    /// thumbnail IS the viewer's home frame" could stop being true.
    pub wrap: [TextureWrap; 2],
}

impl Default for Appearance {
    /// No material: the placeholder surface, shaded exactly as it always was.
    fn default() -> Self {
        Self {
            base_color_texture: None,
            base_color_factor: [1.0; 3],
            wrap: [TextureWrap::Repeat; 2],
        }
    }
}

/// The camera the poster renders from, before it is framed to a mesh. Public
/// so a caller rendering its own variant (a turntable, say) starts from the
/// eye position the gallery uses.
///
/// It carries [`FrameFit::Auto`](crate::hunyuan3d::raster::FrameFit::Auto):
/// this is a statement about where the eye goes, not about how big the mesh
/// is drawn. [`poster_camera_for`] is the framed one.
pub fn poster_camera() -> Camera {
    Camera::orthographic(POSTER_AZIMUTH_DEG, POSTER_ELEVATION_DEG).with_margin(POSTER_MARGIN)
}

/// [`poster_camera`] framed to `mesh` by the shared sweep bound.
///
/// The ONE framing of this mesh: the same value frames every turntable frame
/// ([`crate::hunyuan3d::turntable::turntable_frame_cameras`]) and the
/// interactive viewer's home view (`studio/lib/meshViewerCamera.ts`), so the
/// gallery tile, a GIF's first frame, and the 3-D view a client opens are the
/// same picture at the same size. A mesh with nothing to frame keeps the
/// per-frame auto-fit, which draws nothing either way.
pub fn poster_camera_for(mesh: &Mesh) -> Camera {
    let camera = poster_camera();
    match sweep_fit_for(mesh, POSTER_ELEVATION_DEG) {
        Some(fit) => camera.with_fit(fit),
        None => camera,
    }
}

/// Render `mesh` to a `size x size` PNG.
///
/// Fails on an empty or unrenderable mesh instead of returning a blank tile:
/// the caller's fallback is the placeholder SVG, and a flat slate square would
/// be indistinguishable from a successful render of nothing.
pub fn render_poster(mesh: &Mesh, size: u32) -> anyhow::Result<Vec<u8>> {
    render_poster_with(mesh, &Appearance::default(), size)
}

/// [`render_poster`] of a mesh that carries its own surface colour.
///
/// The entry point every caller holding a stored `.glb` should use: the file's
/// material is what tells a painted mesh apart from a bare one, and rendering
/// it without one is what made every PBR print's gallery tile grey.
pub fn render_poster_with(
    mesh: &Mesh,
    appearance: &Appearance,
    size: u32,
) -> anyhow::Result<Vec<u8>> {
    let img = render_frame(mesh, appearance, &poster_camera_for(mesh), size, false)?;
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), ImageFormat::Png)
        .context("encode mesh poster as PNG")?;
    Ok(png)
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
/// `ltx_video::video_enc`. A turntable is framed ONCE for the whole sweep by
/// the same rotation-invariant bound the poster uses, so the mesh keeps one
/// size as it turns and frame 0 IS the poster, pixel for pixel.
pub fn render_frame_rgb(mesh: &Mesh, camera: &Camera, size: u32) -> anyhow::Result<RgbImage> {
    render_frame(mesh, &Appearance::default(), camera, size, false)
}

/// [`render_frame_rgb`] for one frame of a sequence: a view from which the
/// mesh projects to nothing is a background-only frame, not an error.
///
/// A single poster of nothing is a failure worth reporting (the caller's
/// fallback is the placeholder). A turntable of a flat mesh — a relief, a
/// plane, a cut-out — necessarily passes through edge-on views, and those
/// frames are CORRECT: the object really does vanish there. Failing the whole
/// export for them would refuse exactly the meshes a turntable helps with.
/// An empty mesh and an out-of-range size are still errors.
pub fn render_sequence_frame_rgb(
    mesh: &Mesh,
    camera: &Camera,
    size: u32,
) -> anyhow::Result<RgbImage> {
    render_sequence_frame_rgb_with(mesh, &Appearance::default(), camera, size)
}

/// [`render_sequence_frame_rgb`] of a mesh that carries its own surface
/// colour, so a turntable of a painted mesh spins the painted object.
pub fn render_sequence_frame_rgb_with(
    mesh: &Mesh,
    appearance: &Appearance,
    camera: &Camera,
    size: u32,
) -> anyhow::Result<RgbImage> {
    render_frame(mesh, appearance, camera, size, true)
}

fn render_frame(
    mesh: &Mesh,
    appearance: &Appearance,
    camera: &Camera,
    size: u32,
    allow_empty_view: bool,
) -> anyhow::Result<RgbImage> {
    if mesh.is_empty() {
        bail!("cannot render a poster: the mesh has no geometry");
    }
    if size == 0 || size > MAX_POSTER_SIZE {
        bail!("poster size {size} is outside 1..={MAX_POSTER_SIZE}");
    }

    let ss = size * SUPERSAMPLE;
    let gb = render_gbuffers(mesh, camera, ss, ss);
    if gb.covered_pixels() == 0 && !allow_empty_view {
        bail!("cannot render a poster: the mesh projects to nothing from this view");
    }

    let shaded = shade(&gb, camera, mesh, appearance, Backdrop::Ramp);
    Ok(downsample(&shaded, ss, size))
}

/// [`render_sequence_frame_rgb_with`] on nothing: the mesh over a fully
/// transparent backdrop, its silhouette carried as antialiased alpha.
///
/// A turntable of a 3-D object is exactly the thing people drop onto a slide,
/// a README or a page that is not slate blue, and a baked-in background is
/// what stops them.
pub fn render_sequence_frame_rgba_with(
    mesh: &Mesh,
    appearance: &Appearance,
    camera: &Camera,
    size: u32,
) -> anyhow::Result<RgbaImage> {
    if mesh.is_empty() {
        bail!("cannot render a poster: the mesh has no geometry");
    }
    if size == 0 || size > MAX_POSTER_SIZE {
        bail!("poster size {size} is outside 1..={MAX_POSTER_SIZE}");
    }
    let ss = size * SUPERSAMPLE;
    let gb = render_gbuffers(mesh, camera, ss, ss);
    let shaded = shade(&gb, camera, mesh, appearance, Backdrop::Transparent);
    Ok(downsample_rgba(&shaded, ss, size))
}

/// The cameras of a `frames`-long turntable, starting at [`poster_camera`].
///
/// Every frame keeps the poster's elevation, margin and projection and only
/// the azimuth moves, so the animation reads as that poster set spinning. A
/// turntable is framed ONCE for the whole sweep by the bound the poster
/// itself uses, so frame 0 IS the poster, pixel for pixel. The cameras
/// returned here still carry
/// [`crate::hunyuan3d::raster::FrameFit::Auto`] — stamping the shared fit is
/// [`crate::hunyuan3d::turntable::turntable_frame_cameras`]'s job, so this
/// function stays a pure statement about where the eye goes.
///
/// The azimuth steps by [`TURNTABLE_AZIMUTH_STEP_SIGN`], so the object turns
/// the way a rightward drag turns it in the interactive viewer.
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
///   the playback swings the poster azimuth out half a turn and back —
///   30° -> -150° -> 30° at the shipped [`POSTER_AZIMUTH_DEG`], the azimuth
///   DECREASING on the way out per
///   [`TURNTABLE_AZIMUTH_STEP_SIGN`]: the far side is seen once on the way
///   out, and the reversal reads as a deliberate to-and-fro. A full
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
            azimuth_deg: start.azimuth_deg + TURNTABLE_AZIMUTH_STEP_SIGN * step * index as f32,
            ..start
        })
        .collect()
}

/// What an uncovered pixel becomes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Backdrop {
    /// The slate ramp the placeholder SVG uses. Opaque.
    Ramp,
    /// Nothing at all, so the mesh can be composited onto whatever the frame
    /// is dropped into.
    Transparent,
}

/// Shade the G-buffers into a supersampled sRGB image.
///
/// Always RGBA so the two backdrops share one loop; an opaque render throws
/// the alpha away in [`downsample`] and is byte-for-byte what it was when the
/// shader worked in RGB.
fn shade(
    gb: &GBuffers,
    camera: &Camera,
    mesh: &Mesh,
    appearance: &Appearance,
    backdrop: Backdrop,
) -> Vec<[u8; 4]> {
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

    let surface = Surface::of(mesh, appearance);
    let ao_radius = (gb.height / 128).max(1) as i32;

    let mut out = Vec::with_capacity(gb.len());
    for y in 0..gb.height {
        let bg = match backdrop {
            Backdrop::Ramp => {
                let [r, g, b] = background(y, gb.height);
                [r, g, b, 255]
            }
            Backdrop::Transparent => [0, 0, 0, 0],
        };
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
            let [r, g, b] = surface.pixel(gb, i, l);
            out.push([r, g, b, 255]);
        }
    }
    out
}

/// Where a covered pixel's albedo comes from.
///
/// The split is the whole point: a mesh with no painted surface takes the
/// original constant-albedo ramp and its poster is byte-for-byte what it was,
/// while a painted one pays for a per-pixel lookup it actually needs.
enum Surface<'a> {
    /// One albedo for the whole mesh, collapsed into a 1-D intensity ramp.
    Placeholder(Vec<[u8; 3]>),
    /// Per-pixel albedo, interpolated out of the mesh through the G-buffer's
    /// face ids and barycentrics.
    Painted(Painted<'a>),
}

struct Painted<'a> {
    mesh: &'a Mesh,
    texture: Option<&'a RgbImage>,
    factor: [f32; 3],
    /// `u8 sRGB -> linear`, the texture decode. Heap-resident so the enum's
    /// painted variant does not make every bare render carry a kilobyte of
    /// table it will never read.
    decode: Vec<f32>,
    /// `linear -> u8 sRGB`, the encode the constant-albedo ramp does not need.
    encode: Vec<u8>,
    /// The material's own `wrapS`/`wrapT`, carried so [`Painted::sample`]
    /// resolves a border texel the way the file asked and the viewer does.
    wrap: [TextureWrap; 2],
}

impl<'a> Surface<'a> {
    fn of(mesh: &'a Mesh, appearance: &'a Appearance) -> Self {
        // A texture with no coordinates to sample it through is not a surface.
        let texture = appearance
            .base_color_texture
            .as_ref()
            .filter(|_| mesh.uvs.is_some());
        if texture.is_none() && mesh.vertex_colors.is_none() {
            return Self::Placeholder(surface_ramp());
        }
        Self::Painted(Painted {
            mesh,
            texture,
            factor: appearance.base_color_factor,
            decode: (0..256)
                .map(|code| srgb_to_linear(code as f32 / 255.0))
                .collect(),
            encode: srgb_encode_table(),
            wrap: appearance.wrap,
        })
    }

    fn pixel(&self, gb: &GBuffers, i: usize, l: f32) -> [u8; 3] {
        match self {
            Self::Placeholder(ramp) => ramp[ramp_index(l)],
            Self::Painted(painted) => {
                let albedo = painted.albedo(gb, i);
                [0, 1, 2].map(|axis| encode(&painted.encode, albedo[axis] * l))
            }
        }
    }
}

impl Painted<'_> {
    /// Linear albedo at one covered pixel.
    ///
    /// glTF 2.0 §3.9.2 makes the base colour the product of
    /// `baseColorFactor`, `baseColorTexture` and `COLOR_0` — the factor and
    /// the vertex colours already linear, and only the texture sRGB-encoded.
    fn albedo(&self, gb: &GBuffers, i: usize) -> [f32; 3] {
        let mut albedo = self.factor;
        let Some(face) = self.mesh.faces.get(gb.face_ids[i] as usize).copied() else {
            return albedo;
        };
        let weights = gb.barycentric[i];
        if let Some(colors) = self.mesh.vertex_colors.as_ref() {
            let mut blended = [0.0f32; 3];
            for (corner, vertex) in face.iter().enumerate() {
                let Some(color) = colors.get(*vertex as usize) else {
                    return albedo;
                };
                for axis in 0..3 {
                    blended[axis] += weights[corner] * color[axis];
                }
            }
            for axis in 0..3 {
                albedo[axis] *= blended[axis].clamp(0.0, 1.0);
            }
        }
        if let (Some(texture), Some(uvs)) = (self.texture, self.mesh.uvs.as_ref()) {
            let mut uv = [0.0f32; 2];
            for (corner, vertex) in face.iter().enumerate() {
                let Some(coord) = uvs.get(*vertex as usize) else {
                    return albedo;
                };
                uv[0] += weights[corner] * coord[0];
                uv[1] += weights[corner] * coord[1];
            }
            let texel = self.sample(texture, uv);
            for axis in 0..3 {
                albedo[axis] *= texel[axis];
            }
        }
        albedo
    }

    /// Bilinear texel, linear light.
    ///
    /// glTF's texture origin is the image's top-left corner. A coordinate
    /// outside the unit square is resolved by the material's OWN sampler —
    /// REPEAT only where the file asks for it (or names none, glTF's
    /// default); mold's paint asks for `CLAMP_TO_EDGE`. Bilinear rather than nearest because the poster is a
    /// small tile: the parts of a 2048-square texture that magnify into it
    /// would otherwise show their texels.
    fn sample(&self, texture: &RgbImage, uv: [f32; 2]) -> [f32; 3] {
        let (width, height) = (texture.width(), texture.height());
        let x = uv[0] * width as f32 - 0.5;
        let y = uv[1] * height as f32 - 0.5;
        // A foreign `.glb` can carry a UV of 1e30 or a NaN. Rust's
        // float-to-int casts saturate and send NaN to zero, so the index is
        // always real — but the FRACTIONS would go NaN and take the whole
        // albedo with them, and `x0 + 1` on a saturated `i64::MAX` panics in
        // a debug build. Fall back to the texture's origin instead.
        let (x0, y0, fx, fy) = if x.is_finite() && y.is_finite() {
            let (x0, y0) = (x.floor(), y.floor());
            (x0 as i64, y0 as i64, x - x0, y - y0)
        } else {
            (0, 0, 0.0, 0.0)
        };
        let mut out = [0.0f32; 3];
        for (dx, dy, weight) in [
            (0, 0, (1.0 - fx) * (1.0 - fy)),
            (1, 0, fx * (1.0 - fy)),
            (0, 1, (1.0 - fx) * fy),
            (1, 1, fx * fy),
        ] {
            let texel = texture
                .get_pixel(
                    self.wrap[0].resolve(x0.saturating_add(dx), width),
                    self.wrap[1].resolve(y0.saturating_add(dy), height),
                )
                .0;
            for axis in 0..3 {
                out[axis] += weight * self.decode[texel[axis] as usize];
            }
        }
        out
    }
}

/// `linear -> sRGB u8` as a table.
///
/// The constant-albedo ramp folds the transfer function into the intensity;
/// a per-pixel albedo cannot, and a `powf` per channel per supersampled pixel
/// is exactly the cost [`surface_ramp`] was written to avoid. The table's
/// steepest region is at black, where the curve's slope is 12.92, so
/// [`ENCODE_LEN`] buckets keep the quantization inside a quarter of an 8-bit
/// code everywhere (12.92 x 255 / 16383 = 0.20).
const ENCODE_LEN: usize = 16384;

fn srgb_encode_table() -> Vec<u8> {
    (0..ENCODE_LEN)
        .map(|i| to_u8(i as f32 / (ENCODE_LEN - 1) as f32))
        .collect()
}

fn encode(table: &[u8], linear: f32) -> u8 {
    let t = linear.clamp(0.0, 1.0) * (ENCODE_LEN - 1) as f32;
    table[(t as usize).min(ENCODE_LEN - 1)]
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
fn downsample(src: &[[u8; 4]], ss: u32, size: u32) -> RgbImage {
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

/// [`downsample`] keeping coverage as alpha.
///
/// Averages in PREMULTIPLIED space and un-premultiplies once. A silhouette
/// subpixel that missed the mesh has no colour of its own, and averaging its
/// zero straight into the neighbours' would darken every edge toward black —
/// the halo a naive alpha downsample leaves around a cut-out.
fn downsample_rgba(src: &[[u8; 4]], ss: u32, size: u32) -> RgbaImage {
    let factor = (ss / size) as usize;
    let n = (factor * factor) as u32;
    RgbaImage::from_fn(size, size, |x, y| {
        let mut acc = [0u32; 4];
        for sy in 0..factor {
            let row = (y as usize * factor + sy) * ss as usize;
            for sx in 0..factor {
                let p = src[row + x as usize * factor + sx];
                let a = u32::from(p[3]);
                for k in 0..3 {
                    acc[k] += u32::from(p[k]) * a;
                }
                acc[3] += a;
            }
        }
        if acc[3] == 0 {
            return image::Rgba([0, 0, 0, 0]);
        }
        let alpha = acc[3] / n;
        image::Rgba([
            (acc[0] / acc[3]) as u8,
            (acc[1] / acc[3]) as u8,
            (acc[2] / acc[3]) as u8,
            alpha as u8,
        ])
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

    /// Cube UVs that put every face somewhere inside the unit square, so a
    /// sampled texture is exercised rather than a single wrapped texel.
    fn uv_cube(half: f32) -> Mesh {
        let mut mesh = cube(half);
        mesh.uvs = Some(
            mesh.vertices
                .iter()
                .map(|v| [(v[0] / half + 1.0) * 0.5, (v[1] / half + 1.0) * 0.5])
                .collect(),
        );
        mesh
    }

    fn solid_texture(rgb: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(8, 8, image::Rgb(rgb))
    }

    /// A block of pixels at the centre of the frame.
    ///
    /// Every one of them is on the cube for the poster camera and margin, so
    /// two renders of the same mesh can be compared here without a coverage
    /// test that a darker surface colour would fall out of.
    fn centre_block(img: &RgbImage) -> Vec<[u8; 3]> {
        let (cx, cy) = (img.width() / 2, img.height() / 2);
        (cy - 4..=cy + 4)
            .flat_map(|y| (cx - 4..=cx + 4).map(move |x| (x, y)))
            .map(|(x, y)| img.get_pixel(x, y).0)
            .collect()
    }

    #[test]
    fn a_base_color_texture_paints_the_poster() {
        let mesh = uv_cube(0.5);
        let bare = decode(&render_poster(&mesh, 64).unwrap());
        let painted = decode(
            &render_poster_with(
                &mesh,
                &Appearance {
                    base_color_texture: Some(solid_texture([220, 20, 40])),
                    base_color_factor: [1.0; 3],
                    wrap: [TextureWrap::Repeat; 2],
                },
                64,
            )
            .unwrap(),
        );

        assert_ne!(
            bare, painted,
            "a painted mesh must not render as a bare one"
        );
        // The placeholder surface is near-neutral; the painted one is red.
        for pixel in centre_block(&bare) {
            assert!(
                pixel[0].abs_diff(pixel[2]) < 24,
                "bare geometry keeps the neutral placeholder surface, got {pixel:?}"
            );
        }
        for pixel in centre_block(&painted) {
            assert!(
                pixel[0] > pixel[1] && pixel[0] > pixel[2],
                "the texture's red must dominate every lit pixel, got {pixel:?}"
            );
        }
    }

    #[test]
    fn the_base_color_factor_scales_the_texture() {
        let mesh = uv_cube(0.5);
        let painted = |factor: [f32; 3]| {
            decode(
                &render_poster_with(
                    &mesh,
                    &Appearance {
                        base_color_texture: Some(solid_texture([255, 255, 255])),
                        base_color_factor: factor,
                        wrap: [TextureWrap::Repeat; 2],
                    },
                    64,
                )
                .unwrap(),
            )
        };
        let full = centre_block(&painted([1.0; 3]));
        let quarter = centre_block(&painted([0.25; 3]));
        assert!(full
            .iter()
            .zip(&quarter)
            .all(|(bright, dim)| bright[1] >= dim[1]));
        assert!(
            full.iter()
                .zip(&quarter)
                .all(|(bright, dim)| bright[1] > dim[1] + 20),
            "a quarter factor must visibly darken the surface"
        );
    }

    #[test]
    fn vertex_colours_paint_a_mesh_with_no_texture() {
        let mut mesh = cube(0.5);
        mesh.vertex_colors = Some(vec![[0.1, 0.9, 0.2]; mesh.vertices.len()]);
        let painted = decode(&render_poster(&mesh, 64).unwrap());
        for pixel in centre_block(&painted) {
            assert!(
                pixel[1] > pixel[0] && pixel[1] > pixel[2],
                "COLOR_0 green must reach the poster, got {pixel:?}"
            );
        }
    }

    #[test]
    fn an_unpainted_mesh_renders_exactly_as_it_always_did() {
        // The placeholder path is not merely equivalent, it is the same code:
        // a bare mesh must not shift by a single code because painted meshes
        // gained a shader. Cached tiles across the fleet depend on it.
        let mesh = cube(0.5);
        assert_eq!(
            render_poster(&mesh, 96).unwrap(),
            render_poster_with(&mesh, &Appearance::default(), 96).unwrap()
        );
        // A texture with no UVs to sample it through is not a surface either.
        assert_eq!(
            render_poster(&mesh, 96).unwrap(),
            render_poster_with(
                &mesh,
                &Appearance {
                    base_color_texture: Some(solid_texture([255, 0, 0])),
                    base_color_factor: [1.0; 3],
                    wrap: [TextureWrap::Repeat; 2],
                },
                96,
            )
            .unwrap()
        );
    }

    /// The two modes must actually disagree at a border, or honouring the
    /// sampler is a distinction without a difference.
    ///
    /// A two-column texture sampled at the left edge: REPEAT blends the last
    /// column in, CLAMP_TO_EDGE holds the first. mold's own paint declares
    /// CLAMP, so REPEAT here is the poster disagreeing with the viewer.
    #[test]
    fn the_sampler_decides_what_a_border_texel_blends_with() {
        let mut texture = RgbImage::new(2, 1);
        texture.put_pixel(0, 0, image::Rgb([0, 0, 255]));
        texture.put_pixel(1, 0, image::Rgb([255, 0, 0]));

        let sample = |wrap: TextureWrap| {
            let appearance = Appearance {
                base_color_texture: Some(texture.clone()),
                base_color_factor: [1.0; 3],
                wrap: [wrap; 2],
            };
            // A texture with no coordinates to sample it through is not a
            // painted surface, so the mesh must carry UVs.
            let mut mesh = cube(0.5);
            mesh.uvs = Some(vec![[0.5, 0.5]; mesh.vertices.len()]);
            let surface = Surface::of(&mesh, &appearance);
            match surface {
                // Left edge of the first texel: x = 0*2 - 0.5 = -0.5, so the
                // bilinear tap reaches index -1.
                Surface::Painted(p) => p.sample(&texture, [0.0, 0.5]),
                _ => panic!("expected a painted surface"),
            }
        };

        let clamped = sample(TextureWrap::ClampToEdge);
        let repeated = sample(TextureWrap::Repeat);
        assert!(
            clamped[2] > clamped[0],
            "clamped to the blue first column, got {clamped:?}"
        );
        assert!(
            repeated[0] > clamped[0],
            "repeat pulls the red last column in; clamp {clamped:?} vs repeat {repeated:?}"
        );
    }

    /// The reader and mold's own writer must agree, or every painted poster
    /// silently samples the wrong way.
    #[test]
    fn mold_s_own_glb_is_read_back_as_clamped() {
        let mut mesh = cube(0.5);
        mesh.uvs = Some(vec![[0.5, 0.5]; mesh.vertices.len()]);
        let mut png = std::io::Cursor::new(Vec::new());
        solid_texture([10, 200, 90])
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("encode the texture");
        let glb = crate::hunyuan3d::glb::write_glb(
            &mesh,
            &crate::hunyuan3d::glb::GlbMaterial {
                base_color_texture: Some(png.into_inner()),
                ..Default::default()
            },
            None,
        )
        .expect("write a painted glb");
        let scene = crate::hunyuan3d::glb::read_glb_scene(&glb).expect("read it back");
        assert_eq!(
            scene.texture_wrap,
            [TextureWrap::ClampToEdge; 2],
            "write_glb declares CLAMP_TO_EDGE; the reader must not assume REPEAT"
        );
    }

    #[test]
    fn wrapping_repeats_texels_instead_of_panicking() {
        // glTF's default sampler REPEATs, and a decimated or foreign mesh can
        // carry coordinates far outside the unit square.
        let mut mesh = cube(0.5);
        mesh.uvs = Some(vec![[-7.5, 12.25]; mesh.vertices.len()]);
        let png = render_poster_with(
            &mesh,
            &Appearance {
                base_color_texture: Some(solid_texture([30, 60, 240])),
                base_color_factor: [1.0; 3],
                wrap: [TextureWrap::Repeat; 2],
            },
            48,
        )
        .unwrap();
        for pixel in centre_block(&decode(&png)) {
            assert!(
                pixel[2] > pixel[0],
                "the wrapped texel is blue, got {pixel:?}"
            );
        }
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

    /// The RGB frame is the poster before PNG encoding: the same renderer a
    /// turntable stacks into an animation.
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

    /// A flat mesh seen edge-on projects to nothing. For a poster that is an
    /// error (the placeholder is the better answer); for a frame of a
    /// sequence it is a correct, background-only frame, because a turntable
    /// of a plane really does pass through that view.
    #[test]
    fn a_sequence_frame_of_an_edge_on_mesh_is_background_not_an_error() {
        let plane = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2]],
            ..Default::default()
        };
        // The plane lies in XY; an eye on +X at zero elevation sees its edge.
        let edge_on = Camera::orthographic(90.0, 0.0);
        assert!(render_frame_rgb(&plane, &edge_on, 32).is_err());
        let frame = render_sequence_frame_rgb(&plane, &edge_on, 32).expect("blank frame");
        assert_eq!((frame.width(), frame.height()), (32, 32));
        let top = frame.get_pixel(0, 0);
        assert_eq!([top[0], top[1], top[2]], BG_TOP);
        assert!(
            frame.pixels().all(|p| p[0] <= BG_TOP[0] + 1),
            "an edge-on frame must be background only"
        );
        // The same view of a mesh that is NOT flat still renders it.
        assert!(render_sequence_frame_rgb(&cube(0.5), &edge_on, 32)
            .unwrap()
            .pixels()
            .any(|p| p[0] > 100));
        // The lenient path keeps the real refusals.
        assert!(render_sequence_frame_rgb(&Mesh::default(), &edge_on, 32).is_err());
        assert!(render_sequence_frame_rgb(&plane, &edge_on, 0).is_err());
    }

    /// A looping turntable is ONE full turn whose last frame stops one step
    /// short of the first, so the loop point is a step like any other rather
    /// than a held duplicate. Every frame keeps the poster's elevation and
    /// margin, frame 0 is the poster itself, and the azimuth DECREASES so the
    /// object spins like a rightward drag.
    #[test]
    fn turntable_cameras_loop_is_a_seamless_full_turn() {
        let cameras = turntable_cameras(36, false);
        assert_eq!(cameras.len(), 36);
        let poster = poster_camera();
        assert_eq!(cameras[0], poster);
        for (index, camera) in cameras.iter().enumerate() {
            let expected = POSTER_AZIMUTH_DEG - 10.0 * index as f32;
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
            (last.azimuth_deg - (POSTER_AZIMUTH_DEG - 350.0)).abs() < 1e-3,
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
            let expected = POSTER_AZIMUTH_DEG - 22.5 * index as f32;
            assert!(
                (camera.azimuth_deg - expected).abs() < 1e-3,
                "frame {index}: azimuth {} != {expected}",
                camera.azimuth_deg
            );
        }
        let last = cameras.last().unwrap();
        assert!(
            (last.azimuth_deg - (POSTER_AZIMUTH_DEG - 180.0)).abs() < 1e-3,
            "a bounce ends exactly half a turn from the poster, got {}",
            last.azimuth_deg
        );
        assert_eq!(turntable_cameras(1, true), vec![poster_camera()]);
    }

    /// A wide, thin, off-centre plate: asymmetric on every axis, so a frame
    /// rendered from the wrong azimuth or at the wrong scale cannot
    /// accidentally match one rendered correctly.
    fn plate() -> Mesh {
        let (min, max) = ([-1.5f32, -0.1, -0.3], [1.0f32, 0.15, 0.4]);
        let v = [
            [min[0], min[1], max[2]],
            [max[0], min[1], max[2]],
            [max[0], max[1], max[2]],
            [min[0], max[1], max[2]],
            [min[0], min[1], min[2]],
            [max[0], min[1], min[2]],
            [max[0], max[1], min[2]],
            [min[0], max[1], min[2]],
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

    /// The gallery tile and the first frame of a turntable are the SAME
    /// picture — not the same camera at a different scale, the same bytes.
    ///
    /// This is what the shared rotation-invariant framing buys: a client can
    /// show the poster while the GIF or the interactive viewer loads and
    /// nothing jumps when it arrives. A per-sweep discrete fit made frame 0
    /// slightly smaller than the poster and made the difference depend on the
    /// frame count.
    #[test]
    fn the_poster_is_turntable_frame_zero_pixel_for_pixel() {
        use crate::hunyuan3d::turntable::turntable_frame_cameras;

        let mesh = plate();
        let poster = render_frame_rgb(&mesh, &poster_camera_for(&mesh), 64).expect("poster");
        // The gallery tile is this render, so `render_poster` must take the
        // framed camera and not the bare eye position.
        assert_eq!(
            decode(&render_poster(&mesh, 64).expect("poster png")),
            poster,
            "the stored poster is not rendered from the shared sweep framing"
        );
        for frames in [8usize, 36, 72] {
            let cameras = turntable_frame_cameras(&mesh, frames, false);
            let frame_zero = render_frame_rgb(&mesh, &cameras[0], 64).expect("frame 0");
            assert_eq!(
                frame_zero, poster,
                "a {frames}-frame turntable does not open on the poster"
            );
        }
        let bounce = turntable_frame_cameras(&mesh, 36, true);
        assert_eq!(
            render_frame_rgb(&mesh, &bounce[0], 64).expect("frame 0"),
            poster,
            "a bounce does not open on the poster"
        );

        // Non-vacuity: the very next frame is a different picture, so the
        // comparison above is not passing because everything renders alike.
        let cameras = turntable_frame_cameras(&mesh, 36, false);
        assert_ne!(
            render_frame_rgb(&mesh, &cameras[1], 64).expect("frame 1"),
            poster
        );
    }

    /// The framing is a property of the mesh and the elevation alone, so two
    /// GIFs of one mesh at different frame counts draw it the same size, and
    /// a bounce is framed exactly like a loop.
    ///
    /// The discrete pre-pass this replaced took the max over the cameras it
    /// was GIVEN, so 8, 36 and 72 frames each landed on their own scale and a
    /// half-turn bounce landed on a fourth.
    #[test]
    fn sweep_fit_is_frame_count_independent() {
        use crate::hunyuan3d::turntable::turntable_frame_cameras;

        let mesh = plate();
        let expected = poster_camera_for(&mesh).fit;
        assert!(
            matches!(expected, crate::hunyuan3d::raster::FrameFit::Extent(e) if e > 0.0),
            "the poster must pin an extent, got {expected:?}"
        );
        for (frames, bounce) in [(8, false), (36, false), (72, false), (36, true)] {
            for (index, camera) in turntable_frame_cameras(&mesh, frames, bounce)
                .iter()
                .enumerate()
            {
                assert_eq!(
                    camera.fit, expected,
                    "frame {index} of a {frames}-frame sweep (bounce {bounce}) is framed apart"
                );
            }
        }
    }

    /// The viewer's four literals are the poster's, or the 3-D view opens on
    /// a camera the gallery tile never used.
    ///
    /// `studio/lib/meshViewerCamera.ts` is the TS half of ONE camera
    /// convention; there is no build step that could keep the two in step, so
    /// this test reads that file and compares. The same read-the-source guard
    /// the prompting corpus uses on its own bash fences.
    #[test]
    fn the_viewer_mirrors_the_poster_camera() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../studio/lib/meshViewerCamera.ts"
        );
        let source = std::fs::read_to_string(path).unwrap_or_else(|error| {
            panic!("cannot read the viewer's camera module at {path}: {error}")
        });

        for (name, rust) in [
            ("POSTER_AZIMUTH_DEG", POSTER_AZIMUTH_DEG),
            ("POSTER_ELEVATION_DEG", POSTER_ELEVATION_DEG),
            ("POSTER_MARGIN", POSTER_MARGIN),
            ("TURNTABLE_AZIMUTH_STEP_SIGN", TURNTABLE_AZIMUTH_STEP_SIGN),
        ] {
            let ts = ts_export_const(&source, name)
                .unwrap_or_else(|| {
                    panic!(
                        "{path} does not export `{name}`. It and \
                         crates/mold-inference/src/hunyuan3d/poster.rs are ONE camera \
                         convention: change both files together."
                    )
                })
                .unwrap_or_else(|error| {
                    panic!(
                        "{path}: {error}. The export is there but its value is not a \
                         plain number this test can compare against \
                         crates/mold-inference/src/hunyuan3d/poster.rs."
                    )
                });
            assert!(
                (ts - rust).abs() < 1e-6,
                "{name} is {ts} in studio/lib/meshViewerCamera.ts and {rust} in \
                 crates/mold-inference/src/hunyuan3d/poster.rs. The viewer, the poster \
                 and the turntable are ONE camera convention: change both files \
                 together, and re-run the studio tests as well as this one."
            );
        }
    }

    /// `export const NAME = <number>;` from a TypeScript source, with an
    /// optional trailing `// comment`.
    ///
    /// A hand parser rather than a regex: `regex` is not a dependency of this
    /// crate, and the shape being matched is fixed by the file this test
    /// exists to police.
    ///
    /// The two failures are kept APART. `None` means the export is not in the
    /// file at all; `Some(Err)` means it is there and its value did not
    /// parse. Collapsing them told a reader whose only mistake was writing
    /// `= 0.08 // note` that the constant was missing, and sent them looking
    /// for a line that is right in front of them.
    #[cfg(test)]
    fn ts_export_const(source: &str, name: &str) -> Option<Result<f32, String>> {
        for line in source.lines() {
            let line = line.trim();
            let Some(rest) = line.strip_prefix("export const ") else {
                continue;
            };
            let Some((declared, value)) = rest.split_once('=') else {
                continue;
            };
            // `NAME` or `NAME: number`, either way the identifier comes first.
            let declared = declared.split(':').next().unwrap_or_default().trim();
            if declared != name {
                continue;
            }
            // `30; // a comment` -> `30`. The statement ends at the `;`, so a
            // trailing comment is dropped with everything after it; a
            // comment on a line of its own never reaches here, because it
            // does not start with `export const`.
            let value = value.split("//").next().unwrap_or_default();
            let value = value.trim().trim_end_matches(';').trim();
            return Some(
                value
                    .parse()
                    .map_err(|_| format!("could not parse the value of {name}: `{value}`")),
            );
        }
        None
    }

    /// The parser is only as good as its own pin: a renamed export, a typed
    /// declaration, a trailing comment and a different number must each be
    /// visible to it, and a missing export must not look like an unparseable
    /// one.
    #[test]
    fn the_typescript_parser_reads_what_it_claims_to() {
        let source = "export const A = 30;\nexport const B: number = -1;\nconst C = 5;\n\
                      export const LONG_NAME_A = 7;\n\
                      export const COMMENTED = 0.08; // the poster's margin\n\
                      export const NO_SEMI = 20 // trailing\n\
                      export const NOT_A_NUMBER = Math.PI / 6;\n";
        let value = |name| ts_export_const(source, name).map(|parsed| parsed.expect("a number"));
        assert_eq!(value("A"), Some(30.0));
        assert_eq!(value("B"), Some(-1.0));
        assert_eq!(value("LONG_NAME_A"), Some(7.0));
        // A trailing comment is part of the line, not part of the value.
        assert_eq!(value("COMMENTED"), Some(0.08));
        assert_eq!(value("NO_SEMI"), Some(20.0));

        // Not exported, and not present at all: `None`, never a parse error.
        assert_eq!(ts_export_const(source, "C"), None);
        assert_eq!(ts_export_const(source, "MISSING"), None);

        // Present but not a number a comparison can use: a DISTINCT answer,
        // so the caller reports the value rather than claiming the export is
        // missing.
        let error = ts_export_const(source, "NOT_A_NUMBER")
            .expect("the export is present")
            .expect_err("`Math.PI / 6` is not a plain number");
        assert!(
            error.contains("could not parse the value of NOT_A_NUMBER"),
            "{error}"
        );
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
