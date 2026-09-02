//! CPU triangle rasterizer producing world-space G-buffers.
//!
//! Pure CPU, no candle, no GPU: this runs on the save path (and later on the
//! texture path) after the GPU lock is released, so it must never contend for
//! the device.
//!
//! Two consumers, one core:
//!
//! 1. [`super::poster`] renders a single three-quarter view into a gallery
//!    tile, because mold's gallery cannot display a `.glb`.
//! 2. The PBR texture stage conditions on multi-view normal and position maps
//!    rendered from a ring of cameras ([`camera_ring`]). That stage is
//!    therefore a *caller* change — it picks different cameras and reads
//!    different [`GBuffers`] fields — not a change to this module.
//!
//! Because of (2) the buffers carry **world-space** normals and positions
//! rather than view-space ones: a multi-view conditioning stack needs every
//! view expressed in the same frame, and the poster does not care either way.
//!
//! # Orientation
//!
//! Meshes arrive in glTF's frame (Y up), which `mesh.rs`'s module docs derive
//! from the query grid's axes. So [`Camera::azimuth_deg`] orbits about `+Y`
//! and elevation lifts toward `+Y`; azimuth 0 / elevation 0 puts the eye on
//! `+Z` looking down `-Z`.
//!
//! # Screen winding
//!
//! Screen space flips Y (row 0 is the top), so a front-facing triangle — one
//! wound counter-clockwise as seen from the eye, which is glTF's front — has a
//! *negative* signed screen area. [`Culling`] is spelled in those terms and
//! there is a test pinning the sign, because getting it backwards culls
//! exactly the faces you wanted to keep and the result still looks like a
//! plausible render of the mesh's interior.

use crate::hunyuan3d::mesh::Mesh;

/// How vertices are projected onto the image plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projection {
    /// Parallel projection. The default for conditioning maps: every pixel
    /// covers the same world extent, so a position map is a linear function of
    /// the pixel grid and a downstream network does not have to undo a divide.
    Orthographic,
    /// Pinhole projection. There is deliberately no field-of-view knob: the
    /// frame is auto-fit to the mesh, so a fov could only ever disagree with
    /// the fit. The strength of the foreshortening is set by
    /// [`Camera::distance`] alone — small distances exaggerate it.
    Perspective,
}

/// Which facing gets discarded before the depth test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Culling {
    /// Draw both facings. The default, and the right choice for surface-net
    /// output: the depth buffer already resolves occlusion for a closed mesh,
    /// so culling only saves time — while a single inverted-winding triangle
    /// under `Back` punches a visible hole straight through the model.
    None,
    /// Discard triangles facing away from the eye.
    Back,
    /// Discard triangles facing the eye, leaving the far side of the surface.
    Front,
}

/// How a frame's projection scale is chosen.
///
/// [`FrameFit::Auto`] fits every render to the mesh's own projected extents,
/// which is the right answer for a single still: the subject fills the frame.
/// It is the wrong answer for a SEQUENCE, because those extents change as the
/// mesh turns — a box's projected width swings by a factor of √2 between its
/// face-on and its diagonal view — so an auto-fit turntable breathes in and
/// out once per quarter turn and pops where the x and y fits cross over.
///
/// [`FrameFit::Extent`] pins the scale to a caller-chosen half-extent, so a
/// set of cameras sharing one value renders one rigid orbit. This mirrors
/// ComfyUI's splat turntable, which frames its default camera ONCE from a
/// rotation-invariant extent
/// (`comfy_extras/nodes_gaussian_splat.py:996-1006`) and then rotates that
/// one camera rigidly per frame (`_orbit_camera_info_yaw`, `:640-655`).
///
/// See [`frame_fit_for`] for building the shared value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameFit {
    /// Fit this frame to the mesh's own projected extents.
    Auto,
    /// Map this projected half-extent onto the half-frame:
    /// `scale = min(half_w, half_h) / extent * (1 - margin)`.
    ///
    /// The value is in the camera's *projected* units — the same units
    /// [`projected_half_extent`] reports — not pixels and not world units, so
    /// one extent is meaningful at any resolution. A non-finite or
    /// non-positive extent is not a framing, and a render using it draws
    /// nothing rather than guessing.
    Extent(f32),
}

/// A view of a mesh: an orbit position plus how the frame is fitted.
///
/// The camera holds no absolute placement. It aims at the mesh's bounding-box
/// centre and the projection scale is fitted to the mesh at render time, so
/// one camera renders any mesh at any resolution and always fills the frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// Orbit angle about `+Y`, degrees. 0 places the eye on `+Z`.
    pub azimuth_deg: f32,
    /// Angle above the XZ plane, degrees. Positive looks down at the mesh.
    pub elevation_deg: f32,
    /// Eye distance from the bounding-box centre, in units of the mesh's
    /// bounding-sphere radius. Under [`Projection::Orthographic`] this only
    /// sets where depth is measured from; under [`Projection::Perspective`] it
    /// is the entire shape of the projection. Clamped to
    /// [`MIN_PERSPECTIVE_DISTANCE`] for perspective so the whole bounding
    /// sphere stays in front of the eye and no near-plane clipping is needed.
    pub distance: f32,
    pub projection: Projection,
    /// Fraction of the frame left empty around the mesh, in `[0, 1)`. A little
    /// margin keeps a silhouette from being cut off by the auto-fit, which
    /// fits the *projected bounding box* and so touches the frame edge exactly.
    pub margin: f32,
    pub culling: Culling,
    /// How the projection scale is chosen. [`FrameFit::Auto`] unless a caller
    /// stamps a shared extent on a whole sequence.
    pub fit: FrameFit,
}

/// Below this the bounding sphere would straddle the eye and a perspective
/// render would need real near-plane clipping. Clamping instead keeps the
/// rasterizer free of a clipper it would otherwise need for one cosmetic knob.
pub const MIN_PERSPECTIVE_DISTANCE: f32 = 1.25;

/// Empty border left by the auto-fit unless a caller says otherwise.
pub const DEFAULT_MARGIN: f32 = 0.06;

impl Camera {
    /// Orthographic view from `azimuth_deg` / `elevation_deg`.
    pub fn orthographic(azimuth_deg: f32, elevation_deg: f32) -> Self {
        Self {
            azimuth_deg,
            elevation_deg,
            distance: 2.5,
            projection: Projection::Orthographic,
            margin: DEFAULT_MARGIN,
            culling: Culling::None,
            fit: FrameFit::Auto,
        }
    }

    /// Perspective view. `distance` is in bounding-sphere radii and is clamped
    /// to [`MIN_PERSPECTIVE_DISTANCE`].
    pub fn perspective(azimuth_deg: f32, elevation_deg: f32, distance: f32) -> Self {
        Self {
            distance: distance.max(MIN_PERSPECTIVE_DISTANCE),
            projection: Projection::Perspective,
            ..Self::orthographic(azimuth_deg, elevation_deg)
        }
    }

    pub fn with_margin(mut self, margin: f32) -> Self {
        self.margin = margin.clamp(0.0, 0.9);
        self
    }

    pub fn with_culling(mut self, culling: Culling) -> Self {
        self.culling = culling;
        self
    }

    /// Replace the framing rule. See [`FrameFit`].
    pub fn with_fit(mut self, fit: FrameFit) -> Self {
        self.fit = fit;
        self
    }

    /// Unit vector from the look-at point toward the eye, in world space.
    ///
    /// Also the direction "toward the camera" for every shading term, which is
    /// why it is public: [`super::poster`] needs it and recomputing the
    /// trigonometry there would let the two drift apart.
    pub fn view_direction(&self) -> [f32; 3] {
        let az = self.azimuth_deg.to_radians();
        let el = self.elevation_deg.to_radians();
        let (sa, ca) = az.sin_cos();
        let (se, ce) = el.sin_cos();
        [ce * sa, se, ce * ca]
    }

    /// Right / up / toward-eye basis, in world space.
    ///
    /// Degenerate straight up or straight down (`|elevation| = 90`) has no
    /// unique "up", so the world up used to build the basis swings to `+Z`
    /// there. Without that the cross product collapses and every screen
    /// coordinate becomes NaN.
    pub fn basis(&self) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let dir = self.view_direction();
        let world_up = if dir[1].abs() > 0.999 {
            [0.0, 0.0, 1.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        // right = up x dir is the +X of a right-handed eye frame looking down -dir.
        let right = normalize(cross(world_up, dir));
        let up = normalize(cross(dir, right));
        (right, up, dir)
    }
}

/// Per-pixel geometry, all in world space except `depth`.
///
/// Every vector is `width * height` long and indexed `y * width + x`. Entries
/// where `mask` is false are meaningless, not zero-valued geometry — callers
/// must gate on `mask` rather than testing for a sentinel.
#[derive(Debug, Clone, PartialEq)]
pub struct GBuffers {
    /// Distance from the eye along the view axis. Smaller is nearer. `INFINITY`
    /// where nothing was drawn.
    pub depth: Vec<f32>,
    /// Interpolated, re-normalized shading normal. Follows the mesh's own
    /// winding: it is NOT flipped toward the eye, so a caller that wants
    /// two-sided shading flips it itself.
    pub normal: Vec<[f32; 3]>,
    /// Interpolated world-space surface point.
    pub position: Vec<[f32; 3]>,
    pub mask: Vec<bool>,
    pub width: u32,
    pub height: u32,
}

impl GBuffers {
    /// A buffer with nothing drawn in it — what an empty, degenerate or
    /// entirely off-screen mesh renders to.
    pub fn unset(width: u32, height: u32) -> Self {
        let n = width as usize * height as usize;
        Self {
            depth: vec![f32::INFINITY; n],
            normal: vec![[0.0; 3]; n],
            position: vec![[0.0; 3]; n],
            mask: vec![false; n],
            width,
            height,
        }
    }

    pub fn len(&self) -> usize {
        self.mask.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    pub fn covered_pixels(&self) -> usize {
        self.mask.iter().filter(|c| **c).count()
    }

    /// Flat index of `(x, y)`, or `None` when it is outside the frame.
    pub fn index(&self, x: u32, y: u32) -> Option<usize> {
        (x < self.width && y < self.height).then(|| y as usize * self.width as usize + x as usize)
    }
}

/// Evenly spaced orbit cameras at a fixed elevation.
///
/// The multi-view conditioning stage feeds a fixed number of views into the
/// texture model, and those views must be reproducible across runs — hence a
/// deterministic ring rather than anything sampled.
pub fn camera_ring(count: usize, elevation_deg: f32) -> Vec<Camera> {
    (0..count)
        .map(|i| {
            let azimuth = 360.0 * i as f32 / count as f32;
            Camera::orthographic(azimuth, elevation_deg)
        })
        .collect()
}

/// Rasterize `mesh` from `camera` into `width * height` G-buffers.
///
/// Never fails and never panics. An empty mesh, a mesh whose vertices are all
/// coincident, and a zero-sized frame all produce an all-unset buffer, which is
/// the honest answer: there was nothing to draw. Individual bad triangles
/// (degenerate area, non-finite coordinate, out-of-range index) are skipped
/// while the rest of the mesh still renders.
pub fn render_gbuffers(mesh: &Mesh, camera: &Camera, width: u32, height: u32) -> GBuffers {
    let mut gb = GBuffers::unset(width, height);
    if width == 0 || height == 0 || mesh.vertices.is_empty() || mesh.faces.is_empty() {
        return gb;
    }

    let Some(screen) = project(mesh, camera, width, height) else {
        return gb;
    };

    let normals = mesh
        .normals
        .as_ref()
        .filter(|n| n.len() == mesh.vertices.len());

    let w = width as i64;
    let h = height as i64;
    let vcount = mesh.vertices.len();

    for face in &mesh.faces {
        let (i0, i1, i2) = (face[0] as usize, face[1] as usize, face[2] as usize);
        if i0 >= vcount || i1 >= vcount || i2 >= vcount {
            continue;
        }
        let (s0, s1, s2) = (screen[i0], screen[i1], screen[i2]);
        if !s0.finite() || !s1.finite() || !s2.finite() {
            continue;
        }

        // Signed screen area (twice the triangle area). Zero means the triangle
        // projects to a line or a point and has no interior to shade; it is
        // also the divisor for the barycentrics, so it must be tested first.
        let area2 = (s1.x - s0.x) * (s2.y - s0.y) - (s2.x - s0.x) * (s1.y - s0.y);
        if area2 == 0.0 || !area2.is_finite() {
            continue;
        }
        match camera.culling {
            Culling::None => {}
            // Negative area == front-facing; see the module docs.
            Culling::Back if area2 > 0.0 => continue,
            Culling::Front if area2 < 0.0 => continue,
            _ => {}
        }

        let min_x = (s0.x.min(s1.x).min(s2.x).floor() as i64).max(0);
        let max_x = (s0.x.max(s1.x).max(s2.x).ceil() as i64).min(w - 1);
        let min_y = (s0.y.min(s1.y).min(s2.y).floor() as i64).max(0);
        let max_y = (s0.y.max(s1.y).max(s2.y).ceil() as i64).min(h - 1);
        if min_x > max_x || min_y > max_y {
            continue;
        }

        let (p0, p1, p2) = (mesh.vertices[i0], mesh.vertices[i1], mesh.vertices[i2]);
        // Geometric normal, used when the mesh carries none and as the fallback
        // for a vertex normal that cancelled itself out.
        let face_n = normalize(cross(sub(p1, p0), sub(p2, p0)));
        let (n0, n1, n2) = match normals {
            Some(n) => (n[i0], n[i1], n[i2]),
            None => (face_n, face_n, face_n),
        };

        let inv_area = 1.0 / area2;
        for py in min_y..=max_y {
            let sy = py as f32 + 0.5;
            let row = py as usize * width as usize;
            for px in min_x..=max_x {
                let sx = px as f32 + 0.5;

                // Edge functions, each opposite its like-numbered vertex.
                // Dividing by the signed area makes them barycentric for either
                // winding, so the "all non-negative" interior test works
                // without a separate orientation branch.
                let e0 = (s2.x - s1.x) * (sy - s1.y) - (s2.y - s1.y) * (sx - s1.x);
                let e1 = (s0.x - s2.x) * (sy - s2.y) - (s0.y - s2.y) * (sx - s2.x);
                let e2 = (s1.x - s0.x) * (sy - s0.y) - (s1.y - s0.y) * (sx - s0.x);
                let l0 = e0 * inv_area;
                let l1 = e1 * inv_area;
                let l2 = e2 * inv_area;
                if l0 < 0.0 || l1 < 0.0 || l2 < 0.0 {
                    continue;
                }

                // Perspective-correct weights. Under an orthographic camera
                // every `inv_w` is 1, the divisor collapses to 1, and this
                // reduces exactly to the screen-linear weights.
                let denom = l0 * s0.inv_w + l1 * s1.inv_w + l2 * s2.inv_w;
                if denom <= 0.0 || !denom.is_finite() {
                    continue;
                }
                let inv_denom = 1.0 / denom;
                let b0 = l0 * s0.inv_w * inv_denom;
                let b1 = l1 * s1.inv_w * inv_denom;
                let b2 = l2 * s2.inv_w * inv_denom;

                let depth = b0 * s0.depth + b1 * s1.depth + b2 * s2.depth;
                if !depth.is_finite() {
                    continue;
                }
                let idx = row + px as usize;
                if depth >= gb.depth[idx] {
                    continue;
                }

                let mut n = [
                    b0 * n0[0] + b1 * n1[0] + b2 * n2[0],
                    b0 * n0[1] + b1 * n1[1] + b2 * n2[1],
                    b0 * n0[2] + b1 * n1[2] + b2 * n2[2],
                ];
                if length(n) <= 1e-12 {
                    n = face_n;
                } else {
                    n = normalize(n);
                }

                gb.depth[idx] = depth;
                gb.normal[idx] = n;
                gb.position[idx] = [
                    b0 * p0[0] + b1 * p1[0] + b2 * p2[0],
                    b0 * p0[1] + b1 * p1[1] + b2 * p2[1],
                    b0 * p0[2] + b1 * p1[2] + b2 * p2[2],
                ];
                gb.mask[idx] = true;
            }
        }
    }

    gb
}

/// A projected vertex, precomputed once so the triangle loop allocates nothing.
#[derive(Debug, Clone, Copy)]
struct ScreenVertex {
    x: f32,
    y: f32,
    /// Eye-space depth, increasing away from the eye.
    depth: f32,
    /// `1/depth` under perspective, `1` under orthographic.
    inv_w: f32,
}

impl ScreenVertex {
    fn finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.depth.is_finite() && self.inv_w.is_finite()
    }
}

/// Where the eye sits and which way the axes run for one view of one mesh.
///
/// The mesh-dependent half of a projection: computed once per render and
/// shared by every vertex, so [`projected_half_extent`] and [`project`] can
/// never disagree about the framing they are measuring.
#[derive(Debug, Clone, Copy)]
struct ViewFrame {
    center: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    dir: [f32; 3],
    /// Eye distance from `center`, in world units.
    eye_dist: f32,
    projection: Projection,
}

/// The view frame for `camera` looking at `mesh`.
///
/// `None` means there is nothing to look at: non-finite bounds, or a mesh
/// whose vertices are all coincident.
fn view_frame(mesh: &Mesh, camera: &Camera) -> Option<ViewFrame> {
    let (min, max) = mesh.bounds();
    if !min.iter().chain(max.iter()).all(|v| v.is_finite()) {
        // `bounds` propagates a NaN coordinate into the extremes, and a NaN
        // centre would poison every vertex rather than just its own faces.
        return None;
    }
    let center = [
        0.5 * (min[0] + max[0]),
        0.5 * (min[1] + max[1]),
        0.5 * (min[2] + max[2]),
    ];
    let radius = mesh
        .vertices
        .iter()
        .map(|v| length(sub(*v, center)))
        .filter(|r| r.is_finite())
        .fold(0.0f32, f32::max);
    if radius <= 0.0 || !radius.is_finite() {
        return None;
    }

    let (right, up, dir) = camera.basis();
    let distance = match camera.projection {
        Projection::Orthographic => camera.distance.max(1.0),
        Projection::Perspective => camera.distance.max(MIN_PERSPECTIVE_DISTANCE),
    };
    Some(ViewFrame {
        center,
        right,
        up,
        dir,
        eye_dist: distance * radius,
        projection: camera.projection,
    })
}

/// One vertex in the camera's *projected* units, before the fit scales it to
/// pixels. Shared by [`project`] and [`projected_half_extent`].
fn project_vertex(vertex: [f32; 3], frame: &ViewFrame) -> ScreenVertex {
    let d = sub(vertex, frame.center);
    let depth = frame.eye_dist - dot(d, frame.dir);
    let (x, y, inv_w) = match frame.projection {
        Projection::Orthographic => (dot(d, frame.right), dot(d, frame.up), 1.0),
        Projection::Perspective => {
            let inv = 1.0 / depth;
            (
                dot(d, frame.right) * frame.eye_dist * inv,
                dot(d, frame.up) * frame.eye_dist * inv,
                inv,
            )
        }
    };
    ScreenVertex { x, y, depth, inv_w }
}

/// The mesh's projected half-extents `(x, y)` from `camera`, in the camera's
/// projected units and ignoring `camera.fit`.
///
/// This is the quantity an auto-fit divides the half-frame by, so it is also
/// the quantity a caller compares across a sweep to see the breathing
/// [`FrameFit::Extent`] exists to remove. `None` for a mesh with nothing to
/// look at, exactly as [`render_gbuffers`] draws nothing for one.
pub fn projected_half_extent(mesh: &Mesh, camera: &Camera) -> Option<(f32, f32)> {
    let frame = view_frame(mesh, camera)?;
    let (mut ext_x, mut ext_y) = (0.0f32, 0.0f32);
    for v in &mesh.vertices {
        let s = project_vertex(*v, &frame);
        if s.x.is_finite() && s.y.is_finite() {
            ext_x = ext_x.max(s.x.abs());
            ext_y = ext_y.max(s.y.abs());
        }
    }
    Some((ext_x, ext_y))
}

/// Projected units to pixels, given the extents an [`FrameFit::Auto`] fit
/// would use. Under [`FrameFit::Extent`] the extents are ignored.
fn fit_scale(camera: &Camera, width: u32, height: u32, ext_x: f32, ext_y: f32) -> Option<f32> {
    let half_w = 0.5 * width as f32;
    let half_h = 0.5 * height as f32;
    let scale = match camera.fit {
        FrameFit::Extent(extent) => {
            if !extent.is_finite() || extent <= 0.0 {
                return None;
            }
            half_w.min(half_h) / extent * (1.0 - camera.margin.clamp(0.0, 0.9))
        }
        FrameFit::Auto => {
            let fit_x = if ext_x > 0.0 {
                half_w / ext_x
            } else {
                f32::INFINITY
            };
            let fit_y = if ext_y > 0.0 {
                half_h / ext_y
            } else {
                f32::INFINITY
            };
            fit_x.min(fit_y) * (1.0 - camera.margin.clamp(0.0, 0.9))
        }
    };
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    Some(scale)
}

/// Pixels per projected unit for this view at this resolution.
///
/// The number a sequence must hold constant: it IS the on-screen size of the
/// mesh. `None` wherever [`render_gbuffers`] would draw nothing.
pub fn projection_scale(mesh: &Mesh, camera: &Camera, width: u32, height: u32) -> Option<f32> {
    match camera.fit {
        // A pinned extent needs no measurement of the mesh, but a mesh with
        // nothing to look at still renders nothing, so the frame must exist.
        FrameFit::Extent(_) => {
            view_frame(mesh, camera)?;
            fit_scale(camera, width, height, 0.0, 0.0)
        }
        FrameFit::Auto => {
            let (ext_x, ext_y) = projected_half_extent(mesh, camera)?;
            fit_scale(camera, width, height, ext_x, ext_y)
        }
    }
}

/// One [`FrameFit::Extent`] that frames `mesh` from EVERY camera in `cameras`.
///
/// The largest half-extent any of those views needs, on either axis — so
/// stamping it on all of them gives a rigid orbit in which nothing is ever
/// clipped and nothing changes size. `None` when there is nothing to frame
/// (an empty camera list, or a mesh with no extent), and the caller keeps
/// [`FrameFit::Auto`].
///
/// The pre-pass mirrors ComfyUI's splat turntable, which computes its default
/// camera once from a rotation-invariant extent
/// (`comfy_extras/nodes_gaussian_splat.py:996-1006`) and then orbits that one
/// camera rigidly (`:640-655`).
pub fn frame_fit_for(mesh: &Mesh, cameras: &[Camera]) -> Option<FrameFit> {
    let mut extent = 0.0f32;
    for camera in cameras {
        let (ext_x, ext_y) = projected_half_extent(mesh, camera)?;
        extent = extent.max(ext_x).max(ext_y);
    }
    (extent.is_finite() && extent > 0.0).then_some(FrameFit::Extent(extent))
}

/// Project every vertex to pixel coordinates under `camera.fit`.
///
/// `None` means there is nothing renderable: no finite vertex, a zero-radius
/// mesh, or a fit that came out non-finite. Non-finite *individual* vertices
/// survive as NaN entries and are dropped later per triangle, so one bad vertex
/// costs its incident faces and not the whole render.
fn project(mesh: &Mesh, camera: &Camera, width: u32, height: u32) -> Option<Vec<ScreenVertex>> {
    let frame = view_frame(mesh, camera)?;

    let mut out = Vec::with_capacity(mesh.vertices.len());
    let (mut ext_x, mut ext_y) = (0.0f32, 0.0f32);
    for v in &mesh.vertices {
        let s = project_vertex(*v, &frame);
        if s.x.is_finite() && s.y.is_finite() {
            ext_x = ext_x.max(s.x.abs());
            ext_y = ext_y.max(s.y.abs());
        }
        out.push(s);
    }

    let scale = fit_scale(camera, width, height, ext_x, ext_y)?;
    let half_w = 0.5 * width as f32;
    let half_h = 0.5 * height as f32;
    for s in &mut out {
        s.x = half_w + scale * s.x;
        // Screen rows run downward, so +Y in world becomes -Y on screen.
        s.y = half_h - scale * s.y;
    }
    Some(out)
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

/// Unit-length `a`, or `+Y` when `a` has no direction to preserve. Matching
/// `mesh::compute_smooth_normals`'s degenerate answer keeps a fallback normal
/// from being the one value that is never a legal unit vector.
fn normalize(a: [f32; 3]) -> [f32; 3] {
    let len = length(a);
    if len > 1e-20 && len.is_finite() {
        [a[0] / len, a[1] / len, a[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Axis-aligned cube spanning `[-half, half]` on every axis, wound
    /// counter-clockwise seen from outside (glTF's front facing).
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
            [0, 1, 2, 3], // +Z
            [5, 4, 7, 6], // -Z
            [1, 5, 6, 2], // +X
            [4, 0, 3, 7], // -X
            [3, 2, 6, 7], // +Y
            [4, 5, 1, 0], // -Y
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

    /// Axis-aligned quad in the XY plane at `z`, spanning `[-half, half]`.
    fn quad(half: f32, z: f32) -> ([[f32; 3]; 4], [[u32; 3]; 2]) {
        (
            [
                [-half, -half, z],
                [half, -half, z],
                [half, half, z],
                [-half, half, z],
            ],
            [[0, 1, 2], [0, 2, 3]],
        )
    }

    #[test]
    fn head_on_cube_covers_the_fitted_frame_and_faces_the_eye() {
        let mesh = cube(0.5);
        let cam = Camera::orthographic(0.0, 0.0).with_margin(0.0);
        let gb = render_gbuffers(&mesh, &cam, 64, 64);

        // Head-on, the auto-fit scales the cube's half-extent to half the
        // frame, so the +Z face covers every pixel. One row/column of pixel
        // centres can land exactly on the silhouette edge, so allow a border.
        let covered = gb.covered_pixels();
        assert!(
            (62 * 62..=64 * 64).contains(&covered),
            "expected a nearly full frame, got {covered}"
        );

        for i in 0..gb.len() {
            assert_eq!(
                gb.mask[i],
                gb.depth[i].is_finite(),
                "depth must be finite exactly where the mask is set (pixel {i})"
            );
            if gb.mask[i] {
                // The eye is on +Z, so the visible face's normal is +Z.
                assert!(
                    gb.normal[i][2] > 0.99,
                    "pixel {i} normal {:?} does not face the eye",
                    gb.normal[i]
                );
                assert!((gb.position[i][2] - 0.5).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn nearer_quad_wins_every_contested_pixel() {
        let (near_v, tris) = quad(0.5, 1.0);
        let (far_v, _) = quad(0.5, -1.0);
        let mut vertices = far_v.to_vec();
        vertices.extend_from_slice(&near_v);
        let mut faces: Vec<[u32; 3]> = tris.to_vec();
        faces.extend(tris.iter().map(|t| [t[0] + 4, t[1] + 4, t[2] + 4]));

        let mesh = Mesh {
            vertices,
            faces,
            ..Default::default()
        };
        let cam = Camera::orthographic(0.0, 0.0).with_margin(0.1);
        let gb = render_gbuffers(&mesh, &cam, 48, 48);

        assert!(gb.covered_pixels() > 0);
        for i in 0..gb.len() {
            if gb.mask[i] {
                assert!(
                    (gb.position[i][2] - 1.0).abs() < 1e-4,
                    "pixel {i} shows the far quad at z={}",
                    gb.position[i][2]
                );
            }
        }

        // Face order must not matter: the same scene with the far quad drawn
        // last still resolves to the near one.
        let mut swapped = mesh.clone();
        swapped.faces.reverse();
        let gb2 = render_gbuffers(&swapped, &cam, 48, 48);
        assert_eq!(gb.mask, gb2.mask);
        for i in 0..gb2.len() {
            if gb2.mask[i] {
                assert!((gb2.position[i][2] - 1.0).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn front_facing_triangles_have_negative_screen_area() {
        // Pins the sign the `Culling` arms are written against.
        let mesh = cube(0.5);
        let cam = Camera::orthographic(0.0, 0.0).with_margin(0.0);
        let culled = render_gbuffers(&mesh, &cam.with_culling(Culling::Back), 32, 32);
        let inverted = render_gbuffers(&mesh, &cam.with_culling(Culling::Front), 32, 32);

        // Back-culling keeps the face pointing at us; front-culling removes it
        // and leaves the far side, which is further away.
        assert!(culled.covered_pixels() > 0);
        assert!(inverted.covered_pixels() > 0);
        for i in 0..culled.len() {
            if culled.mask[i] {
                assert!(culled.normal[i][2] > 0.99);
            }
            if inverted.mask[i] {
                assert!(inverted.normal[i][2] < -0.99);
            }
        }
    }

    #[test]
    fn vertex_normals_are_interpolated_when_present() {
        let mut mesh = cube(0.5);
        crate::hunyuan3d::mesh::compute_smooth_normals(&mut mesh);
        let gb = render_gbuffers(
            &mesh,
            &Camera::orthographic(0.0, 0.0).with_margin(0.0),
            32,
            32,
        );

        // Smooth cube normals point at the corners, so the centre of the +Z
        // face interpolates to something near +Z while a corner does not.
        let mid = gb.index(16, 16).expect("centre pixel");
        assert!(gb.mask[mid]);
        assert!(gb.normal[mid][2] > 0.5);
        let corner = gb.index(1, 1).expect("corner pixel");
        assert!(gb.mask[corner]);
        assert!(gb.normal[corner][2] < gb.normal[mid][2]);
        for n in gb
            .normal
            .iter()
            .zip(&gb.mask)
            .filter(|(_, m)| **m)
            .map(|(n, _)| n)
        {
            assert!((length(*n) - 1.0).abs() < 1e-4, "normal {n:?} is not unit");
        }
    }

    #[test]
    fn degenerate_meshes_render_an_unset_buffer_without_panicking() {
        let cam = Camera::orthographic(30.0, 20.0);
        let unset =
            |gb: &GBuffers| gb.covered_pixels() == 0 && gb.depth.iter().all(|d| d.is_infinite());

        assert!(unset(&render_gbuffers(&Mesh::default(), &cam, 16, 16)));

        // Zero-area triangles: three collinear vertices, and a triangle whose
        // vertices coincide.
        let collinear = Mesh {
            vertices: vec![[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            faces: vec![[0, 1, 2]],
            ..Default::default()
        };
        assert!(unset(&render_gbuffers(&collinear, &cam, 16, 16)));

        let coincident = Mesh {
            vertices: vec![[0.25, 0.25, 0.25]; 3],
            faces: vec![[0, 1, 2]],
            ..Default::default()
        };
        assert!(unset(&render_gbuffers(&coincident, &cam, 16, 16)));

        // An all-NaN mesh: `Mesh::bounds` compares with `<`/`>`, which are
        // false for NaN, so the extremes stay at their infinite seeds and the
        // fit has nothing to work with.
        let all_nan = Mesh {
            vertices: vec![[f32::NAN; 3]; 3],
            faces: vec![[0, 1, 2]],
            ..Default::default()
        };
        assert!(unset(&render_gbuffers(&all_nan, &cam, 16, 16)));

        // An infinite coordinate does reach the bounds, and an infinite centre
        // would poison every vertex rather than just its own faces.
        let mut inf = cube(0.5);
        inf.vertices[3][0] = f32::INFINITY;
        assert!(unset(&render_gbuffers(&inf, &cam, 16, 16)));

        // Out-of-range face indices are skipped, not indexed.
        let bad_index = Mesh {
            faces: vec![[0, 1, 99]],
            ..cube(0.5)
        };
        assert!(unset(&render_gbuffers(&bad_index, &cam, 16, 16)));

        // A zero-sized frame is not a panic either.
        assert!(render_gbuffers(&cube(0.5), &cam, 0, 0).is_empty());
    }

    #[test]
    fn a_nan_vertex_only_costs_its_own_faces() {
        // `Mesh::bounds` skips NaN, so an otherwise sound mesh still fits and
        // renders; only the triangles touching the bad vertex drop out.
        let mut mesh = cube(0.5);
        mesh.vertices[0] = [f32::NAN; 3];
        let gb = render_gbuffers(
            &mesh,
            &Camera::orthographic(0.0, 0.0).with_margin(0.0),
            32,
            32,
        );
        assert!(gb.covered_pixels() > 0, "the good faces must still render");

        // Vertex 0 is the lower-left corner of the +Z face, so the triangle
        // covering that half is gone and the -Z face shows through the hole.
        let hole = gb.index(6, 30).expect("lower-left pixel");
        assert!(gb.mask[hole]);
        assert!(
            gb.position[hole][2] < 0.0,
            "expected the far face through the hole, got z={}",
            gb.position[hole][2]
        );
    }

    #[test]
    fn camera_ring_is_evenly_spaced_and_distinct() {
        let ring = camera_ring(6, 0.0);
        assert_eq!(ring.len(), 6);
        for (i, cam) in ring.iter().enumerate() {
            assert!((cam.azimuth_deg - 60.0 * i as f32).abs() < 1e-4);
            assert_eq!(cam.elevation_deg, 0.0);
        }
        for i in 0..ring.len() {
            for j in (i + 1)..ring.len() {
                assert_ne!(ring[i], ring[j], "cameras {i} and {j} are identical");
            }
        }
        assert!(camera_ring(0, 0.0).is_empty());
    }

    #[test]
    fn straight_down_still_produces_a_basis() {
        // Elevation 90 makes the view direction parallel to world up; without
        // the fallback the basis collapses and every pixel is NaN.
        let gb = render_gbuffers(&cube(0.5), &Camera::orthographic(0.0, 90.0), 32, 32);
        assert!(gb.covered_pixels() > 0);
        for i in 0..gb.len() {
            if gb.mask[i] {
                assert!(gb.normal[i][1] > 0.99, "looking down should show +Y");
            }
        }
    }

    #[test]
    fn perspective_depth_varies_across_a_slanted_surface() {
        // Off-axis so real surface slant is visible: an orthographic camera
        // would still give a depth range, but a perspective one must also keep
        // every sample strictly in front of the eye, which is the property the
        // `MIN_PERSPECTIVE_DISTANCE` clamp exists to guarantee.
        let mesh = cube(0.5);
        let cam = Camera::perspective(30.0, 20.0, 1.0);
        assert_eq!(
            cam.distance, MIN_PERSPECTIVE_DISTANCE,
            "distance is clamped"
        );
        let gb = render_gbuffers(&mesh, &cam, 64, 64);
        let depths: Vec<f32> = gb
            .depth
            .iter()
            .zip(&gb.mask)
            .filter(|(_, m)| **m)
            .map(|(d, _)| *d)
            .collect();
        assert!(!depths.is_empty());
        let lo = depths.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = depths.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(hi > lo, "depth should vary across a slanted surface");
        assert!(lo > 0.0, "nothing may sit behind the eye");
    }

    /// A camera fits itself to the mesh unless a caller says otherwise, and
    /// that is the framing the poster has always used.
    #[test]
    fn auto_is_the_default_fit_and_leaves_the_render_alone() {
        let cam = Camera::orthographic(30.0, 20.0);
        assert_eq!(cam.fit, FrameFit::Auto);
        assert_eq!(Camera::perspective(30.0, 20.0, 2.0).fit, FrameFit::Auto);
        assert_eq!(cam.with_margin(0.2).fit, FrameFit::Auto);
        assert_eq!(cam.with_culling(Culling::Back).fit, FrameFit::Auto);
        assert_eq!(
            cam.with_fit(FrameFit::Extent(1.0)).fit,
            FrameFit::Extent(1.0)
        );

        // Explicitly asking for the default changes nothing about the pixels.
        let mesh = cube(0.5);
        assert_eq!(
            render_gbuffers(&mesh, &cam, 32, 32),
            render_gbuffers(&mesh, &cam.with_fit(FrameFit::Auto), 32, 32)
        );
    }

    /// `projected_half_extent` reports exactly the quantity the auto-fit
    /// divides the half-frame by, so the two can never drift apart.
    #[test]
    fn projected_half_extent_agrees_with_the_auto_fit() {
        let mesh = cube(0.5);
        for camera in [
            Camera::orthographic(0.0, 0.0).with_margin(0.0),
            Camera::orthographic(30.0, 20.0),
            Camera::perspective(45.0, -10.0, 3.0).with_margin(0.1),
        ] {
            let (ext_x, ext_y) = projected_half_extent(&mesh, &camera).expect("extents");
            let scale = projection_scale(&mesh, &camera, 64, 48).expect("scale");
            let expected = (32.0f32 / ext_x).min(24.0 / ext_y) * (1.0 - camera.margin);
            assert!(
                (scale - expected).abs() <= 1e-4 * expected,
                "{scale} != {expected}"
            );
            // The fitted silhouette touches the frame edge exactly, less the
            // margin, on whichever axis is tighter.
            assert!(scale * ext_x <= 32.0 + 1e-3);
            assert!(scale * ext_y <= 24.0 + 1e-3);
        }
        assert_eq!(
            projected_half_extent(&Mesh::default(), &Camera::orthographic(0.0, 0.0)),
            None
        );
    }

    /// A pinned extent is the whole fit: two meshes of very different sizes
    /// seen through cameras carrying the same [`FrameFit::Extent`] render at
    /// the SAME pixels-per-unit, which is what makes a sweep rigid.
    #[test]
    fn a_pinned_extent_overrides_the_per_frame_autofit() {
        let small = cube(0.25);
        let large = cube(4.0);
        let camera = Camera::orthographic(30.0, 20.0)
            .with_margin(0.08)
            .with_fit(FrameFit::Extent(2.5));

        let expected = 32.0f32.min(24.0) / 2.5 * (1.0 - 0.08);
        for mesh in [&small, &large] {
            let scale = projection_scale(mesh, &camera, 64, 48).expect("scale");
            assert!(
                (scale - expected).abs() <= 1e-5 * expected,
                "{scale} != {expected}"
            );
        }

        // Under Auto the same two meshes disagree by their size ratio; the
        // pin is doing real work.
        let auto = camera.with_fit(FrameFit::Auto);
        let a = projection_scale(&small, &auto, 64, 48).expect("scale");
        let b = projection_scale(&large, &auto, 64, 48).expect("scale");
        assert!(a > 10.0 * b, "auto fit did not track the mesh: {a} vs {b}");

        // A frame-filling render is still a render: the pinned extent covers
        // the small cube with room to spare and clips nothing.
        let gb = render_gbuffers(&small, &camera, 64, 48);
        assert!(gb.covered_pixels() > 0);
    }

    /// A degenerate pin is not a framing. Rather than guessing a scale, the
    /// render comes back unset — the same answer a degenerate mesh gets.
    #[test]
    fn a_non_positive_or_non_finite_pinned_extent_renders_nothing() {
        let mesh = cube(0.5);
        for extent in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let camera = Camera::orthographic(30.0, 20.0).with_fit(FrameFit::Extent(extent));
            assert_eq!(
                projection_scale(&mesh, &camera, 32, 32),
                None,
                "extent {extent}"
            );
            let gb = render_gbuffers(&mesh, &camera, 32, 32);
            assert_eq!(gb.covered_pixels(), 0, "extent {extent} drew something");
            assert!(gb.depth.iter().all(|d| d.is_infinite()));
            assert!(gb
                .normal
                .iter()
                .chain(gb.position.iter())
                .all(|v| v.iter().all(|c| c.is_finite())));
        }
        // And a pin on a mesh with nothing to look at is still nothing.
        assert_eq!(
            projection_scale(
                &Mesh::default(),
                &Camera::orthographic(0.0, 0.0).with_fit(FrameFit::Extent(1.0)),
                32,
                32
            ),
            None
        );
    }

    /// The shared fit is the largest half-extent any camera in the set needs,
    /// so no view is ever clipped and every view is the same size.
    #[test]
    fn frame_fit_for_covers_every_camera_in_the_set() {
        let mesh = cube(0.5);
        let cameras = camera_ring(12, 20.0);
        let FrameFit::Extent(extent) = frame_fit_for(&mesh, &cameras).expect("a fit") else {
            panic!("frame_fit_for must pin an extent");
        };
        let mut largest = 0.0f32;
        for camera in &cameras {
            let (ext_x, ext_y) = projected_half_extent(&mesh, camera).expect("extents");
            assert!(ext_x <= extent + 1e-6 && ext_y <= extent + 1e-6);
            largest = largest.max(ext_x).max(ext_y);
        }
        assert!((extent - largest).abs() < 1e-6);

        assert_eq!(frame_fit_for(&mesh, &[]), None, "nothing to frame");
        assert_eq!(frame_fit_for(&Mesh::default(), &cameras), None);
    }
}
