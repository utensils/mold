//! Five-point face alignment: the two templates PuLID needs and the
//! least-squares similarity fit that lands landmarks on them.
//!
//! ## The 112x112 ArcFace crop
//!
//! `insightface/python-package/insightface/utils/face_align.py:6-9` defines
//! `arcface_dst`, and `:11-24` (`estimate_norm`) fits it with
//! `skimage.transform.SimilarityTransform`, whose `estimate` is Umeyama's
//! closed-form optimal similarity (`skimage/transform/_geometric.py`,
//! `_umeyama(src, dst, estimate_scale=True)`). [`umeyama_similarity`] is a
//! direct port, including the rank and determinant branches.
//!
//! ## The 512x512 EVA crop
//!
//! `facexlib/utils/face_restoration_helper.py:73-75` defines the FFHQ 512
//! template, and `:242-244` fits it with
//! `cv2.estimateAffinePartial2D(landmark, template, method=cv2.LMEDS)`.
//!
//! **Deliberate deviation, and the only one in this module.** LMEDS is a
//! randomized robust estimator: OpenCV draws random 2-point subsets, scores
//! each by the median squared residual, and then refines the best model by
//! least squares over its inliers. The draw is seeded from OpenCV's global
//! RNG, so a faithful port would be neither deterministic nor reproducible
//! across versions. Mold performs the refinement step alone — the
//! least-squares 4-DOF similarity fit, which is exactly [`umeyama_similarity`],
//! since Umeyama's solution *is* the LS optimum for a similarity. facexlib's
//! own comment says the reason it chose LMEDS was "the equivalence to skimage
//! transform" (`face_utils.py:167`, `face_restoration_helper.py:242`), i.e. it
//! wanted the LS similarity and reached for LMEDS to get it. With five
//! landmarks from one detector there is no outlier for the robust step to
//! reject, so the refinement is the whole answer; the fixture goldens record
//! the measured element-wise difference against real `cv2.LMEDS` output.
//!
//! ## What is deferred
//!
//! PuLID takes the 512 crop's landmarks from **facexlib's RetinaFace**
//! (`PuLID/pulid/pipeline_flux.py:145-147`, `get_face_landmarks_5(
//! only_center_face=True)`), not from SCRFD, and then masks the crop's
//! background with BiSeNet (`:161-170`). Issue #1222 scopes both out: mold
//! warps the 512 crop from the SAME SCRFD landmarks and applies no mask.
//! Issue #1225 owns closing that gap, and the fidelity gate in #1222 decides
//! whether it must. This is a named, measured divergence — never a silent one.

use super::warp::Affine2x3;

/// Five landmarks in `(x, y)` image pixels: right eye, left eye, nose, right
/// mouth corner, left mouth corner, in InsightFace's order.
pub type Landmarks5 = [[f64; 2]; 5];

/// `arcface_dst` — the 112x112 ArcFace destination template.
///
/// `insightface/python-package/insightface/utils/face_align.py:6-9`.
pub const ARCFACE_DST_112: Landmarks5 = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// facexlib's standard FFHQ five-point template at 512x512.
///
/// `facexlib/utils/face_restoration_helper.py:73-74`.
pub const FACEXLIB_FFHQ_512: Landmarks5 = [
    [192.98138, 239.94708],
    [318.90277, 240.19360],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
];

/// The ArcFace template scaled to `image_size`, per
/// `face_align.py:12-21`.
///
/// Sizes that are multiples of 112 scale directly; multiples of 128 scale by
/// `size / 128` and shift x by `8 * ratio`. Anything else is refused, matching
/// upstream's assertion.
pub fn arcface_template(image_size: u32) -> Option<Landmarks5> {
    let (ratio, diff_x) = if image_size.is_multiple_of(112) {
        (image_size as f64 / 112.0, 0.0)
    } else if image_size.is_multiple_of(128) {
        let ratio = image_size as f64 / 128.0;
        (ratio, 8.0 * ratio)
    } else {
        return None;
    };
    let mut dst = ARCFACE_DST_112;
    for point in dst.iter_mut() {
        point[0] = point[0] * ratio + diff_x;
        point[1] *= ratio;
    }
    Some(dst)
}

/// Why a similarity fit could not be produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityFitError {
    /// Every landmark coincides: there is no shape to align.
    Degenerate,
    /// The fit produced a non-finite matrix.
    NonFinite,
}

impl std::fmt::Display for SimilarityFitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Degenerate => write!(f, "the detected landmarks are degenerate (all coincident)"),
            Self::NonFinite => write!(f, "the alignment fit produced a non-finite transform"),
        }
    }
}

/// Singular value decomposition of a 2x2 matrix, `a = u * diag(s) * vt`, with
/// non-negative singular values in descending order and `u`, `vt` orthogonal.
///
/// Closed form; `nalgebra` is not a mold dependency and a 2x2 SVD does not
/// justify one.
fn svd2(a: [[f64; 2]; 2]) -> ([[f64; 2]; 2], [f64; 2], [[f64; 2]; 2]) {
    let (a00, a01, a10, a11) = (a[0][0], a[0][1], a[1][0], a[1][1]);
    let e = (a00 + a11) / 2.0;
    let f = (a00 - a11) / 2.0;
    let g = (a10 + a01) / 2.0;
    let h = (a10 - a01) / 2.0;
    let q = (e * e + h * h).sqrt();
    let r = (f * f + g * g).sqrt();
    let sx = q + r;
    let mut sy = q - r;
    // Expanding `u * diag(sx, sy) * vt` gives
    //   (a + d)/2 = (sx + sy)/2 * cos(phi - theta)
    //   (c - b)/2 = (sx + sy)/2 * sin(phi - theta)
    //   (a - d)/2 = (sx - sy)/2 * cos(phi + theta)
    //   (b + c)/2 = (sx - sy)/2 * sin(phi + theta)
    // so `a1` recovers `phi + theta` and `a2` recovers `phi - theta`.
    let a1 = g.atan2(f);
    let a2 = h.atan2(e);
    let theta = (a1 - a2) / 2.0;
    let phi = (a1 + a2) / 2.0;
    let rot = |t: f64| [[t.cos(), -t.sin()], [t.sin(), t.cos()]];
    let mut u = rot(phi);
    let vt = {
        let v = rot(theta);
        // `v` is V; the decomposition needs V^T.
        [[v[0][0], v[1][0]], [v[0][1], v[1][1]]]
    };
    // `sy` is signed by construction; numpy's SVD returns magnitudes and
    // folds the sign into U.
    if sy < 0.0 {
        sy = -sy;
        u[0][1] = -u[0][1];
        u[1][1] = -u[1][1];
    }
    // `sx = q + r` and `|sy| = |q - r|` with `q, r >= 0`, so the order is
    // descending by construction and never needs a swap.
    debug_assert!(sx + 1e-12 >= sy, "svd2 produced {sx} < {sy}");
    (u, [sx, sy], vt)
}

fn det2(m: [[f64; 2]; 2]) -> f64 {
    m[0][0] * m[1][1] - m[0][1] * m[1][0]
}

fn matmul2(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// Umeyama's optimal similarity transform with scale, in two dimensions.
///
/// Port of `skimage.transform._geometric._umeyama(src, dst,
/// estimate_scale=True)`, which is what `SimilarityTransform.estimate` calls
/// and therefore what `face_align.estimate_norm` produces. The `rank == dim`
/// path is the one real faces take; the reflected and rank-deficient branches
/// are ported for completeness so a pathological detection degrades the same
/// way upstream does rather than differently.
///
/// The result is a 2x3 matrix in OpenCV's `warpAffine` convention (source to
/// destination), matching `tform.params[0:2, :]`.
pub fn umeyama_similarity(
    src: &Landmarks5,
    dst: &Landmarks5,
) -> Result<Affine2x3, SimilarityFitError> {
    let n = src.len() as f64;
    let mean = |points: &Landmarks5| {
        let mut m = [0.0f64; 2];
        for p in points {
            m[0] += p[0];
            m[1] += p[1];
        }
        [m[0] / n, m[1] / n]
    };
    let src_mean = mean(src);
    let dst_mean = mean(dst);

    // `a = dst_demean.T @ src_demean / num`
    let mut a = [[0.0f64; 2]; 2];
    let mut src_var = [0.0f64; 2];
    for i in 0..src.len() {
        let sd = [src[i][0] - src_mean[0], src[i][1] - src_mean[1]];
        let dd = [dst[i][0] - dst_mean[0], dst[i][1] - dst_mean[1]];
        for (r, row) in a.iter_mut().enumerate() {
            for (c, cell) in row.iter_mut().enumerate() {
                *cell += dd[r] * sd[c];
            }
        }
        src_var[0] += sd[0] * sd[0];
        src_var[1] += sd[1] * sd[1];
    }
    for row in a.iter_mut() {
        for cell in row.iter_mut() {
            *cell /= n;
        }
    }
    // `src_demean.var(axis=0).sum()` — numpy's population variance.
    let src_var_sum = (src_var[0] + src_var[1]) / n;
    if src_var_sum <= 0.0 {
        return Err(SimilarityFitError::Degenerate);
    }

    let mut d = [1.0f64, 1.0];
    if det2(a) < 0.0 {
        d[1] = -1.0;
    }
    let (u, s, vt) = svd2(a);
    // numpy's `matrix_rank` tolerance: `S.max() * max(M, N) * eps`.
    let tol = s[0] * 2.0 * f64::EPSILON;
    let rank = s.iter().filter(|v| **v > tol).count();
    let rotation = match rank {
        0 => return Err(SimilarityFitError::Degenerate),
        1 => {
            // `rank == dim - 1`
            if det2(u) * det2(vt) > 0.0 {
                matmul2(u, vt)
            } else {
                let saved = d[1];
                d[1] = -1.0;
                let scaled = [[u[0][0], u[0][1] * d[1]], [u[1][0], u[1][1] * d[1]]];
                let r = matmul2(scaled, vt);
                d[1] = saved;
                r
            }
        }
        _ => {
            let scaled = [[u[0][0], u[0][1] * d[1]], [u[1][0], u[1][1] * d[1]]];
            matmul2(scaled, vt)
        }
    };

    let scale = (s[0] * d[0] + s[1] * d[1]) / src_var_sum;
    let tx = dst_mean[0] - scale * (rotation[0][0] * src_mean[0] + rotation[0][1] * src_mean[1]);
    let ty = dst_mean[1] - scale * (rotation[1][0] * src_mean[0] + rotation[1][1] * src_mean[1]);
    let m: Affine2x3 = [
        [rotation[0][0] * scale, rotation[0][1] * scale, tx],
        [rotation[1][0] * scale, rotation[1][1] * scale, ty],
    ];
    if m.iter().flatten().any(|v| !v.is_finite()) {
        return Err(SimilarityFitError::NonFinite);
    }
    Ok(m)
}

/// `face_align.estimate_norm` — the transform that lands `landmarks` on the
/// ArcFace template at `image_size`.
pub fn estimate_arcface_norm(
    landmarks: &Landmarks5,
    image_size: u32,
) -> Result<Affine2x3, SimilarityFitError> {
    let dst = arcface_template(image_size).ok_or(SimilarityFitError::Degenerate)?;
    umeyama_similarity(landmarks, &dst)
}

/// The transform that lands `landmarks` on facexlib's FFHQ 512 template.
///
/// See the module doc for the LMEDS deviation.
pub fn estimate_facexlib_512(landmarks: &Landmarks5) -> Result<Affine2x3, SimilarityFitError> {
    umeyama_similarity(landmarks, &FACEXLIB_FFHQ_512)
}

/// Root-mean-square residual, in destination pixels, between the fitted
/// landmarks and the template. Used to report alignment quality and to pin
/// parity in tests.
pub fn fit_residual_rms(m: &Affine2x3, src: &Landmarks5, dst: &Landmarks5) -> f64 {
    let mut total = 0.0;
    for i in 0..src.len() {
        let (x, y) = super::warp::apply_affine(m, src[i][0], src[i][1]);
        total += (x - dst[i][0]).powi(2) + (y - dst[i][1]).powi(2);
    }
    (total / src.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform_all(m: &Affine2x3, points: &Landmarks5) -> Landmarks5 {
        let mut out = [[0.0; 2]; 5];
        for (i, p) in points.iter().enumerate() {
            let (x, y) = super::super::warp::apply_affine(m, p[0], p[1]);
            out[i] = [x, y];
        }
        out
    }

    #[test]
    fn a_pure_similarity_is_recovered_exactly() {
        // Rotate 30 degrees, scale 2.5, translate — the fit must invert it.
        let theta: f64 = 30.0_f64.to_radians();
        let (s, c) = (theta.sin(), theta.cos());
        let scale = 2.5;
        let m: Affine2x3 = [[scale * c, -scale * s, 11.0], [scale * s, scale * c, -4.0]];
        let src = transform_all(&m, &ARCFACE_DST_112);
        // Fitting `src -> template` must undo it.
        let fitted = umeyama_similarity(&src, &ARCFACE_DST_112).unwrap();
        let round_trip = transform_all(&fitted, &src);
        for (got, want) in round_trip.iter().zip(ARCFACE_DST_112.iter()) {
            assert!((got[0] - want[0]).abs() < 1e-9, "{got:?} vs {want:?}");
            assert!((got[1] - want[1]).abs() < 1e-9, "{got:?} vs {want:?}");
        }
        assert!(fit_residual_rms(&fitted, &src, &ARCFACE_DST_112) < 1e-9);
    }

    #[test]
    fn the_identity_fit_is_the_identity() {
        let m = umeyama_similarity(&ARCFACE_DST_112, &ARCFACE_DST_112).unwrap();
        assert!((m[0][0] - 1.0).abs() < 1e-9, "{m:?}");
        assert!(m[0][1].abs() < 1e-9, "{m:?}");
        assert!(m[0][2].abs() < 1e-6, "{m:?}");
        assert!(m[1][0].abs() < 1e-9, "{m:?}");
        assert!((m[1][1] - 1.0).abs() < 1e-9, "{m:?}");
        assert!(m[1][2].abs() < 1e-6, "{m:?}");
    }

    /// A similarity fit has four degrees of freedom, so it must never mirror
    /// a face to reduce residual — `d[1] = -1` when `det(A) < 0` is what stops
    /// it, and a mirrored input must therefore keep a positive determinant.
    #[test]
    fn a_mirrored_input_is_not_fitted_by_a_reflection() {
        let mut mirrored = ARCFACE_DST_112;
        for p in mirrored.iter_mut() {
            p[0] = 112.0 - p[0];
        }
        let m = umeyama_similarity(&mirrored, &ARCFACE_DST_112).unwrap();
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        assert!(det > 0.0, "the fit reflected: det = {det}");
    }

    #[test]
    fn coincident_landmarks_are_a_typed_error() {
        let degenerate: Landmarks5 = [[5.0, 5.0]; 5];
        assert_eq!(
            umeyama_similarity(&degenerate, &ARCFACE_DST_112),
            Err(SimilarityFitError::Degenerate)
        );
    }

    #[test]
    fn collinear_landmarks_still_produce_a_finite_transform() {
        let collinear: Landmarks5 = [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
            [40.0, 0.0],
        ];
        let m = umeyama_similarity(&collinear, &ARCFACE_DST_112).unwrap();
        assert!(m.iter().flatten().all(|v| v.is_finite()), "{m:?}");
    }

    #[test]
    fn the_112_template_is_upstreams_verbatim() {
        assert_eq!(arcface_template(112).unwrap(), ARCFACE_DST_112);
    }

    #[test]
    fn the_224_template_doubles_the_112_one() {
        let dst = arcface_template(224).unwrap();
        for (got, want) in dst.iter().zip(ARCFACE_DST_112.iter()) {
            assert!((got[0] - want[0] * 2.0).abs() < 1e-12);
            assert!((got[1] - want[1] * 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn the_128_template_shifts_x_by_eight() {
        // 128 % 112 != 0, so the `% 128` branch applies: ratio 1.0, diff_x 8.0.
        let dst = arcface_template(128).unwrap();
        for (got, want) in dst.iter().zip(ARCFACE_DST_112.iter()) {
            assert!((got[0] - (want[0] + 8.0)).abs() < 1e-9, "{dst:?}");
            assert!((got[1] - want[1]).abs() < 1e-9, "{dst:?}");
        }
    }

    #[test]
    fn the_two_by_two_svd_reconstructs_its_input() {
        for a in [
            [[3.0, 1.0], [0.5, -2.0]],
            [[0.0, 4.0], [-4.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.5, 2.25], [7.0, 0.125]],
            [[1e-9, 0.0], [0.0, 5.0]],
        ] {
            let (u, s, vt) = svd2(a);
            assert!(
                s[0] >= s[1] && s[1] >= 0.0,
                "singular values {s:?} for {a:?}"
            );
            assert!((det2(u).abs() - 1.0).abs() < 1e-9, "u not orthogonal {u:?}");
            assert!(
                (det2(vt).abs() - 1.0).abs() < 1e-9,
                "vt not orthogonal {vt:?}"
            );
            let recon = matmul2(matmul2(u, [[s[0], 0.0], [0.0, s[1]]]), vt);
            for r in 0..2 {
                for c in 0..2 {
                    assert!(
                        (recon[r][c] - a[r][c]).abs() < 1e-9,
                        "reconstruction {recon:?} != {a:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn an_unsupported_crop_size_is_refused_like_upstreams_assertion() {
        assert!(arcface_template(100).is_none());
        assert!(arcface_template(113).is_none());
    }

    #[test]
    fn the_512_template_is_facexlibs_verbatim() {
        assert_eq!(FACEXLIB_FFHQ_512[0], [192.98138, 239.94708]);
        assert_eq!(FACEXLIB_FFHQ_512[4], [313.08905, 371.15118]);
        // Nothing in the FFHQ crop sits outside the frame.
        for p in FACEXLIB_FFHQ_512 {
            assert!(p[0] > 0.0 && p[0] < 512.0);
            assert!(p[1] > 0.0 && p[1] < 512.0);
        }
    }

    /// The two crops come from ONE detection, so the 512 fit must be a scaled
    /// sibling of the 112 fit rather than a second, independent alignment.
    #[test]
    fn both_templates_fit_the_same_landmarks() {
        let landmarks: Landmarks5 = [
            [120.5, 180.25],
            [190.75, 179.0],
            [155.0, 220.5],
            [128.0, 262.0],
            [186.5, 261.25],
        ];
        let m112 = estimate_arcface_norm(&landmarks, 112).unwrap();
        let m512 = estimate_facexlib_512(&landmarks).unwrap();
        assert!(fit_residual_rms(&m112, &landmarks, &ARCFACE_DST_112) < 5.0);
        assert!(fit_residual_rms(&m512, &landmarks, &FACEXLIB_FFHQ_512) < 20.0);
        // The 512 template's inter-eye span is ~3.57x the 112 one's, so the
        // fitted scales must differ by roughly that.
        let scale112 = (m112[0][0].powi(2) + m112[0][1].powi(2)).sqrt();
        let scale512 = (m512[0][0].powi(2) + m512[0][1].powi(2)).sqrt();
        let ratio = scale512 / scale112;
        assert!(
            (3.0..4.2).contains(&ratio),
            "unexpected scale ratio {ratio}"
        );
    }
}
