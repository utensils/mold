//! SCRFD face detection.
//!
//! Ported from `insightface/python-package/insightface/model_zoo/scrfd.py`.
//! The ONNX graph emits raw per-anchor scores, distance-encoded boxes, and
//! distance-encoded keypoints; **anchor decoding, thresholding, and NMS all
//! happen outside it** (`scrfd.py:158-225` and `:352-380`), which is why this
//! module exists at all.
//!
//! `scrfd_10g_bnkps.onnx` reports nine outputs, so upstream's `_init_vars`
//! (`scrfd.py:120-137`) selects `fmc = 3`, strides `[8, 16, 32]`,
//! `_num_anchors = 2`, `use_kps = True`. Its input is `[1, 3, ?, ?]`, so
//! `static_input_size` is `None` and the detector runs at the 640x640 default
//! (`scrfd.py:97`, `DEFAULT_DET_SIZES[-1]`).

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use candle_onnx::onnx::ModelProto;
use image::RgbImage;

use super::align::Landmarks5;
use super::scrfd_net::ScrfdNet;
use super::warp::letterbox_top_left;

/// Feature-pyramid strides, `scrfd.py:122`.
pub const FEATURE_STRIDES: [usize; 3] = [8, 16, 32];
/// Anchors per feature-map cell, `scrfd.py:123`.
pub const ANCHORS_PER_CELL: usize = 2;
/// `SCRFD.det_thresh`, `scrfd.py:86`.
pub const DEFAULT_SCORE_THRESHOLD: f32 = 0.5;
/// `SCRFD.nms_thresh`, `scrfd.py:85`.
pub const DEFAULT_NMS_THRESHOLD: f32 = 0.4;
/// The detector input mold pins, `scrfd.py:17` `DEFAULT_DET_SIZES[-1]`.
pub const DETECTOR_INPUT: u32 = 640;
/// `SCRFD.input_mean`, `scrfd.py:110`.
const INPUT_MEAN: f32 = 127.5;
/// `SCRFD.input_std`, `scrfd.py:111`.
const INPUT_STD: f32 = 128.0;

/// One detected face in SOURCE image pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectedFace {
    /// `[x1, y1, x2, y2]`.
    pub bbox: [f32; 4],
    /// Detection confidence.
    pub score: f32,
    /// Five keypoints in InsightFace order.
    pub landmarks: Landmarks5,
}

impl DetectedFace {
    /// Bounding-box area, which is how PuLID picks the face it conditions on
    /// (`PuLID/pulid/pipeline_flux.py:127-129`).
    pub fn area(&self) -> f32 {
        (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    }
}

/// `distance2bbox`, `scrfd.py:29-49`.
fn distance_to_bbox(cx: f32, cy: f32, d: [f32; 4]) -> [f32; 4] {
    [cx - d[0], cy - d[1], cx + d[2], cy + d[3]]
}

/// Anchor centres for one stride, `scrfd.py:184-208`.
///
/// `np.mgrid[:height, :width][::-1]` stacked on the last axis yields `(x, y)`
/// in row-major order; each centre is then repeated `ANCHORS_PER_CELL` times
/// **consecutively** (`np.stack([c] * n, axis=1).reshape(-1, 2)`), which is
/// what interleaves the two anchors of a cell rather than concatenating two
/// full grids.
pub fn anchor_centers(height: usize, width: usize, stride: usize, anchors: usize) -> Vec<[f32; 2]> {
    let mut out = Vec::with_capacity(height * width * anchors);
    for y in 0..height {
        for x in 0..width {
            let centre = [(x * stride) as f32, (y * stride) as f32];
            for _ in 0..anchors {
                out.push(centre);
            }
        }
    }
    out
}

/// Upstream's NMS, `scrfd.py:352-380`.
///
/// Deliberately local rather than
/// `candle_transformers::object_detection::non_maximum_suppression`: that
/// helper computes plain areas, while InsightFace's has the legacy `+1` in
/// both extents (`areas = (x2 - x1 + 1) * (y2 - y1 + 1)`). At a 40-pixel face
/// the `+1` moves the IoU by ~5%, which is enough to keep or drop a
/// neighbouring box. Input must already be sorted by descending score, as
/// `detect` guarantees.
pub fn non_max_suppression(boxes: &[[f32; 4]], threshold: f32) -> Vec<usize> {
    let areas: Vec<f32> = boxes
        .iter()
        .map(|b| (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0))
        .collect();
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    let mut keep = Vec::new();
    while !order.is_empty() {
        let i = order[0];
        keep.push(i);
        order = order[1..]
            .iter()
            .copied()
            .filter(|&j| {
                let xx1 = boxes[i][0].max(boxes[j][0]);
                let yy1 = boxes[i][1].max(boxes[j][1]);
                let xx2 = boxes[i][2].min(boxes[j][2]);
                let yy2 = boxes[i][3].min(boxes[j][3]);
                let w = (xx2 - xx1 + 1.0).max(0.0);
                let h = (yy2 - yy1 + 1.0).max(0.0);
                let inter = w * h;
                let ovr = inter / (areas[i] + areas[j] - inter);
                ovr <= threshold
            })
            .collect();
    }
    keep
}

/// The SCRFD detector: the resident network plus upstream's thresholds.
///
/// The `ModelProto` is consumed at construction and dropped — see
/// [`super::scrfd_net`] and `docs/architecture/pulid-perf.md` §1. Nothing here
/// touches `candle-onnx` after `new` returns.
pub struct ScrfdDetector {
    net: ScrfdNet,
    /// Score threshold; defaults to [`DEFAULT_SCORE_THRESHOLD`].
    pub score_threshold: f32,
    /// NMS IoU threshold; defaults to [`DEFAULT_NMS_THRESHOLD`].
    pub nms_threshold: f32,
}

impl ScrfdDetector {
    /// Wrap a decoded `scrfd_10g_bnkps` graph, on the CPU.
    pub fn new(model: ModelProto) -> Result<Self> {
        Self::new_on_device(model, &Device::Cpu)
    }

    /// Wrap a decoded `scrfd_10g_bnkps` graph, placing its weights on `device`.
    ///
    /// Fails closed on any graph whose output arity is not the nine-tensor
    /// keypoint variant, because every stride offset below is derived from
    /// `fmc = 3`.
    pub fn new_on_device(model: ModelProto, device: &Device) -> Result<Self> {
        let graph = model
            .graph
            .as_ref()
            .context("the SCRFD model carries no graph")?;
        let outputs = graph.output.len();
        if outputs != FEATURE_STRIDES.len() * 3 {
            bail!(
                "expected a 9-output SCRFD keypoint graph (scrfd.py:127-132), got {outputs} outputs"
            );
        }
        Ok(Self {
            net: ScrfdNet::new(&model, device).context("building the SCRFD network")?,
            score_threshold: DEFAULT_SCORE_THRESHOLD,
            nms_threshold: DEFAULT_NMS_THRESHOLD,
        })
    }

    /// The device the detector's weights are resident on.
    pub fn device(&self) -> &Device {
        self.net.device()
    }

    /// Build the network blob from an already letterboxed RGB canvas.
    ///
    /// `cv2.dnn.blobFromImage(img, 1/128.0, size, (127.5, 127.5, 127.5),
    /// swapRB=True)` (`scrfd.py:164`). Upstream's `img` is BGR, so `swapRB`
    /// means the graph is fed **RGB**; mold already holds RGB, so there is no
    /// swap here — adding one would feed the network mirrored channels.
    pub fn blob(image: &RgbImage) -> Result<Tensor> {
        let (w, h) = (image.width() as usize, image.height() as usize);
        let mut data = vec![0f32; 3 * h * w];
        for (x, y, px) in image.enumerate_pixels() {
            let (x, y) = (x as usize, y as usize);
            for c in 0..3 {
                data[c * h * w + y * w + x] = (px.0[c] as f32 - INPUT_MEAN) / INPUT_STD;
            }
        }
        Tensor::from_vec(data, (1, 3, h, w), &Device::Cpu).map_err(Into::into)
    }

    /// Detect every face at or above the score threshold, in source pixels,
    /// sorted by descending score.
    pub fn detect(&self, image: &RgbImage) -> Result<Vec<DetectedFace>> {
        let boxed = letterbox_top_left(image, DETECTOR_INPUT);
        let blob = Self::blob(&boxed.image)?;
        let raw = self.net.forward(&blob).context("SCRFD evaluation failed")?;

        let mut candidates: Vec<DetectedFace> = Vec::new();
        for (idx, stride) in FEATURE_STRIDES.iter().copied().enumerate() {
            let scores = &raw.scores[idx];
            let bbox_preds = &raw.bboxes[idx];
            let kps_preds = &raw.keypoints[idx];
            let extent = DETECTOR_INPUT as usize / stride;
            let centres = anchor_centers(extent, extent, stride, ANCHORS_PER_CELL);
            if scores.len() != centres.len() {
                bail!(
                    "SCRFD stride {stride} produced {} scores for {} anchors",
                    scores.len(),
                    centres.len()
                );
            }
            for (anchor, centre) in centres.iter().enumerate() {
                let score = scores[anchor];
                if score < self.score_threshold {
                    continue;
                }
                let d = [
                    bbox_preds[anchor * 4] * stride as f32,
                    bbox_preds[anchor * 4 + 1] * stride as f32,
                    bbox_preds[anchor * 4 + 2] * stride as f32,
                    bbox_preds[anchor * 4 + 3] * stride as f32,
                ];
                let bbox = distance_to_bbox(centre[0], centre[1], d);
                // `distance2kps`, `scrfd.py:51-70`: alternating x/y offsets
                // from the SAME anchor centre.
                let mut landmarks: Landmarks5 = [[0.0; 2]; 5];
                for (point, slot) in landmarks.iter_mut().enumerate() {
                    let px = centre[0] + kps_preds[anchor * 10 + point * 2] * stride as f32;
                    let py = centre[1] + kps_preds[anchor * 10 + point * 2 + 1] * stride as f32;
                    *slot = [px as f64 / boxed.det_scale, py as f64 / boxed.det_scale];
                }
                let scale = boxed.det_scale as f32;
                candidates.push(DetectedFace {
                    bbox: [
                        bbox[0] / scale,
                        bbox[1] / scale,
                        bbox[2] / scale,
                        bbox[3] / scale,
                    ],
                    score,
                    landmarks,
                });
            }
        }
        // `scrfd.py:238-241`: descending score before NMS.
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let boxes: Vec<[f32; 4]> = candidates.iter().map(|f| f.bbox).collect();
        let keep = non_max_suppression(&boxes, self.nms_threshold);
        Ok(keep.into_iter().map(|i| candidates[i]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_centres_interleave_the_two_anchors_of_a_cell() {
        let centres = anchor_centers(2, 3, 8, 2);
        assert_eq!(centres.len(), 12);
        // Row-major (x fastest), each centre repeated twice consecutively.
        assert_eq!(centres[0], [0.0, 0.0]);
        assert_eq!(centres[1], [0.0, 0.0]);
        assert_eq!(centres[2], [8.0, 0.0]);
        assert_eq!(centres[3], [8.0, 0.0]);
        assert_eq!(centres[4], [16.0, 0.0]);
        assert_eq!(centres[6], [0.0, 8.0]);
        assert_eq!(centres[11], [16.0, 8.0]);
    }

    #[test]
    fn every_stride_produces_the_anchor_count_the_graph_declares() {
        // The pinned graph's output rows: 12800, 3200, 800.
        let expected = [12800, 3200, 800];
        for (stride, want) in FEATURE_STRIDES.iter().zip(expected) {
            let extent = DETECTOR_INPUT as usize / stride;
            assert_eq!(
                anchor_centers(extent, extent, *stride, ANCHORS_PER_CELL).len(),
                want,
                "stride {stride}"
            );
        }
    }

    #[test]
    fn distance_decoding_expands_around_the_anchor() {
        let bbox = distance_to_bbox(100.0, 50.0, [10.0, 5.0, 20.0, 15.0]);
        assert_eq!(bbox, [90.0, 45.0, 120.0, 65.0]);
    }

    #[test]
    fn nms_keeps_the_highest_scoring_of_an_overlapping_pair() {
        // Identical boxes: IoU 1.0 > 0.4, so only the first survives.
        let boxes = [[0.0, 0.0, 100.0, 100.0], [1.0, 1.0, 101.0, 101.0]];
        assert_eq!(non_max_suppression(&boxes, DEFAULT_NMS_THRESHOLD), vec![0]);
    }

    #[test]
    fn nms_keeps_disjoint_boxes() {
        let boxes = [[0.0, 0.0, 50.0, 50.0], [200.0, 200.0, 250.0, 250.0]];
        assert_eq!(
            non_max_suppression(&boxes, DEFAULT_NMS_THRESHOLD),
            vec![0, 1]
        );
    }

    /// The legacy `+1` is why this is not
    /// `candle_transformers::object_detection::non_maximum_suppression`.
    #[test]
    fn nms_uses_insightfaces_inclusive_extents() {
        // Two boxes spanning pixels 0..=9 and 4..=13, overlapping on 4..=9.
        //   plain extents: 25 / (81 + 81 - 25)    = 0.1825
        //   +1 extents:    36 / (100 + 100 - 36)  = 0.2195
        let boxes = [[0.0, 0.0, 9.0, 9.0], [4.0, 4.0, 13.0, 13.0]];
        // A threshold between the two answers keeps or drops the second box
        // depending on which convention is in force. Under the plain
        // convention both of these would return `[0, 1]`.
        assert_eq!(non_max_suppression(&boxes, 0.19), vec![0]);
        assert_eq!(non_max_suppression(&boxes, 0.23), vec![0, 1]);
    }

    #[test]
    fn nms_on_an_empty_set_is_empty() {
        assert!(non_max_suppression(&[], 0.4).is_empty());
    }

    #[test]
    fn the_blob_matches_upstreams_mean_and_std() {
        let image = RgbImage::from_pixel(2, 2, image::Rgb([255, 0, 127]));
        let blob = ScrfdDetector::blob(&image).unwrap();
        assert_eq!(blob.dims(), &[1, 3, 2, 2]);
        let values = blob.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Channel-planar: R plane first, then G, then B. No BGR swap.
        assert!((values[0] - (255.0 - 127.5) / 128.0).abs() < 1e-6);
        assert!((values[4] - (0.0 - 127.5) / 128.0).abs() < 1e-6);
        assert!((values[8] - (127.0 - 127.5) / 128.0).abs() < 1e-6);
    }

    #[test]
    fn area_is_the_bbox_extent_pulid_sorts_on() {
        let face = DetectedFace {
            bbox: [10.0, 20.0, 40.0, 60.0],
            score: 0.9,
            landmarks: [[0.0; 2]; 5],
        };
        assert!((face.area() - 1200.0).abs() < 1e-3);
    }
}
