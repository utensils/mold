//! Real-checkpoint parity against the M1 upstream captures
//! (`testdata/qwen_image21/README.md`). Every test is `#[ignore]` and needs:
//!
//! - `QWEN_IMAGE21_MODEL_ROOT` — a mold models dir holding `shared/qwen-image21`
//!   and `qwen-image-2.1-bf16/transformer`;
//! - `QWEN_IMAGE21_FIXTURES` — the large-capture directory.
//!
//! GPU tests run on CUDA device 0 when built with `cuda` (select the card
//! with `CUDA_VISIBLE_DEVICES`), otherwise on the CPU.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, IndexOp, Tensor};

use super::layout::QwenImage21JointLayout;
use super::reference::{
    encode_prompt_with_images, encode_vision, load_vision_tower, pack_vision_inputs,
    prepare_reference, tokenize_image_conditioned, PreparedReference,
};
use super::transformer::QwenImage21Transformer;
use super::{encode_t2i_prompts, PrefixCacheDecision, QwenImage21TextConditioning};
use crate::encoders::qwen3::Qwen3Encoder;
use crate::encoders::qwen3_bf16::Qwen3BF16Config;
use crate::progress::ProgressReporter;

const P6_PROMPT: &str =
    "Put the red apple from image 2 on the white sign in image 1, keep everything else unchanged.";
const P6_NEGATIVE: &str = "blurry, lowres, watermark";
const P8_PROMPT: &str =
    "Change the sky to a warm sunset with orange clouds, keep the house and the sign unchanged.";

pub(super) struct Env {
    pub models: PathBuf,
    pub fixtures: PathBuf,
}

/// The two directories every parity test needs. A test that is run
/// (`--ignored`) without them PANICS naming the missing variable: returning
/// early would report a parity gate as passed that never ran.
pub(super) fn env() -> Env {
    let dir = |name: &str| {
        PathBuf::from(
            std::env::var_os(name)
                .unwrap_or_else(|| panic!("{name} must be set to run this parity test")),
        )
    };
    Env {
        models: dir("QWEN_IMAGE21_MODEL_ROOT"),
        fixtures: dir("QWEN_IMAGE21_FIXTURES"),
    }
}

impl Env {
    pub fn tokenizer(&self) -> PathBuf {
        self.models
            .join("shared/qwen-image21/processor/tokenizer.json")
    }

    pub fn text_encoder(&self) -> Vec<PathBuf> {
        (1..=4)
            .map(|index| {
                self.models.join(format!(
                    "shared/qwen-image21/text_encoder/model-0000{index}-of-00004.safetensors"
                ))
            })
            .collect()
    }

    pub fn transformer(&self) -> Vec<PathBuf> {
        (1..=2)
            .map(|index| {
                self.models.join(format!(
                    "qwen-image-2.1-bf16/transformer/diffusion_pytorch_model-0000{index}-of-00002.safetensors"
                ))
            })
            .collect()
    }

    pub fn vae(&self) -> PathBuf {
        self.models
            .join("shared/qwen-image21/vae/diffusion_pytorch_model.safetensors")
    }

    pub fn capture(&self, name: &str) -> HashMap<String, Tensor> {
        load_capture(&self.fixtures.join(name))
    }
}

pub(super) fn device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(device) = Device::new_cuda(0) {
        return device;
    }
    Device::Cpu
}

/// Load a capture on the CPU. torch writes masks as safetensors `BOOL`,
/// which candle does not read, so those become `U8`.
pub(super) fn load_capture(path: &Path) -> HashMap<String, Tensor> {
    use safetensors::tensor::Dtype;
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let file = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    file.tensors()
        .into_iter()
        .map(|(name, view)| {
            let dtype = match view.dtype() {
                Dtype::BOOL | Dtype::U8 => DType::U8,
                Dtype::F32 => DType::F32,
                Dtype::BF16 => DType::BF16,
                Dtype::F16 => DType::F16,
                Dtype::I64 => DType::I64,
                Dtype::U32 => DType::U32,
                other => panic!("{name}: unsupported capture dtype {other:?}"),
            };
            let tensor =
                Tensor::from_raw_buffer(view.data(), dtype, view.shape(), &Device::Cpu).unwrap();
            (name, tensor)
        })
        .collect()
}

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata/qwen_image21")
        .join(name)
}

pub(super) fn references(names: &[&str]) -> Vec<PreparedReference> {
    names
        .iter()
        .map(|name| prepare_reference(&std::fs::read(testdata(name)).unwrap()).unwrap())
        .collect()
}

/// `max|a - b| / max|b|` and `mean|a - b| / mean|b|`, in F32.
pub(super) fn relative_error(actual: &Tensor, expected: &Tensor) -> (f32, f32) {
    let actual = actual
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap();
    let expected = expected
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap();
    assert_eq!(actual.dims(), expected.dims());
    let diff = (&actual - &expected).unwrap().abs().unwrap();
    let scalar = |t: Tensor| t.to_scalar::<f32>().unwrap();
    let max_error = scalar(diff.max(0).unwrap()) / scalar(expected.abs().unwrap().max(0).unwrap());
    let mean_error =
        scalar(diff.mean_all().unwrap()) / scalar(expected.abs().unwrap().mean_all().unwrap());
    (max_error, mean_error)
}

/// bf16 parity: mold's mean error against the fp32 capture may be at most
/// `ratio` times upstream's own bf16 capture's.
fn check_against_truth(
    label: &str,
    actual: &Tensor,
    upstream: &Tensor,
    truth: &Tensor,
    ratio: f32,
) {
    let (_, ours) = relative_error(actual, truth);
    let (_, theirs) = relative_error(upstream, truth);
    eprintln!("{label} vs fp32 truth: mold mean {ours:.3e}, upstream mean {theirs:.3e}");
    assert!(ours <= theirs * ratio, "{label}: {ours} vs {theirs}");
}

fn check(label: &str, actual: &Tensor, expected: &Tensor, max_tolerance: f32) {
    let (max_error, mean_error) = relative_error(actual, expected);
    eprintln!("{label}: relative max {max_error:.3e}, mean {mean_error:.3e}");
    assert!(max_error <= max_tolerance, "{label}: {max_error}");
}

fn bools(tensor: &Tensor) -> Vec<bool> {
    tensor
        .flatten_all()
        .unwrap()
        .to_dtype(DType::U8)
        .unwrap()
        .to_vec1::<u8>()
        .unwrap()
        .into_iter()
        .map(|value| value != 0)
        .collect()
}

/// P1: token ids, grid and `pixel_values` for the two captured references,
/// exactly. CPU.
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn p1_processor_matches_the_upstream_capture() {
    let env = env();
    let refs = references(&["ref_opaque.png", "ref_rgba.png"]);
    let tokenizer = tokenizers::Tokenizer::from_file(env.tokenizer()).unwrap();
    let counts: Vec<usize> = refs.iter().map(|r| r.pad_count().unwrap()).collect();
    let ids = tokenize_image_conditioned(&tokenizer, P6_PROMPT, &counts).unwrap();
    let captured =
        candle_core::safetensors::load(testdata("p1_processor_ids.safetensors"), &Device::Cpu)
            .unwrap();
    let expected: Vec<u32> = captured["input_ids"]
        .flatten_all()
        .unwrap()
        .to_vec1::<i64>()
        .unwrap()
        .into_iter()
        .map(|id| id as u32)
        .collect();
    assert_eq!(ids, expected);
    let (pixels, grid, _) = pack_vision_inputs(&refs, &Device::Cpu).unwrap();
    let pixel_capture = env.capture("p1_processor_pixels.safetensors");
    assert_eq!(
        grid.to_dtype(DType::I64).unwrap().to_vec2::<i64>().unwrap(),
        pixel_capture["image_grid_thw"].to_vec2::<i64>().unwrap()
    );
    let actual = pixels.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let expected = pixel_capture["pixel_values"]
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mismatched = actual
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(mismatched, 0, "{mismatched} pixel values differ");
}

/// P2: the vision merger and three DeepStack maps, fp32.
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn p2_vision_tower_matches_the_upstream_capture() {
    let env = env();
    let device = device();
    let refs = references(&["ref_opaque.png", "ref_rgba.png"]);
    let tower = load_vision_tower(
        &env.text_encoder(),
        &device,
        DType::F32,
        &ProgressReporter::default(),
    )
    .unwrap();
    let features = encode_vision(&tower, &refs, &device, &mut || Ok(())).unwrap();
    let captured = env.capture("p2_vision_fp32.safetensors");
    check("merger", &features.embeds, &captured["vision_merger"], 3e-4);
    for (index, map) in features.deepstack.iter().enumerate() {
        check(
            &format!("deepstack {index}"),
            map,
            &captured[&format!("vision_deepstack_{index}")],
            1e-3,
        );
    }
}

fn p3_cases() -> Vec<(&'static str, &'static str, Vec<&'static str>)> {
    vec![
        ("p6_pos", P6_PROMPT, vec!["ref_opaque.png", "ref_rgba.png"]),
        (
            "p6_neg",
            P6_NEGATIVE,
            vec!["ref_opaque.png", "ref_rgba.png"],
        ),
        ("p8_pos", P8_PROMPT, vec!["ref_opaque.png"]),
        ("t2i_p8", P8_PROMPT, vec![]),
    ]
}

/// P3 bf16's per-case ceilings on the mean relative error of mold's
/// `prompt_embeds` against the FP32 capture. Measured on an L40S with the
/// shipped F32 vision tower (BF16 language model): p6_pos 5.18e-2, p6_neg
/// 5.28e-2, p8_pos 4.36e-2, t2i_p8 2.86e-2 — against upstream's bf16 run's
/// 2.99e-1, 3.01e-1, 2.85e-1 and 2.45e-2. Each ceiling is 1.25x mold's
/// measurement, so a defect adding a few percent fails where the old gate
/// (1.5x UPSTREAM's error) allowed the image cases to drift to 45%: upstream
/// ran its vision tower in bf16, which alone moves its conditioning ~30%.
fn p3_bf16_ceiling(case: &str) -> f32 {
    match case {
        "p6_pos" | "p6_neg" => 6.6e-2,
        "p8_pos" => 5.5e-2,
        "t2i_p8" => 3.6e-2,
        other => panic!("no P3 bf16 ceiling for {other}"),
    }
}

fn p3(dtype: DType, suffix: &str, tolerance: f32) {
    let env = env();
    let device = device();
    let progress = ProgressReporter::default();
    let mut encoder = Qwen3Encoder::load_bf16(
        &env.text_encoder(),
        &env.tokenizer(),
        &device,
        dtype,
        &Qwen3BF16Config::qwen3_image_21_text_encoder(),
        &progress,
    )
    .unwrap();
    let tower = load_vision_tower(
        &env.text_encoder(),
        &device,
        super::reference::vision_tower_dtype(),
        &progress,
    )
    .unwrap();
    for (case, prompt, files) in p3_cases() {
        let conditioning: QwenImage21TextConditioning = if files.is_empty() {
            encode_t2i_prompts(&mut encoder, &[prompt.to_string()]).unwrap()
        } else {
            let refs = references(&files);
            let vision = encode_vision(&tower, &refs, &device, &mut || Ok(())).unwrap();
            encode_prompt_with_images(&mut encoder, &vision, prompt).unwrap()
        };
        let captured = env.capture(&format!("p3_{case}_{suffix}.safetensors"));
        assert_eq!(
            conditioning.image_slots[0],
            bools(&captured["image_pad_mask"]),
            "{case} image pad mask"
        );
        if dtype == DType::F32 {
            check(
                &format!("{case} prompt_embeds {suffix}"),
                &conditioning.embeddings,
                &captured["prompt_embeds"],
                tolerance,
            );
        } else {
            // The bf16 vision tower alone moves upstream's own merger output
            // ~14% from its fp32 run, so a bf16-to-bf16 comparison measures
            // rounding chaos. Hold mold's bf16 to the fp32 truth under an
            // absolute per-case ceiling, and never worse than `tolerance`
            // times upstream's own bf16 error.
            let truth =
                env.capture(&format!("p3_{case}_fp32.safetensors"))["prompt_embeds"].clone();
            let (_, ours) = relative_error(&conditioning.embeddings, &truth);
            let (_, theirs) = relative_error(&captured["prompt_embeds"], &truth);
            eprintln!(
                "{case} bf16 vs fp32 truth: mold mean {ours:.3e}, upstream mean {theirs:.3e}"
            );
            assert!(ours <= theirs * tolerance, "{case}: {ours} vs {theirs}");
            let ceiling = p3_bf16_ceiling(case);
            assert!(
                ours <= ceiling,
                "{case}: mean {ours:.3e} from the fp32 truth exceeds {ceiling:.1e}"
            );
        }
    }
}

/// P3 fp32: trimmed pre-norm hidden states and `image_pad_mask` for the
/// positive and negative P6 prompts, the P8 edit prompt, and text-to-image.
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn p3_fp32_conditioning_matches_the_upstream_capture() {
    p3(DType::F32, "fp32", 5e-3);
}

/// P3 bf16: the same through the bf16 checkpoint.
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn p3_bf16_conditioning_matches_the_upstream_capture() {
    // `tolerance` is the allowed ratio to upstream's own bf16 error; the
    // absolute ceilings are `p3_bf16_ceiling`'s.
    p3(DType::BF16, "bf16", 1.25);
}

fn p6(dtype: DType, suffix: &str, tolerance: f32) {
    let env = env();
    let device = device();
    let inputs = env.capture("p6_inputs.safetensors");
    let outputs = env.capture(&format!("p6_outputs_{suffix}.safetensors"));
    let slots =
        bools(&load_capture(&testdata("p3_p6_pos_image_pad_mask.safetensors"))["image_pad_mask"]);
    let embeddings = inputs["prompt_embeds"]
        .to_device(&device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let text_len = slots.len();
    let conditioning = QwenImage21TextConditioning {
        embeddings,
        valid_tokens: vec![vec![true; text_len]],
        image_slots: vec![slots.clone()],
    };
    let layout = QwenImage21JointLayout::build(
        &slots,
        &conditioning.valid_tokens,
        &[(52, 78), (72, 58)],
        (32, 32),
    )
    .unwrap();
    assert_eq!(layout.total_len(), 9298);
    let cond = inputs["cond_latents"]
        .to_device(&device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let transformer = QwenImage21Transformer::load(
        &env.transformer(),
        &device,
        dtype,
        &ProgressReporter::default(),
    )
    .unwrap();
    let timestep = |name: &str| -> f64 {
        f64::from(
            outputs[name]
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0],
        )
    };
    let (t_a, t_b) = (timestep("timestep_a"), timestep("timestep_b"));
    eprintln!("timesteps {t_a} {t_b}");
    let latent = |name: &str| {
        inputs[name]
            .to_device(&device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    };
    let truth = env.capture("p6_outputs_fp32.safetensors");
    let target_of = |set: &HashMap<String, Tensor>, name: &str| {
        if name == "cached_b" {
            set[name].clone()
        } else {
            set[name].narrow(1, 9298 - 1024, 1024).unwrap()
        }
    };
    let check = |name: &str, actual: &Tensor| {
        let label = format!("{name} {suffix}");
        if dtype == DType::F32 {
            check(&label, actual, &target_of(&outputs, name), tolerance);
        } else {
            check_against_truth(
                &label,
                actual,
                &target_of(&outputs, name),
                &target_of(&truth, name),
                tolerance,
            );
        }
    };

    let mut full = transformer
        .prepare(
            &conditioning,
            layout.clone(),
            Some(cond.clone()),
            PrefixCacheDecision::Recompute,
        )
        .unwrap();
    check("full_a", &full.forward(&latent("x_a"), t_a).unwrap());
    check("full_b", &full.forward(&latent("x_b"), t_b).unwrap());
    drop(full);
    let mut cached = transformer
        .prepare(
            &conditioning,
            layout,
            Some(cond),
            PrefixCacheDecision::Retain,
        )
        .unwrap();
    check("extract_a", &cached.forward(&latent("x_a"), t_a).unwrap());
    check("cached_b", &cached.forward(&latent("x_b"), t_b).unwrap());
}

/// P6 fp32: one two-reference transformer forward — full, prefix extract,
/// then cached — against upstream.
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn p6_fp32_transformer_matches_the_upstream_capture() {
    p6(DType::F32, "fp32", 1e-4);
}

/// P6 bf16.
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn p6_bf16_transformer_matches_the_upstream_capture() {
    // `tolerance` is the allowed ratio to upstream's own bf16 error.
    p6(DType::BF16, "bf16", 1.5);
}
/// The language-model hidden states `p3_p8_pos_lm_internals_*` captures:
/// the scatter (`lm_hidden_0`), layers 0-3, 17, 34 (`1`-`4`, `18`, `35`) and
/// the last layer before the final norm (`36`).
const LM_CAPTURED_LAYERS: [usize; 8] = [0, 1, 2, 3, 4, 18, 35, 36];

/// Where the P8 reference's vision features come from for a language-model
/// comparison.
#[derive(Debug, Clone, Copy)]
enum LmVision {
    /// mold's own tower at this dtype.
    Tower(DType),
    /// P2's captured fp32 merger and DeepStack rows, so the comparison sees
    /// the language model and nothing upstream of it.
    CapturedFp32,
}

/// Run the P8 edit prompt (one reference) through mold's multimodal Qwen3
/// language model at `lm_dtype` and return the captured layers'
/// `(max, mean)` relative errors against `p3_p8_pos_lm_internals_{suffix}`,
/// in [`LM_CAPTURED_LAYERS`] order. MRoPE position ids must match exactly.
fn lm_internal_errors(lm_dtype: DType, vision: LmVision, suffix: &str) -> Vec<(f32, f32)> {
    use crate::encoders::qwen3::Qwen3Model;
    use crate::encoders::qwen3_vl_inject::VisualInjection;
    use mold_candle::qwen3_vl::{create_mm_token_type_ids, qwen_mrope_positions};
    let env = env();
    let device = device();
    let progress = ProgressReporter::default();
    let refs = references(&["ref_opaque.png"]);
    let tower_dtype = match vision {
        LmVision::Tower(dtype) => dtype,
        LmVision::CapturedFp32 => DType::F32,
    };
    let tower = load_vision_tower(&env.text_encoder(), &device, tower_dtype, &progress).unwrap();
    let mut features = encode_vision(&tower, &refs, &device, &mut || Ok(())).unwrap();
    drop(tower);
    let rows = features.embeds.dim(0).unwrap();
    let p2 = env.capture("p2_vision_fp32.safetensors");
    let (max_error, mean_error) = relative_error(
        &features.embeds,
        &p2["vision_merger"].narrow(0, 0, rows).unwrap(),
    );
    eprintln!(
        "{vision:?}: tower merger vs fp32 capture: relative max {max_error:.3e}, mean {mean_error:.3e}"
    );
    if let LmVision::CapturedFp32 = vision {
        let captured = |name: &str| {
            p2[name]
                .narrow(0, 0, rows)
                .unwrap()
                .to_device(&device)
                .unwrap()
        };
        features.embeds = captured("vision_merger");
        features.deepstack = (0..features.deepstack.len())
            .map(|index| captured(&format!("vision_deepstack_{index}")))
            .collect();
    }
    let encoder = Qwen3Encoder::load_bf16(
        &env.text_encoder(),
        &env.tokenizer(),
        &device,
        lm_dtype,
        &Qwen3BF16Config::qwen3_image_21_text_encoder(),
        &progress,
    )
    .unwrap();
    let counts: Vec<usize> = refs.iter().map(|r| r.pad_count().unwrap()).collect();
    let ids = tokenize_image_conditioned(&encoder.tokenizer, P8_PROMPT, &counts).unwrap();
    let mrope =
        qwen_mrope_positions(&create_mm_token_type_ids(&ids), &features.grids, &[], 2).unwrap();
    let captured = env.capture(&format!("p3_p8_pos_lm_internals_{suffix}.safetensors"));
    let positions = captured["lm_position_ids"].to_vec3::<i64>().unwrap();
    for axis in 0..3 {
        let expected: Vec<u32> = positions[axis][0].iter().map(|p| *p as u32).collect();
        assert_eq!(mrope[axis], expected, "MRoPE axis {axis}");
    }
    let visual = VisualInjection {
        positions: ids
            .iter()
            .enumerate()
            .filter_map(|(i, id)| (*id == super::reference::QWEN3_VL_IMAGE_PAD_ID).then_some(i))
            .collect(),
        embeds: features.embeds.clone(),
        deepstack: features.deepstack.clone(),
    };
    let input_ids = Tensor::from_vec(ids.clone(), (1, ids.len()), &device).unwrap();
    let Some(Qwen3Model::BF16(model)) = encoder.model.as_ref() else {
        panic!("BF16 encoder expected")
    };
    let states = model
        .multimodal_hidden_states(&input_ids, &visual, &mrope)
        .unwrap();
    LM_CAPTURED_LAYERS
        .iter()
        .map(|&index| {
            let (max_error, mean_error) =
                relative_error(&states[index], &captured[&format!("lm_hidden_{index}")]);
            eprintln!(
                "{lm_dtype:?} LM, {vision:?} vs {suffix}: lm_hidden_{index} relative max {max_error:.3e}, mean {mean_error:.3e}"
            );
            (max_error, mean_error)
        })
        .collect()
}
/// Upstream's own bf16 language model's per-layer `(max, mean)` relative
/// error against its fp32 run: `p3_p8_pos_lm_internals_bf16` against
/// `_fp32`, in [`LM_CAPTURED_LAYERS`] order.
fn upstream_bf16_lm_errors() -> Vec<(f32, f32)> {
    let env = env();
    let bf16 = env.capture("p3_p8_pos_lm_internals_bf16.safetensors");
    let fp32 = env.capture("p3_p8_pos_lm_internals_fp32.safetensors");
    LM_CAPTURED_LAYERS
        .iter()
        .map(|index| {
            let name = format!("lm_hidden_{index}");
            let (max_error, mean_error) = relative_error(&bf16[&name], &fp32[&name]);
            eprintln!(
                "upstream bf16 vs fp32: {name} relative max {max_error:.3e}, mean {mean_error:.3e}"
            );
            (max_error, mean_error)
        })
        .collect()
}

/// Assert every captured layer's mean relative error is within its ceiling.
fn assert_lm_ceilings(label: &str, errors: &[(f32, f32)], ceilings: &[f32; 8]) {
    for ((index, (_, mean)), ceiling) in LM_CAPTURED_LAYERS.iter().zip(errors).zip(ceilings) {
        assert!(
            mean <= ceiling,
            "{label}: lm_hidden_{index} mean relative error {mean:.3e} exceeds {ceiling:.1e}"
        );
    }
}

/// fp32 per-layer mean relative-error ceilings against the fp32 capture,
/// [`LM_CAPTURED_LAYERS`] order. Measured (L40S, captured fp32 vision rows
/// injected so nothing upstream of the language model contributes):
/// [1.02e-5, 8.74e-6, 7.49e-6, 6.95e-6, 6.92e-6, 2.07e-5, 5.05e-5,
/// 5.43e-5]. The ceilings are 10x those, and at least 1e-4: fp32
/// accumulation-order noise, three orders of magnitude below the percent-level
/// error of any wrong constant (a RoPE section, an eps, a DeepStack layer).
const LM_FP32_CEILINGS: [f32; 8] = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 2.5e-4, 6e-4, 6e-4];

/// bf16 per-layer mean relative-error ceilings against the FP32 capture,
/// [`LM_CAPTURED_LAYERS`] order. Measured with the shipped F32 vision tower:
/// [1.41e-3, 3.42e-3, 4.24e-3, 4.82e-3, 5.55e-3, 2.15e-2, 3.92e-2,
/// 4.34e-2]. Each ceiling is 1.5x the measurement. The early layers
/// carry the discrimination — a layer-level bug appears at the first layer it
/// touches, where bf16's own rounding is still a fraction of a percent — and
/// the late layers keep the accumulated rounding from growing unnoticed.
///
/// The bf16 CAPTURE is not the target: upstream ran its vision tower in bf16,
/// which moves the merger ~14% (mean) from fp32, so its language-model states
/// sit 8-28% from the fp32 run (`upstream_bf16_lm_errors`) — no
/// several-percent defect could be seen against it. It is the ceiling of a
/// second gate instead: mold's bf16 language model must be no further from
/// the fp32 truth than upstream's own bf16 run is, layer by layer.
const LM_BF16_CEILINGS: [f32; 8] = [
    2.2e-3, 5.2e-3, 6.4e-3, 7.3e-3, 8.4e-3, 3.3e-2, 5.9e-2, 6.6e-2,
];

/// P3 language-model internals, fp32: MRoPE position ids exactly, then the
/// scatter and layers 0-3, 17, 34 and the pre-norm output against
/// transformers' `hidden_states`, with P2's captured vision rows injected so
/// the comparison isolates the language model.
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn p3_p8_language_model_internals_match_the_fp32_capture() {
    let errors = lm_internal_errors(DType::F32, LmVision::CapturedFp32, "fp32");
    assert_lm_ceilings("fp32 LM", &errors, &LM_FP32_CEILINGS);
}

/// P3 language-model internals, bf16 (the shipped configuration: BF16
/// language model, F32 vision tower), against the fp32 truth under absolute
/// ceilings and against upstream's own bf16 run layer by layer.
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn p3_p8_language_model_internals_match_the_bf16_capture() {
    let errors = lm_internal_errors(
        DType::BF16,
        LmVision::Tower(super::reference::vision_tower_dtype()),
        "fp32",
    );
    assert_lm_ceilings("bf16 LM", &errors, &LM_BF16_CEILINGS);
    for ((index, (_, ours)), (_, theirs)) in LM_CAPTURED_LAYERS
        .iter()
        .zip(&errors)
        .zip(upstream_bf16_lm_errors())
    {
        assert!(
            *ours <= theirs,
            "bf16 LM: lm_hidden_{index} is {ours:.3e} from fp32, upstream bf16 {theirs:.3e}"
        );
    }
}

/// P3 diagnostics, measurement only: why the bf16 CAPTURE cannot be a
/// target — upstream's own bf16 run against its fp32 run, and mold's bf16
/// language model (with the shipped F32 tower, then a BF16 one) against the
/// bf16 capture.
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn p3_p8_language_model_internals_study() {
    upstream_bf16_lm_errors();
    lm_internal_errors(DType::BF16, LmVision::Tower(DType::F32), "bf16");
    lm_internal_errors(DType::BF16, LmVision::Tower(DType::BF16), "bf16");
}
fn engine_paths(env: &Env) -> mold_core::ModelPaths {
    mold_core::ModelPaths {
        low_noise_transformer: None,
        low_noise_distilled_lora: None,
        transformer: env.transformer()[0].clone(),
        transformer_shards: env.transformer(),
        vae: env.vae(),
        spatial_upscaler: None,
        temporal_upscaler: None,
        distilled_lora: None,
        t5_encoder: None,
        clip_encoder: None,
        t5_tokenizer: None,
        clip_tokenizer: None,
        clip_encoder_2: None,
        clip_tokenizer_2: None,
        text_encoder_files: env.text_encoder(),
        text_tokenizer: Some(env.tokenizer()),
        decoder: None,
    }
}

/// P8's positive margins over upstream's own bf16 render, in dB of PSNR
/// against upstream's fp32 render
/// (`docs/qualification/qwen-image-2.1-image-conditioning.md`).
///
/// mold's BF16 engine is held to BEAT upstream's bf16 pipeline, not merely
/// to come within some distance of it, because the F32 vision tower and VAE
/// encoder (`reference::{vision_tower_dtype, vae_encoder_dtype}`) are exactly
/// what buys that lead: `conditioning_precision_study` re-runs the same
/// trajectories with both components in BF16 and asserts that variant FAILS
/// the turbo gate, so flipping either const back is a parity failure.
///
/// Measured on an L40S (CUDA fast path, upstream float32 rotary angles,
/// rounded timestep): the engine renders base4 at 38.80 dB and turbo6 at
/// 34.67 dB against upstream bf16's 37.59 and 33.22, i.e. +1.21 and +1.45 dB.
/// With both consts flipped to BF16 the ENGINE renders turbo6 at 33.48 dB
/// (+0.26 dB, failing this gate by 0.74 dB), and the component-level study
/// lands its BF16 variant at -6.43 dB, so the turbo margin sits between the
/// shipped path and every BF16 measurement. The 4-step
/// base render is not tower-sensitive (both variants clear it), and its
/// margin guards against the render falling behind upstream at all.
const P8_BASE_MARGIN_DB: f64 = 0.5;
const P8_TURBO_MARGIN_DB: f64 = 1.0;

fn p8_margin_db(case: &str) -> f64 {
    match case {
        "base4" => P8_BASE_MARGIN_DB,
        "turbo6" => P8_TURBO_MARGIN_DB,
        other => panic!("no P8 gate for {other}"),
    }
}

/// The P8 gate: mold's PSNR against upstream's fp32 render must exceed
/// upstream's own bf16 render's by at least `margin_db`.
fn p8_gate(ours_db: f64, upstream_bf16_db: f64, margin_db: f64) -> bool {
    ours_db >= upstream_bf16_db + margin_db
}

/// PSNR in dB of two `[H, W, 3]` float images in `[0, 1]`.
fn psnr(a: &Tensor, b: &Tensor) -> f64 {
    let mse = (a - b)
        .unwrap()
        .sqr()
        .unwrap()
        .mean_all()
        .unwrap()
        .to_dtype(DType::F64)
        .unwrap()
        .to_scalar::<f64>()
        .unwrap();
    10.0 * (1.0 / mse.max(1e-12)).log10()
}

/// P8 (base, 4 steps): the whole engine — reference preparation, vision,
/// multimodal encode, VAE encode, joint-layout denoise with the prefix cache,
/// decode — on upstream's own injected noise, against the upstream bf16
/// render. The engine runs at its device's working dtype (BF16 on CUDA), so
/// the gate is relative: mold's PSNR against upstream's fp32 render must beat
/// upstream's own bf16 render's by [`P8_BASE_MARGIN_DB`] ([`p8_gate`]).
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn p8_base_end_to_end_matches_the_upstream_capture() {
    use crate::engine::{InferenceEngine, LoadStrategy};
    let env = env();
    let mut engine = super::QwenImage21Engine::new(
        "qwen-image-2.1:bf16".to_string(),
        engine_paths(&env),
        LoadStrategy::Sequential,
        0,
    );
    engine.inject_initial_latents(env.capture("p8_noise.safetensors")["latents"].clone());
    let request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
        "prompt": P8_PROMPT,
        "model": "qwen-image-2.1:bf16",
        "width": 512,
        "height": 512,
        "steps": 4,
        "guidance": 1.0,
        "seed": 1234,
        "output_format": "png"
    }))
    .unwrap();
    let mut request = request;
    request.edit_images = Some(vec![std::fs::read(testdata("ref_opaque.png")).unwrap()]);
    let response = engine.generate(&request).unwrap();
    let decoded = image::load_from_memory(&response.images[0].data)
        .unwrap()
        .to_rgb8();
    assert_eq!(decoded.dimensions(), (512, 512));
    let ours = Tensor::from_vec(
        decoded
            .as_raw()
            .iter()
            .map(|byte| f32::from(*byte) / 255.0)
            .collect::<Vec<_>>(),
        (512, 512, 3),
        &Device::Cpu,
    )
    .unwrap();
    let rgb = |name: &str| {
        env.capture(name)["decoded_rgba_float"]
            .narrow(2, 0, 3)
            .unwrap()
            .clamp(0f32, 1f32)
            .unwrap()
    };
    let truth = rgb("p8_base4_fp32.safetensors");
    let upstream_bf16 = rgb("p8_base4_bf16.safetensors");
    let ours_psnr = psnr(&ours, &truth);
    let theirs_psnr = psnr(&upstream_bf16, &truth);
    let direct = psnr(&ours, &upstream_bf16);
    eprintln!(
        "P8 base4: mold vs fp32 {ours_psnr:.2} dB, upstream bf16 vs fp32 {theirs_psnr:.2} dB, mold vs upstream bf16 {direct:.2} dB"
    );
    std::fs::write(
        std::env::temp_dir().join("qwen21_p8_base4_mold.png"),
        &response.images[0].data,
    )
    .unwrap();
    assert!(
        p8_gate(ours_psnr, theirs_psnr, P8_BASE_MARGIN_DB),
        "{ours_psnr} dB vs upstream's {theirs_psnr} dB (gate: {P8_BASE_MARGIN_DB:+} dB)"
    );
}
/// Calibration and UAT probe (not a parity gate): render a 1024² CFG request
/// conditioned on `QWEN_IMAGE21_CALIBRATION_REFS` (default 1) copies of
/// `QWEN_IMAGE21_CALIBRATION_REF` (default `ref_opaque.png`), sequentially,
/// for `QWEN_IMAGE21_CALIBRATION_STEPS` (default 4) steps, optionally with
/// `QWEN_IMAGE21_CALIBRATION_TRANSPARENT=1` and a
/// `QWEN_IMAGE21_CALIBRATION_PROMPT`; print the sizing functions' estimate so
/// an external VRAM sampler can be read against it, and write the PNG to the
/// temp dir for inspection.
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn calibration_reference_render() {
    use crate::engine::{InferenceEngine, LoadStrategy};
    let env = env();
    let count: usize = std::env::var("QWEN_IMAGE21_CALIBRATION_REFS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let var = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
    let steps: u32 = var("QWEN_IMAGE21_CALIBRATION_STEPS")
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    let reference =
        var("QWEN_IMAGE21_CALIBRATION_REF").unwrap_or_else(|| "ref_opaque.png".to_string());
    let prompt = var("QWEN_IMAGE21_CALIBRATION_PROMPT").unwrap_or_else(|| P8_PROMPT.to_string());
    let transparent = var("QWEN_IMAGE21_CALIBRATION_TRANSPARENT").is_some();
    let mut engine = super::QwenImage21Engine::new(
        "qwen-image-2.1:bf16".to_string(),
        engine_paths(&env),
        LoadStrategy::Sequential,
        0,
    );
    let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
        "prompt": prompt,
        "negative_prompt": P6_NEGATIVE,
        "model": "qwen-image-2.1:bf16",
        "width": 1024,
        "height": 1024,
        "steps": steps,
        "guidance": 4.0,
        "seed": 7,
        "output_format": "png"
    }))
    .unwrap();
    if transparent {
        request.transparent_background = Some(true);
    }
    request.edit_images =
        (count > 0).then(|| vec![std::fs::read(testdata(&reference)).unwrap(); count]);
    let shape = crate::device::QwenImage21SequenceShape::for_request(
        1024,
        1024,
        &vec![(1536, 1024); count],
    );
    let base = crate::device::activation_bytes(
        1024,
        1024,
        1,
        2,
        crate::device::ActivationFamily::QwenImage21Dit,
    );
    let cache = crate::device::qwen_image21_prefix_cache_bytes(
        shape,
        2,
        2,
        crate::device::qwen_image21_prefix_cache_budget(None, 0, 0),
    );
    let workspace = crate::device::qwen_image21_reference_workspace_bytes(base, shape, 1, 2);
    eprintln!(
        "CALIBRATION refs={count} prefix_tokens={} cache={:.2} GiB workspace={:.2} GiB encode={:.2} GiB",
        shape.prefix_tokens(),
        cache as f64 / (1u64 << 30) as f64,
        workspace as f64 / (1u64 << 30) as f64,
        crate::device::qwen_image21_encode_phase_bytes(shape, 2) as f64 / (1u64 << 30) as f64,
    );
    let started = std::time::Instant::now();
    let response = engine.generate(&request).unwrap();
    eprintln!(
        "CALIBRATION done in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    std::fs::write(
        std::env::temp_dir().join(format!(
            "qwen21_calibration_{count}_{steps}{}.png",
            if transparent { "_rgba" } else { "" }
        )),
        &response.images[0].data,
    )
    .unwrap();
}
fn viggle(env: &Env, rank: usize) -> PathBuf {
    env.fixtures.join(format!(
        "../viggle/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r{rank}.safetensors"
    ))
}

fn p7(dtype: DType, suffix: &str, tolerance: f32) {
    use super::lora::{build_registry, Qwen21LoraEntry};
    let env = env();
    let device = device();
    let inputs = env.capture("p6_inputs.safetensors");
    let outputs = env.capture(&format!("p7_lora_r128_{suffix}.safetensors"));
    let truth = env.capture("p7_lora_r128_fp32.safetensors");
    let slots =
        bools(&load_capture(&testdata("p3_p6_pos_image_pad_mask.safetensors"))["image_pad_mask"]);
    let text_len = slots.len();
    let conditioning = QwenImage21TextConditioning {
        embeddings: inputs["prompt_embeds"]
            .to_device(&device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap(),
        valid_tokens: vec![vec![true; text_len]],
        image_slots: vec![slots.clone()],
    };
    let layout = QwenImage21JointLayout::build(
        &slots,
        &conditioning.valid_tokens,
        &[(52, 78), (72, 58)],
        (32, 32),
    )
    .unwrap();
    let cond = inputs["cond_latents"]
        .to_device(&device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let mut transformer = QwenImage21Transformer::load(
        &env.transformer(),
        &device,
        dtype,
        &ProgressReporter::default(),
    )
    .unwrap();
    let registry = build_registry(
        &[Qwen21LoraEntry {
            path: viggle(&env, 128),
            scale: 1.0,
        }],
        &device,
        dtype,
    )
    .unwrap();
    assert_eq!(registry.len(), 227);
    transformer.install_lora(Some(&registry)).unwrap();
    let timestep = f64::from(
        outputs["timestep_a"]
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0],
    );
    let latents = inputs["x_a"]
        .to_device(&device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let mut branch = transformer
        .prepare(
            &conditioning,
            layout,
            Some(cond),
            PrefixCacheDecision::Recompute,
        )
        .unwrap();
    let actual = branch.forward(&latents, timestep).unwrap();
    let target =
        |set: &HashMap<String, Tensor>, name: &str| set[name].narrow(1, 9298 - 1024, 1024).unwrap();
    // The adapter must actually move the prediction.
    let (moved, _) = relative_error(
        &target(&truth, "full_a_lora"),
        &target(&truth, "full_a_no_lora"),
    );
    assert!(
        moved > 1e-2,
        "the Viggle adapter barely moves upstream's output"
    );
    let label = format!("full_a_lora {suffix}");
    if dtype == DType::F32 {
        check(&label, &actual, &target(&outputs, "full_a_lora"), tolerance);
    } else {
        check_against_truth(
            &label,
            &actual,
            &target(&outputs, "full_a_lora"),
            &target(&truth, "full_a_lora"),
            tolerance,
        );
    }
    // Clearing restores the base forward.
    drop(branch);
    transformer.install_lora(None).unwrap();
}

/// P7 fp32: one two-reference transformer forward with the Viggle r128
/// adapter applied unmerged (PEFT upstream, bypass here).
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn p7_fp32_lora_matches_the_upstream_capture() {
    p7(DType::F32, "fp32", 1e-4);
}

/// P7 bf16.
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn p7_bf16_lora_matches_the_upstream_capture() {
    // `tolerance` is the allowed ratio to upstream's own bf16 error.
    p7(DType::BF16, "bf16", 1.5);
}

/// P8 (turbo, 6 steps): the turbo tier end to end — Viggle r256 installed by
/// the engine from the tier's distilled adapter, the recipe's six sigmas with
/// no terminal stretch — on upstream's injected noise, gated like P8 base by
/// [`P8_TURBO_MARGIN_DB`].
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn p8_turbo_end_to_end_matches_the_upstream_capture() {
    use crate::engine::{InferenceEngine, LoadStrategy};
    let env = env();
    let mut paths = engine_paths(&env);
    paths.distilled_lora = Some(viggle(&env, 256));
    let mut engine = super::QwenImage21Engine::new(
        "qwen-image-2.1-turbo:bf16".to_string(),
        paths,
        LoadStrategy::Sequential,
        0,
    );
    engine.inject_initial_latents(env.capture("p8_noise.safetensors")["latents"].clone());
    if std::env::var_os("QWEN_IMAGE21_DIAG_EVENTS").is_some() {
        engine.set_on_progress(Box::new(|event| eprintln!("EVENT {event:?}")));
    }
    let mut request: mold_core::GenerateRequest = serde_json::from_value(serde_json::json!({
        "prompt": P8_PROMPT,
        "model": "qwen-image-2.1-turbo:bf16",
        "width": 512,
        "height": 512,
        "steps": 6,
        "guidance": 1.0,
        "seed": 1234,
        "output_format": "png"
    }))
    .unwrap();
    request.edit_images = Some(vec![std::fs::read(testdata("ref_opaque.png")).unwrap()]);
    let response = engine.generate(&request).unwrap();
    let decoded = image::load_from_memory(&response.images[0].data)
        .unwrap()
        .to_rgb8();
    let ours = Tensor::from_vec(
        decoded
            .as_raw()
            .iter()
            .map(|byte| f32::from(*byte) / 255.0)
            .collect::<Vec<_>>(),
        (512, 512, 3),
        &Device::Cpu,
    )
    .unwrap();
    let rgb = |name: &str| {
        env.capture(name)["decoded_rgba_float"]
            .narrow(2, 0, 3)
            .unwrap()
            .clamp(0f32, 1f32)
            .unwrap()
    };
    let truth = rgb("p8_turbo6_fp32.safetensors");
    let upstream_bf16 = rgb("p8_turbo6_bf16.safetensors");
    let ours_psnr = psnr(&ours, &truth);
    let theirs_psnr = psnr(&upstream_bf16, &truth);
    eprintln!(
        "P8 turbo6: mold vs fp32 {ours_psnr:.2} dB, upstream bf16 vs fp32 {theirs_psnr:.2} dB, mold vs upstream bf16 {:.2} dB",
        psnr(&ours, &upstream_bf16)
    );
    std::fs::write(
        std::env::temp_dir().join("qwen21_p8_turbo6_mold.png"),
        &response.images[0].data,
    )
    .unwrap();
    assert!(
        p8_gate(ours_psnr, theirs_psnr, P8_TURBO_MARGIN_DB),
        "{ours_psnr} dB vs upstream's {theirs_psnr} dB (gate: {P8_TURBO_MARGIN_DB:+} dB)"
    );
}
/// P8 diagnostics: step-by-step latents of the base (4) and turbo (6)
/// trajectories against both upstream captures, from upstream's own
/// conditioning (P3 p8_pos) and condition latents (P4), isolating the
/// denoise loop from the encoders. Set `QWEN_IMAGE21_DIAG_TURBO=1` for turbo
/// and `QWEN_IMAGE21_DIAG_ROUND=1` to round the timestep like upstream.
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn p8_denoise_diagnostics() {
    use super::lora::{build_registry, Qwen21LoraEntry};
    use super::scheduler::{scheduler_for, ScheduleKind};
    let env = env();
    let turbo = std::env::var_os("QWEN_IMAGE21_DIAG_TURBO").is_some();
    let round = std::env::var_os("QWEN_IMAGE21_DIAG_ROUND").is_some();
    let dtype = DType::BF16;
    let device = device();
    let own_conditioning = std::env::var_os("QWEN_IMAGE21_DIAG_OWN_COND").is_some();
    let own_latents = std::env::var_os("QWEN_IMAGE21_DIAG_OWN_LATENTS").is_some();
    let conditioning = if own_conditioning {
        let progress = ProgressReporter::default();
        let mut encoder = Qwen3Encoder::load_bf16(
            &env.text_encoder(),
            &env.tokenizer(),
            &device,
            dtype,
            &Qwen3BF16Config::qwen3_image_21_text_encoder(),
            &progress,
        )
        .unwrap();
        let tower = load_vision_tower(
            &env.text_encoder(),
            &device,
            super::reference::vision_tower_dtype(),
            &progress,
        )
        .unwrap();
        let refs = references(&["ref_opaque.png"]);
        let vision = encode_vision(&tower, &refs, &device, &mut || Ok(())).unwrap();
        encode_prompt_with_images(&mut encoder, &vision, P8_PROMPT)
            .unwrap()
            .to_device_dtype(&device, dtype)
            .unwrap()
    } else {
        let capture = env.capture("p3_p8_pos_bf16.safetensors");
        let slots = bools(&capture["image_pad_mask"]);
        let text_len = slots.len();
        QwenImage21TextConditioning {
            embeddings: capture["prompt_embeds"]
                .to_device(&device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap(),
            valid_tokens: vec![vec![true; text_len]],
            image_slots: vec![slots],
        }
    };
    let slots = conditioning.image_slots[0].clone();
    let layout =
        QwenImage21JointLayout::build(&slots, &conditioning.valid_tokens, &[(52, 78)], (32, 32))
            .unwrap();
    let cond = if own_latents {
        let encoder = super::vae_encoder::QwenImage21VaeEncoder::load(
            &env.vae(),
            &device,
            crate::engine::gpu_dtype(&device),
            &ProgressReporter::default(),
        )
        .unwrap();
        let reference = &references(&["ref_opaque.png"])[0];
        let input = reference
            .vae_input(&device, crate::engine::gpu_dtype(&device))
            .unwrap();
        let packed = encoder.encode_packed(&input).unwrap();
        let (_, error) = relative_error(
            &packed,
            &env.capture("p4_vae_encode_fp32.safetensors")["opaque_packed"],
        );
        eprintln!("own condition latents vs fp32 capture: mean {error:.3e}");
        packed.to_dtype(dtype).unwrap()
    } else {
        env.capture("p4_vae_encode_fp32.safetensors")["opaque_packed"]
            .to_device(&device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    };
    let mut transformer = QwenImage21Transformer::load(
        &env.transformer(),
        &device,
        dtype,
        &ProgressReporter::default(),
    )
    .unwrap();
    let (name, steps, kind) = if turbo {
        let registry = build_registry(
            &[Qwen21LoraEntry {
                path: viggle(&env, 256),
                scale: 1.0,
            }],
            &device,
            dtype,
        )
        .unwrap();
        transformer.install_lora(Some(&registry)).unwrap();
        (
            "turbo6",
            6,
            ScheduleKind::for_model("qwen-image-2.1-turbo:bf16"),
        )
    } else {
        ("base4", 4, ScheduleKind::Base)
    };
    let fp32 = env.capture(&format!("p8_{name}_fp32.safetensors"));
    let bf16 = env.capture(&format!("p8_{name}_bf16.safetensors"));
    let (mut scheduler, _) = scheduler_for(kind, steps, 1024);
    let mut latents = env.capture("p8_noise.safetensors")["latents"]
        .to_device(&device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let mut branch = transformer
        .prepare(
            &conditioning,
            layout,
            Some(cond),
            PrefixCacheDecision::Retain,
        )
        .unwrap();
    for step in 0..steps {
        let timestep = super::scheduler::step_timestep(&scheduler, dtype, round);
        let prediction = branch.forward(&latents, timestep).unwrap();
        latents = scheduler.step(&prediction, &latents).unwrap();
        let key = format!("step{step}_latents");
        let (_, vs_fp32) = relative_error(&latents, &fp32[&key]);
        let (_, vs_bf16) = relative_error(&latents, &bf16[&key]);
        let (_, upstream) = relative_error(&bf16[&key], &fp32[&key]);
        eprintln!(
            "{name} step {step} t={timestep:.6}: mold vs fp32 {vs_fp32:.3e}, mold vs bf16 {vs_bf16:.3e}, upstream bf16 vs fp32 {upstream:.3e}"
        );
    }
    // Decode mold's final latents and upstream's fp32 final latents through
    // mold's VAE at the engine's dtype, and score both against the captures.
    let vae_dtype = crate::engine::gpu_dtype(&device);
    let vae = super::vae::QwenImage21Vae::load(
        &env.vae(),
        &device,
        vae_dtype,
        &ProgressReporter::default(),
    )
    .unwrap();
    let decode = |latents: &Tensor| -> Tensor {
        let decoded = vae
            .decode_packed(&latents.to_dtype(vae_dtype).unwrap(), 32, 32)
            .unwrap();
        // [1, 4, H, W] in [-1, 1] -> [H, W, 3] in [0, 1].
        ((decoded
            .i(0)
            .unwrap()
            .narrow(0, 0, 3)
            .unwrap()
            .permute((1, 2, 0))
            .unwrap()
            + 1.0)
            .unwrap()
            / 2.0)
            .unwrap()
            .clamp(0f32, 1f32)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
    };
    let rgb = |set: &HashMap<String, Tensor>| {
        set["decoded_rgba_float"]
            .narrow(2, 0, 3)
            .unwrap()
            .clamp(0f32, 1f32)
            .unwrap()
    };
    let ours = decode(&latents);
    let fp32_final = decode(&fp32["final_latents"].to_device(&device).unwrap());
    eprintln!(
        "{name} decode: mold latents vs fp32 image {:.2} dB, upstream fp32 latents through mold VAE vs fp32 image {:.2} dB, upstream bf16 image vs fp32 {:.2} dB",
        psnr(&ours, &rgb(&fp32)),
        psnr(&fp32_final, &rgb(&fp32)),
        psnr(&rgb(&bf16), &rgb(&fp32)),
    );
}
/// The F32 vision tower / VAE encoder evidence
/// (`reference::{vision_tower_dtype, vae_encoder_dtype}`), re-derived in one
/// command: both components run in BF16 and in F32 against the fp32 captures
/// (P2's merger and DeepStack maps, P4's packed condition latents), then the
/// P8 base and turbo trajectories are driven from each variant's own
/// conditioning and condition latents — the engine's composition, component
/// by component — and scored against upstream's fp32 render with the P8 gate.
/// On the turbo recipe F32 must pass it and the BF16 tower/encoder must fail
/// it, so the dtype choice is pinned by a measurement, not by a const fn; the
/// base4 rows are printed, not gated (see the note at the assertions).
///
/// ```text
/// QWEN_IMAGE21_MODEL_ROOT=/storage/mold/models \
/// QWEN_IMAGE21_FIXTURES=/storage/mold/fixtures/qwen_image21/captures \
/// CUDA_VISIBLE_DEVICES=3 cargo test -p mold-ai-inference --features cuda,cudnn,flash-attn \
///   --lib qwen_image21::parity_tests::conditioning_precision_study -- --ignored --nocapture
/// ```
#[test]
#[ignore = "requires QWEN_IMAGE21_MODEL_ROOT and QWEN_IMAGE21_FIXTURES"]
fn conditioning_precision_study() {
    use super::lora::{build_registry, Qwen21LoraEntry};
    use super::scheduler::{scheduler_for, ScheduleKind};
    let env = env();
    let device = device();
    let progress = ProgressReporter::default();
    let working = crate::engine::gpu_dtype(&device);
    let variants = [DType::BF16, DType::F32];
    let p2 = env.capture("p2_vision_fp32.safetensors");
    let p4 = env.capture("p4_vae_encode_fp32.safetensors");
    let both = references(&["ref_opaque.png", "ref_rgba.png"]);
    let opaque = references(&["ref_opaque.png"]);

    // Vision tower: P2 errors on both references, and the P8 reference's
    // features for the conditioning below.
    let mut vision = Vec::new();
    for dtype in variants {
        let tower = load_vision_tower(&env.text_encoder(), &device, dtype, &progress).unwrap();
        let features = encode_vision(&tower, &both, &device, &mut || Ok(())).unwrap();
        let (max, mean) = relative_error(&features.embeds, &p2["vision_merger"]);
        eprintln!("STUDY tower {dtype:?}: merger relative max {max:.3e}, mean {mean:.3e}");
        for (index, map) in features.deepstack.iter().enumerate() {
            let (max, mean) = relative_error(map, &p2[&format!("vision_deepstack_{index}")]);
            eprintln!(
                "STUDY tower {dtype:?}: deepstack {index} relative max {max:.3e}, mean {mean:.3e}"
            );
        }
        vision.push(encode_vision(&tower, &opaque, &device, &mut || Ok(())).unwrap());
    }
    let mut encoder = Qwen3Encoder::load_bf16(
        &env.text_encoder(),
        &env.tokenizer(),
        &device,
        working,
        &Qwen3BF16Config::qwen3_image_21_text_encoder(),
        &progress,
    )
    .unwrap();
    let truth_embeds = env.capture("p3_p8_pos_fp32.safetensors")["prompt_embeds"].clone();
    let conditioning: Vec<QwenImage21TextConditioning> = vision
        .iter()
        .zip(variants)
        .map(|(features, dtype)| {
            let conditioning = encode_prompt_with_images(&mut encoder, features, P8_PROMPT)
                .unwrap()
                .to_device_dtype(&device, working)
                .unwrap();
            let (max, mean) = relative_error(&conditioning.embeddings, &truth_embeds);
            eprintln!(
                "STUDY tower {dtype:?}: P3 p8_pos prompt_embeds relative max {max:.3e}, mean {mean:.3e}"
            );
            conditioning
        })
        .collect();
    drop(encoder);
    drop(vision);

    // VAE encoder: P4 errors on both references, and the P8 condition latents.
    let mut condition_latents = Vec::new();
    for dtype in variants {
        let vae_encoder =
            super::vae_encoder::QwenImage21VaeEncoder::load(&env.vae(), &device, dtype, &progress)
                .unwrap();
        for (reference, name) in both.iter().zip(["opaque", "rgba"]) {
            let packed = vae_encoder
                .encode_packed(&reference.vae_input(&device, dtype).unwrap())
                .unwrap();
            let (max, mean) = relative_error(&packed, &p4[&format!("{name}_packed")]);
            eprintln!(
                "STUDY vae encoder {dtype:?}: {name} packed latents relative max {max:.3e}, mean {mean:.3e}"
            );
            if name == "opaque" {
                condition_latents.push(packed.to_dtype(working).unwrap());
            }
        }
    }

    // The P8 trajectories from each variant's own conditioning.
    let mut transformer =
        QwenImage21Transformer::load(&env.transformer(), &device, working, &progress).unwrap();
    let noise = env.capture("p8_noise.safetensors")["latents"].clone();
    let mut finals = Vec::new();
    for (name, steps) in [("base4", 4usize), ("turbo6", 6)] {
        let kind = if name == "turbo6" {
            let registry = build_registry(
                &[Qwen21LoraEntry {
                    path: viggle(&env, 256),
                    scale: 1.0,
                }],
                &device,
                working,
            )
            .unwrap();
            transformer.install_lora(Some(&registry)).unwrap();
            ScheduleKind::for_model("qwen-image-2.1-turbo:bf16")
        } else {
            transformer.install_lora(None).unwrap();
            ScheduleKind::Base
        };
        for ((conditioning, cond), dtype) in
            conditioning.iter().zip(&condition_latents).zip(variants)
        {
            let layout = QwenImage21JointLayout::build(
                &conditioning.image_slots[0],
                &conditioning.valid_tokens,
                &[(52, 78)],
                (32, 32),
            )
            .unwrap();
            let (mut scheduler, _) = scheduler_for(kind, steps, 1024);
            let mut latents = noise.to_device(&device).unwrap().to_dtype(working).unwrap();
            let mut branch = transformer
                .prepare(
                    conditioning,
                    layout,
                    Some(cond.clone()),
                    PrefixCacheDecision::Retain,
                )
                .unwrap();
            let exec_path = transformer.exec_path();
            for _ in 0..steps {
                let timestep =
                    super::scheduler::step_timestep(&scheduler, working, !exec_path.is_legacy());
                let prediction = branch.forward(&latents, timestep).unwrap();
                latents = scheduler.step(&prediction, &latents).unwrap();
            }
            drop(branch);
            finals.push((name, dtype, latents));
        }
    }
    drop(transformer);

    let vae_dtype = working;
    let vae = super::vae::QwenImage21Vae::load(&env.vae(), &device, vae_dtype, &progress).unwrap();
    let decode = |latents: &Tensor| -> Tensor {
        let decoded = vae
            .decode_packed(&latents.to_dtype(vae_dtype).unwrap(), 32, 32)
            .unwrap();
        ((decoded
            .i(0)
            .unwrap()
            .narrow(0, 0, 3)
            .unwrap()
            .permute((1, 2, 0))
            .unwrap()
            + 1.0)
            .unwrap()
            / 2.0)
            .unwrap()
            .clamp(0f32, 1f32)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
    };
    let mut verdicts = Vec::new();
    for (name, dtype, latents) in &finals {
        let rgb = |file: &str| {
            env.capture(file)["decoded_rgba_float"]
                .narrow(2, 0, 3)
                .unwrap()
                .clamp(0f32, 1f32)
                .unwrap()
        };
        let truth = rgb(&format!("p8_{name}_fp32.safetensors"));
        let upstream = psnr(&rgb(&format!("p8_{name}_bf16.safetensors")), &truth);
        let ours = psnr(&decode(latents), &truth);
        let margin = p8_margin_db(name);
        let passes = p8_gate(ours, upstream, margin);
        eprintln!(
            "STUDY P8 {name} tower+encoder {dtype:?}: mold vs fp32 {ours:.2} dB, upstream bf16 vs fp32 {upstream:.2} dB, margin {:+.2} dB (gate {margin:+.2} dB: {})",
            ours - upstream,
            if passes { "pass" } else { "FAIL" }
        );
        verdicts.push((*name, *dtype, ours, passes));
    }
    // Only the turbo recipe discriminates the tower/encoder dtype, so only it
    // is asserted here. The component-level base4 row is not a stable
    // measurement: switching the rotary angles to upstream's float32 (a
    // change below 1e-5 in every table value) moved its F32 variant from
    // 39.39 to 33.66 dB and left its BF16 variant ABOVE it, while the engine's
    // own base4 render moved 0.11 dB. The engine-level gate
    // (`p8_base_end_to_end_matches_the_upstream_capture`) owns base4.
    let turbo = |dtype| {
        verdicts
            .iter()
            .find(|(name, d, _, _)| *name == "turbo6" && *d == dtype)
            .unwrap()
    };
    assert!(
        !turbo(DType::BF16).3,
        "a BF16 vision tower and VAE encoder pass the P8 turbo gate: the F32 choice is no longer evidenced"
    );
    eprintln!(
        "STUDY turbo P8 delta F32 - BF16: {:+.2} dB",
        turbo(DType::F32).2 - turbo(DType::BF16).2
    );
    assert!(
        turbo(DType::F32).3,
        "the shipped F32 conditioning fails the P8 turbo gate"
    );
}
