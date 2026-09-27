use super::*;
use candle_core::quantized::gguf_file;
use std::io::Write;
use std::path::Path;

fn cpu() -> Device {
    Device::Cpu
}

fn values(count: usize, salt: f32) -> Vec<f32> {
    (0..count)
        .map(|i| ((i as f32 + salt) * 0.37).sin() * 0.5)
        .collect()
}

fn tensor(rows: usize, cols: usize, salt: f32) -> Tensor {
    Tensor::from_vec(values(rows * cols, salt), (rows, cols), &cpu()).unwrap()
}

fn input(tokens: usize, width: usize) -> Tensor {
    Tensor::from_vec(values(tokens * width, 91.0), (1, tokens, width), &cpu()).unwrap()
}

fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

fn relative_error(actual: &Tensor, expected: &Tensor) -> f64 {
    let diff = (actual.to_dtype(DType::F64).unwrap() - expected.to_dtype(DType::F64).unwrap())
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f64>()
        .unwrap();
    let norm = expected
        .to_dtype(DType::F64)
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f64>()
        .unwrap();
    (diff / norm).sqrt()
}

fn reporter() -> crate::progress::ProgressReporter {
    crate::progress::ProgressReporter::default()
}

/// Raw safetensors writer: `(name, dtype, shape, bytes)`. Candle cannot build
/// an `I8` tensor, so the INT8 fixtures are written byte for byte.
fn write_safetensors(path: &Path, records: &[(String, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in records {
        let begin = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [begin, data.len()]}),
        );
    }
    let encoded = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(encoded.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&encoded).unwrap();
    file.write_all(&data).unwrap();
}

fn f32_bytes(t: &Tensor) -> Vec<u8> {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

fn fp8_bytes(t: &Tensor) -> Vec<u8> {
    // F8E4M3 is one byte per element; candle stores `float8::F8E4M3`.
    t.to_dtype(DType::F8E4M3)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<float8::F8E4M3>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

const DIM: usize = 256;
const HIDDEN: usize = 8;

#[test]
fn the_dense_arm_is_bit_identical_to_candle_linear() {
    let linear = candle_nn::Linear::new(tensor(6, DIM, 1.0), None);
    let wrapped = Q21Linear::dense(linear.clone()).unwrap();
    let x = input(3, DIM);
    assert_eq!(wrapped.kind(), Q21LinearKind::Dense);
    assert_eq!(
        wrapped
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        linear
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        "the BF16 tier must not move a single bit"
    );
}

#[test]
fn split_gate_up_takes_the_gate_rows_first() {
    let fused = Tensor::from_vec((0..8).map(|v| v as f32).collect(), (1, 1, 8), &cpu()).unwrap();
    let (gate, up) = split_gate_up(&fused, 4).unwrap();
    assert_eq!(
        gate.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        [0.0, 1.0, 2.0, 3.0]
    );
    assert_eq!(
        up.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        [4.0, 5.0, 6.0, 7.0]
    );
    assert!(split_gate_up(&fused, 3).is_err());
}

fn split_and_fused() -> (Q21GateUp, Q21GateUp) {
    let gate = tensor(HIDDEN, DIM, 2.0);
    let proj = tensor(HIDDEN, DIM, 3.0);
    let fused = Tensor::cat(&[&gate, &proj], 0).unwrap();
    (
        Q21GateUp::Split {
            gate: Q21Linear::dense(candle_nn::Linear::new(gate, None)).unwrap(),
            proj: Q21Linear::dense(candle_nn::Linear::new(proj, None)).unwrap(),
        },
        Q21GateUp::Fused {
            gate_up: Q21Linear::dense(candle_nn::Linear::new(fused, None)).unwrap(),
            hidden: HIDDEN,
        },
    )
}

/// `silu(gate_layer(x)) * proj(x)` is the same whether the checkpoint stores
/// the halves split (diffusers) or fused gate-first (ComfyUI/GGUF).
#[test]
fn fused_and_split_gate_up_compute_the_same_swiglu() {
    let (split, fused) = split_and_fused();
    let x = input(5, DIM);
    let a = split.swiglu(&x).unwrap();
    let b = fused.swiglu(&x).unwrap();
    assert_eq!(a.dims(), [1, 5, HIDDEN]);
    assert!(max_abs(&a, &b) < 1e-6);
    // Swapping the halves would change the result: gate order is load-bearing.
    let (gate, up) = split.forward(&x).unwrap();
    let swapped = (candle_nn::ops::silu(&up).unwrap() * gate).unwrap();
    assert!(max_abs(&a, &swapped) > 1e-3);
}

fn adapter(rows: usize, salt: f32, scale: f32) -> LinearLoraAdapter {
    LinearLoraAdapter {
        down: tensor(2, DIM, salt),
        up: tensor(rows, 2, salt + 1.0),
        scale,
        fused_slice: None,
    }
}

/// A `gate_layer` / `proj` LoRA installed on a fused checkpoint lands on its
/// own half, exactly as it would on the split one.
#[test]
fn gate_and_proj_adapters_land_on_their_halves_of_a_fused_linear() {
    let (mut split, mut fused) = split_and_fused();
    let gate = vec![adapter(HIDDEN, 4.0, 0.7)];
    let proj = vec![adapter(HIDDEN, 6.0, -0.4)];
    split.set_adapters(gate.clone(), proj.clone()).unwrap();
    fused.set_adapters(gate, proj).unwrap();
    let x = input(4, DIM);
    let (sg, sp) = split.forward(&x).unwrap();
    let (fg, fp) = fused.forward(&x).unwrap();
    assert!(max_abs(&sg, &fg) < 1e-5);
    assert!(max_abs(&sp, &fp) < 1e-5);

    // Clearing restores the base projections exactly.
    let (base_split, _) = split_and_fused();
    fused.clear_adapters();
    let (bg, bp) = base_split.forward(&x).unwrap();
    let (cg, cp) = fused.forward(&x).unwrap();
    assert_eq!(max_abs(&bg, &cg), 0.0);
    assert_eq!(max_abs(&bp, &cp), 0.0);
}

#[test]
fn set_adapters_refuses_a_stack_that_does_not_fit() {
    let mut linear = Q21Linear::dense(candle_nn::Linear::new(tensor(6, DIM, 1.0), None)).unwrap();
    // Wrong output height.
    assert!(linear.set_adapters(vec![adapter(5, 1.0, 1.0)]).is_err());
    // A slice that runs off the end.
    let mut sliced = adapter(4, 1.0, 1.0);
    sliced.fused_slice = Some(FusedSlice {
        offset: 4,
        length: 4,
    });
    assert!(linear.set_adapters(vec![sliced]).is_err());
    // A fitting one installs and is charged.
    linear.set_adapters(vec![adapter(6, 1.0, 1.0)]).unwrap();
    assert_eq!(linear.adapters().len(), 1);
    assert_eq!(linear.adapter_bytes(), ((2 * DIM + 6 * 2) * 4) as u64);
}

/// The adapter slot is the same on every arm: the delta is added to whatever
/// the base weight produced.
#[test]
fn every_arm_adds_the_same_bypass_delta() {
    let x = input(3, DIM);
    let bypass = adapter(6, 8.0, 0.5);
    let delta = bypass
        .apply(&x, &Tensor::zeros((1, 3, 6), DType::F32, &cpu()).unwrap())
        .unwrap();

    let weight = tensor(6, DIM, 1.0);
    let arms = [
        Q21Linear::dense(candle_nn::Linear::new(weight.clone(), None)).unwrap(),
        Q21Linear::fp8(
            weight.to_dtype(DType::F8E4M3).unwrap(),
            Tensor::ones((6, 1), DType::F32, &cpu()).unwrap(),
            None,
        )
        .unwrap(),
        Q21Linear::quantized(
            Arc::new(QTensor::quantize(&weight, GgmlDType::Q8_0).unwrap()),
            &cpu(),
            DType::F32,
            false,
        )
        .unwrap(),
    ];
    for mut arm in arms {
        let base = arm.forward(&x).unwrap();
        arm.set_adapters(vec![bypass.clone()]).unwrap();
        let adapted = arm.forward(&x).unwrap();
        assert!(
            max_abs(&(adapted - base).unwrap(), &delta) < 1e-5,
            "{:?}",
            arm.kind()
        );
    }
}

/// FP8: widen, one GEMM, per-row scale on the output — equal to multiplying
/// by the scaled weight.
#[test]
fn the_fp8_arm_matches_the_row_scaled_weight() {
    let weight = tensor(6, DIM, 5.0).to_dtype(DType::F8E4M3).unwrap();
    let scale = Tensor::from_vec(vec![0.5f32, 2.0, 1.0, 0.25, 3.0, 1.5], (6, 1), &cpu()).unwrap();
    let linear = Q21Linear::fp8(weight.clone(), scale.clone(), None).unwrap();
    assert_eq!(linear.kind(), Q21LinearKind::Fp8);
    let dense = weight
        .to_dtype(DType::F32)
        .unwrap()
        .broadcast_mul(&scale)
        .unwrap();
    let x = input(4, DIM);
    let expected = x.broadcast_matmul(&dense.t().unwrap()).unwrap();
    assert!(max_abs(&linear.forward(&x).unwrap(), &expected) < 1e-4);
    // Rank-4 activations take the same path.
    let x4 = x.unsqueeze(0).unwrap();
    assert!(
        max_abs(
            &linear.forward(&x4).unwrap(),
            &expected.unsqueeze(0).unwrap()
        ) < 1e-4
    );
}

/// INT8 ConvRot: the W8A8 forward agrees with the dequantized (unrotated)
/// weight within the dynamic activation quantizer's error.
#[test]
fn the_int8_arm_tracks_its_dequantized_weight() {
    let rows = 8;
    let packed_bytes: Vec<u8> = (0..rows * DIM)
        .map(|i| ((i * 37 + 11) % 255) as i32 - 127)
        .map(|v| v as i8 as u8)
        .collect();
    let packed = Tensor::from_vec(packed_bytes, (rows, DIM), &cpu()).unwrap();
    let scales = Tensor::from_vec(
        (0..rows)
            .map(|r| 0.002 + r as f32 * 1e-4)
            .collect::<Vec<_>>(),
        (rows, 1),
        &cpu(),
    )
    .unwrap();
    let linear = Q21Linear::int8(packed.clone(), scales.clone(), None).unwrap();
    assert_eq!(linear.kind(), Q21LinearKind::Int8ConvRot);
    let reference = ComfyInt8ConvRotLinear::new_on_device(packed, scales)
        .unwrap()
        .dequantize_weight(DType::F32, &cpu(), 256)
        .unwrap();
    let x = input(5, DIM);
    let expected = x.broadcast_matmul(&reference.t().unwrap()).unwrap();
    let actual = linear.forward(&x).unwrap();
    assert!(
        relative_error(&actual, &expected) < 0.02,
        "W8A8 diverged from the dequantized weight: {}",
        relative_error(&actual, &expected)
    );
}

#[test]
fn a_finite_prediction_passes_and_a_nan_names_the_tier_and_switch() {
    let finite = input(2, 4);
    ensure_finite_prediction(&finite, 0, "gguf (Q4K)", false).unwrap();
    let nan = Tensor::from_vec(vec![1.0f32, f32::NAN, 0.0, 2.0], (1, 4), &cpu()).unwrap();
    let error = ensure_finite_prediction(&nan, 6, "gguf (Q4K)", true)
        .unwrap_err()
        .to_string();
    assert!(error.contains("gguf (Q4K)"), "{error}");
    assert!(error.contains("step 7"), "{error}");
    assert!(error.contains(QMATMUL_ENV), "{error}");
    let inf = Tensor::from_vec(vec![f32::INFINITY, 0.0], (1, 2), &cpu()).unwrap();
    let error = ensure_finite_prediction(&inf, 0, "fp8", false)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("fp8") && !error.contains(QMATMUL_ENV),
        "{error}"
    );
    // Large but finite values never false-positive.
    let large = Tensor::from_vec(vec![3.0e38f32, -3.0e38, 3.0e38], (1, 3), &cpu()).unwrap();
    ensure_finite_prediction(&large, 0, "bf16", false).unwrap();
}

#[test]
fn the_qmatmul_switch_parses_like_every_other_family() {
    for truthy in ["1", "true", "ON", " yes "] {
        assert!(parse_qwen_image21_qmatmul(Some(truthy)), "{truthy}");
    }
    for falsey in [None, Some("0"), Some("off"), Some("banana")] {
        assert!(!parse_qwen_image21_qmatmul(falsey), "{falsey:?}");
    }
}

/// A block's worth of tensors in the diffusers key space.
fn block_tensors() -> Vec<(String, Tensor)> {
    vec![
        ("img_in.weight".into(), tensor(DIM, 64, 1.0)),
        (
            "transformer_blocks.0.attn.to_q.weight".into(),
            tensor(DIM, DIM, 2.0),
        ),
        (
            "transformer_blocks.0.attn.norm_q.weight".into(),
            Tensor::ones(128, DType::F32, &cpu()).unwrap(),
        ),
        (
            "transformer_blocks.0.img_mlp.gate_layer.weight".into(),
            tensor(HIDDEN, DIM, 3.0),
        ),
        (
            "transformer_blocks.0.img_mlp.proj.weight".into(),
            tensor(HIDDEN, DIM, 4.0),
        ),
    ]
}

#[test]
fn a_bf16_source_builds_dense_split_linears() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("shard.safetensors");
    let map = block_tensors()
        .into_iter()
        .map(|(name, t)| (name, t.to_dtype(DType::BF16).unwrap()))
        .collect::<std::collections::HashMap<_, _>>();
    candle_core::safetensors::save(&map, &path).unwrap();
    let source = Q21WeightSource::open(&[path], &cpu(), DType::F32, false, &reporter()).unwrap();
    assert_eq!(source.format(), QwenImage21TransformerFormat::Bf16);
    assert_eq!(source.tier_label(), "bf16");
    let q = source
        .linear("transformer_blocks.0.attn.to_q", DIM, DIM)
        .unwrap();
    assert_eq!(q.kind(), Q21LinearKind::Dense);
    assert!(source
        .linear("transformer_blocks.0.attn.to_q", DIM, 7)
        .is_err());
    let mlp = source
        .gate_up("transformer_blocks.0.img_mlp", DIM, HIDDEN)
        .unwrap();
    assert!(matches!(mlp, Q21GateUp::Split { .. }));
    let norm = source
        .tensor("transformer_blocks.0.attn.norm_q.weight", DType::F32)
        .unwrap();
    assert_eq!(norm.dims(), [128]);
    assert_eq!(
        source.logical_linear_names(),
        [
            "img_in",
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.img_mlp.gate_layer",
            "transformer_blocks.0.img_mlp.proj",
        ]
    );
}

#[test]
fn an_fp8_source_builds_fp8_linears_and_dense_leftovers() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("fp8.safetensors");
    let q = tensor(DIM, DIM, 2.0);
    let scale = Tensor::full(0.5f32, (DIM, 1), &cpu()).unwrap();
    let records = vec![
        (
            "img_in.weight".to_string(),
            "F32",
            vec![DIM, 64],
            f32_bytes(&tensor(DIM, 64, 1.0)),
        ),
        (
            "transformer_blocks.0.attn.to_q._weight_qdata".to_string(),
            "F8_E4M3",
            vec![DIM, DIM],
            fp8_bytes(&q),
        ),
        (
            "transformer_blocks.0.attn.to_q._weight_scale".to_string(),
            "F32",
            vec![DIM, 1],
            f32_bytes(&scale),
        ),
    ];
    write_safetensors(&path, &records);
    let source = Q21WeightSource::open(&[path], &cpu(), DType::F32, false, &reporter()).unwrap();
    assert_eq!(source.format(), QwenImage21TransformerFormat::TorchaoFp8);
    assert!(source.contains("transformer_blocks.0.attn.to_q.weight"));
    let linear = source
        .linear("transformer_blocks.0.attn.to_q", DIM, DIM)
        .unwrap();
    assert_eq!(linear.kind(), Q21LinearKind::Fp8);
    assert_eq!(
        source.linear("img_in", 64, DIM).unwrap().kind(),
        Q21LinearKind::Dense
    );
    let dequant = source
        .dequantized_weight("transformer_blocks.0.attn.to_q", &cpu())
        .unwrap();
    let expected = q
        .to_dtype(DType::F8E4M3)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .affine(0.5, 0.0)
        .unwrap();
    assert_eq!(max_abs(&dequant, &expected), 0.0);
    assert_eq!(
        source.logical_linear_names(),
        ["img_in", "transformer_blocks.0.attn.to_q"]
    );
}

#[test]
fn an_int8_source_keeps_gate_up_fused_and_answers_both_halves() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("int8.safetensors");
    let rows = 2 * HIDDEN;
    let packed: Vec<u8> = (0..rows * DIM)
        .map(|i| (((i * 13 + 5) % 251) as i32 - 125) as i8 as u8)
        .collect();
    let scales = Tensor::from_vec(
        (0..rows)
            .map(|r| 0.01 + r as f32 * 1e-3)
            .collect::<Vec<_>>(),
        (rows, 1),
        &cpu(),
    )
    .unwrap();
    let marker = br#"{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}"#;
    let base = "transformer_blocks.0.img_mlp.gate_up";
    let records = vec![
        (
            "img_in.weight".to_string(),
            "F32",
            vec![DIM, 64],
            f32_bytes(&tensor(DIM, 64, 1.0)),
        ),
        (
            format!("{base}.weight"),
            "I8",
            vec![rows, DIM],
            packed.clone(),
        ),
        (
            format!("{base}.weight_scale"),
            "F32",
            vec![rows, 1],
            f32_bytes(&scales),
        ),
        (
            format!("{base}.comfy_quant"),
            "U8",
            vec![marker.len()],
            marker.to_vec(),
        ),
    ];
    write_safetensors(&path, &records);
    let source = Q21WeightSource::open(&[path], &cpu(), DType::F32, false, &reporter()).unwrap();
    assert_eq!(
        source.format(),
        QwenImage21TransformerFormat::ComfyInt8ConvRot
    );
    let mlp = source
        .gate_up("transformer_blocks.0.img_mlp", DIM, HIDDEN)
        .unwrap();
    let Q21GateUp::Fused { gate_up, hidden } = &mlp else {
        panic!("a Comfy checkpoint's gate_up must stay fused");
    };
    assert_eq!(*hidden, HIDDEN);
    assert_eq!(gate_up.kind(), Q21LinearKind::Int8ConvRot);

    let full = ComfyInt8ConvRotLinear::new_on_device(
        Tensor::from_vec(packed, (rows, DIM), &cpu()).unwrap(),
        scales,
    )
    .unwrap()
    .dequantize_weight(DType::F32, &cpu(), 256)
    .unwrap();
    let gate = source
        .dequantized_weight("transformer_blocks.0.img_mlp.gate_layer", &cpu())
        .unwrap();
    let proj = source
        .dequantized_weight("transformer_blocks.0.img_mlp.proj", &cpu())
        .unwrap();
    assert_eq!(max_abs(&gate, &full.narrow(0, 0, HIDDEN).unwrap()), 0.0);
    assert_eq!(
        max_abs(&proj, &full.narrow(0, HIDDEN, HIDDEN).unwrap()),
        0.0
    );
    assert_eq!(
        source.logical_linear_names(),
        [
            "img_in",
            "transformer_blocks.0.img_mlp.gate_layer",
            "transformer_blocks.0.img_mlp.proj",
        ]
    );
}

/// Unsloth's key space: every tensor under `model.diffusion_model.`, mixed
/// block types, fused `gate_up`. The source strips the prefix, so the
/// transformer asks for the same names on every tier.
#[test]
fn an_unsloth_style_gguf_source_strips_the_prefix_and_mixes_block_types() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("unsloth.gguf");
    let gate = tensor(HIDDEN, DIM, 3.0);
    let proj = tensor(HIDDEN, DIM, 4.0);
    let fused = Tensor::cat(&[&gate, &proj], 0).unwrap();
    let tensors = [
        (
            "model.diffusion_model.img_in.weight",
            QTensor::quantize(&tensor(DIM, 64, 1.0), GgmlDType::F32).unwrap(),
        ),
        (
            "model.diffusion_model.transformer_blocks.0.attn.to_q.weight",
            QTensor::quantize(&tensor(DIM, DIM, 2.0), GgmlDType::Q5K).unwrap(),
        ),
        (
            "model.diffusion_model.transformer_blocks.0.attn.norm_q.weight",
            QTensor::quantize(
                &Tensor::ones(128, DType::F32, &cpu()).unwrap(),
                GgmlDType::F32,
            )
            .unwrap(),
        ),
        (
            "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight",
            QTensor::quantize(&fused, GgmlDType::Q8_0).unwrap(),
        ),
    ];
    let refs = tensors.iter().map(|(n, t)| (*n, t)).collect::<Vec<_>>();
    let mut file = std::fs::File::create(&path).unwrap();
    gguf_file::write(&mut file, &[], &refs).unwrap();
    drop(file);

    let source = Q21WeightSource::open(&[path], &cpu(), DType::F32, false, &reporter()).unwrap();
    assert_eq!(
        source.format(),
        QwenImage21TransformerFormat::Gguf {
            diffusion_model_prefix: true
        }
    );
    assert_eq!(source.tier_label(), "gguf (Q5K+Q8_0)");
    let q = source
        .linear("transformer_blocks.0.attn.to_q", DIM, DIM)
        .unwrap();
    // The CPU takes the dequant arm (candle's CPU QMatMul is not the path
    // mold qualifies for this family); a float tensor resolves Dense.
    assert_eq!(q.kind(), Q21LinearKind::GgufDequant);
    assert_eq!(
        source.linear("img_in", 64, DIM).unwrap().kind(),
        Q21LinearKind::GgufDense
    );
    let mlp = source
        .gate_up("transformer_blocks.0.img_mlp", DIM, HIDDEN)
        .unwrap();
    assert!(matches!(mlp, Q21GateUp::Fused { .. }));
    let x = input(3, DIM);
    let (g, p) = mlp.forward(&x).unwrap();
    let expected_gate = x.broadcast_matmul(&gate.t().unwrap()).unwrap();
    let expected_proj = x.broadcast_matmul(&proj.t().unwrap()).unwrap();
    assert!(relative_error(&g, &expected_gate) < 0.01);
    assert!(relative_error(&p, &expected_proj) < 0.01);
    assert_eq!(
        source
            .tensor("transformer_blocks.0.attn.norm_q.weight", DType::F32)
            .unwrap()
            .dims(),
        [128]
    );
    assert_eq!(
        source.logical_linear_names(),
        [
            "img_in",
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.img_mlp.gate_layer",
            "transformer_blocks.0.img_mlp.proj",
        ]
    );
}

// ── Weight-gated parity against the BF16 shards ─────────────────────────────

/// Relative Frobenius-error ceilings per tier, set from the measured
/// full-checkpoint maxima (see the qualification record) with headroom.
fn tier_bound(file: &str) -> f64 {
    match file {
        f if f.contains("int8") => 0.013,
        f if f.contains("FP8") => 0.035,
        f if f.contains("Q8_0") => 0.009,
        f if f.contains("Q6_K") => 0.028,
        f if f.contains("Q5_0") => 0.07,
        f if f.contains("Q4_K") => 0.10,
        f if f.contains("Q3_K") => 0.20,
        f if f.contains("Q2_K") => 0.35,
        other => panic!("no bound for {other}"),
    }
}

fn parity_device() -> Device {
    Device::cuda_if_available(0).unwrap()
}

/// Every linear of every staged tier, dequantized, against the BF16 shards —
/// the whole checkpoint, not a sample — plus every non-linear tensor exactly.
///
/// `MOLD_QWEN_IMAGE21_TIERS_DIR` names the staging directory,
/// `MOLD_QWEN_IMAGE21_BF16_DIR` the BF16 model directory (on plato
/// `/storage/mold/models/qwen-image-2.1-bf16`); `MOLD_QWEN_IMAGE21_TIERS`
/// optionally narrows the files (comma-separated).
#[test]
#[ignore = "needs the staged Qwen Image 2.1 tier checkpoints and the BF16 shards"]
fn every_staged_tier_dequantizes_to_the_bf16_shards() {
    let tiers_dir = PathBuf::from(std::env::var("MOLD_QWEN_IMAGE21_TIERS_DIR").unwrap());
    let bf16_dir =
        PathBuf::from(std::env::var("MOLD_QWEN_IMAGE21_BF16_DIR").unwrap()).join("transformer");
    let device = parity_device();
    let bf16_paths = vec![
        bf16_dir.join("diffusion_pytorch_model-00001-of-00002.safetensors"),
        bf16_dir.join("diffusion_pytorch_model-00002-of-00002.safetensors"),
    ];
    let reference =
        Q21WeightSource::open(&bf16_paths, &device, DType::BF16, false, &reporter()).unwrap();
    let reference_names = reference.logical_linear_names();
    assert_eq!(reference_names.len(), 8 + 32 * 7, "{reference_names:?}");

    let all = [
        "qwen_image_2.1_int8_convrot.safetensors",
        "Qwen-Image-2.1-FP8.safetensors",
        "qwen_image_2.1-Q8_0.gguf",
        "qwen_image_2.1-Q6_K.gguf",
        "qwen_image_2.1-Q5_0.gguf",
        "qwen_image_2.1-Q4_K.gguf",
        "qwen_image_2.1-Q3_K.gguf",
        "qwen_image_2.1-Q2_K.gguf",
    ];
    let selected = std::env::var("MOLD_QWEN_IMAGE21_TIERS").ok();
    for file in all.iter().filter(|file| {
        selected
            .as_deref()
            .is_none_or(|list| list.split(',').any(|entry| file.contains(entry.trim())))
    }) {
        let source = Q21WeightSource::open(
            &[tiers_dir.join(file)],
            &device,
            DType::BF16,
            false,
            &reporter(),
        )
        .unwrap();
        assert_eq!(
            source.logical_linear_names(),
            reference_names,
            "{file} must carry exactly the reference's linears"
        );
        let (mut worst, mut worst_name, mut sum, mut quantized) = (0.0f64, String::new(), 0.0, 0);
        let (mut dense_worst, mut dense_worst_name) = (0.0f64, String::new());
        for name in &reference_names {
            let expected = reference.dequantized_weight(name, &device).unwrap();
            let actual = source.dequantized_weight(name, &device).unwrap();
            assert_eq!(actual.dims(), expected.dims(), "{file} {name}");
            let error = relative_error(&actual, &expected);
            // Linears the tier left in BF16 (the stems, and every non-block
            // linear of the leejet and Comfy files) must be bit-exact.
            let stored_dense = match &source.backend {
                Q21Backend::Gguf(vb) => {
                    let key = if name.ends_with(".img_mlp.gate_layer")
                        || name.ends_with(".img_mlp.proj")
                    {
                        format!("{}.gate_up.weight", name.rsplit_once('.').unwrap().0)
                    } else {
                        format!("{name}.weight")
                    };
                    matches!(
                        vb.get_no_shape(&key).unwrap().dtype(),
                        GgmlDType::BF16 | GgmlDType::F16 | GgmlDType::F32
                    )
                }
                Q21Backend::Safetensors(st) => st
                    .get(&format!("{name}.weight"))
                    .is_ok_and(|view| format!("{:?}", view.dtype()) == "BF16"),
            };
            if stored_dense {
                if error > dense_worst {
                    dense_worst = error;
                    dense_worst_name = name.clone();
                }
                continue;
            }
            quantized += 1;
            sum += error;
            if error > worst {
                worst = error;
                worst_name = name.clone();
            }
        }
        eprintln!(
            "PARITY {file}: {quantized} quantized linears, mean rel err {:.5}, max {:.5} ({worst_name}); dense max {:.2e} ({dense_worst_name})",
            sum / quantized.max(1) as f64,
            worst,
            dense_worst
        );
        assert_eq!(dense_worst, 0.0, "{file}: a BF16-stored linear moved");
        assert!(
            worst <= tier_bound(file),
            "{file}: {worst_name} relative error {worst} exceeds {}",
            tier_bound(file)
        );
        // Norm scales are stored at full precision in every tier.
        for norm in [
            "txt_in.text_norm.weight",
            "transformer_blocks.0.attn.norm_q.weight",
            "transformer_blocks.31.attn.norm_k.weight",
        ] {
            let a = source.tensor(norm, DType::F32).unwrap();
            let b = reference.tensor(norm, DType::F32).unwrap();
            assert_eq!(max_abs(&a, &b), 0.0, "{file}: {norm}");
        }
    }
}

/// One whole transformer block's linears built through each tier's arm run
/// the forward those weights describe: the arm (dequant GEMM, W8A8, FP8
/// widen) against `x @ dequantized_weightᵀ`, on the parity device.
#[test]
#[ignore = "needs the staged Qwen Image 2.1 tier checkpoints"]
fn every_staged_tiers_linear_arms_compute_with_their_weights() {
    let tiers_dir = PathBuf::from(std::env::var("MOLD_QWEN_IMAGE21_TIERS_DIR").unwrap());
    let device = parity_device();
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    for (file, bound) in [
        ("qwen_image_2.1_int8_convrot.safetensors", 0.03),
        ("Qwen-Image-2.1-FP8.safetensors", 0.01),
        ("qwen_image_2.1-Q8_0.gguf", 0.01),
        ("qwen_image_2.1-Q4_K.gguf", 0.01),
    ] {
        let source =
            Q21WeightSource::open(&[tiers_dir.join(file)], &device, dtype, false, &reporter())
                .unwrap();
        let x = Tensor::randn(0f32, 1.0, (1, 64, 4096), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let block = "transformer_blocks.7";
        let mut cases = vec![(
            format!("{block}.attn.to_q"),
            source
                .linear(&format!("{block}.attn.to_q"), 4096, 4096)
                .unwrap()
                .forward(&x)
                .unwrap(),
        )];
        let (gate, proj) = source
            .gate_up(&format!("{block}.img_mlp"), 4096, 12288)
            .unwrap()
            .forward(&x)
            .unwrap();
        cases.push((format!("{block}.img_mlp.gate_layer"), gate));
        cases.push((format!("{block}.img_mlp.proj"), proj));
        for (name, actual) in cases {
            let weight = source.dequantized_weight(&name, &device).unwrap();
            let expected = x
                .to_dtype(DType::F32)
                .unwrap()
                .broadcast_matmul(&weight.t().unwrap())
                .unwrap();
            let error = relative_error(&actual.to_dtype(DType::F32).unwrap(), &expected);
            eprintln!("ARM {file} {name}: rel err {error:.5}");
            assert!(error < bound, "{file} {name}: {error}");
        }
    }
}
