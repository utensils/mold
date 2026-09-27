//! Weight-gated parity: the official `Qwen/Qwen3-VL-8B-Instruct-GGUF` language
//! model against the BF16 text-encoder shards Qwen Image 2.1 ships.
//!
//! Three gates, all on real checkpoints:
//! 1. every dequantized GGUF tensor against its BF16 shard counterpart (the
//!    whole language model, not a sample) — this is also what proves
//!    llama.cpp's converter kept HuggingFace's Q/K row order;
//! 2. the conditioning hidden states `encode_t2i_prompts` produces, through
//!    the Qwen Image 2.1 t2i template, per valid token: cosine >= 0.995
//!    against the BF16 encoder;
//! 3. the same per-token gate on the reference-conditioned (multimodal)
//!    forward: interleaved MRoPE, visual rows, DeepStack and more than one
//!    attention chunk.
//!
//! Environment: `MOLD_QWEN_IMAGE21_TIERS_DIR` (the GGUF files; on plato
//! `/storage/mold/fixtures/qwen_image21/tiers-staging`) and
//! `MOLD_QWEN_IMAGE21_SHARED_DIR` (the BF16 shards and tokenizer;
//! `/storage/mold/models/shared/qwen-image21`).

use std::path::PathBuf;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, IndexOp, Tensor};

use super::qwen3::Qwen3Encoder;
use super::qwen3_bf16::Qwen3BF16Config;

const GGUF_TIERS: [(&str, f64); 2] = [
    ("Qwen3VL-8B-Instruct-Q8_0.gguf", 0.01),
    ("Qwen3VL-8B-Instruct-Q4_K_M.gguf", 0.12),
];

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set")))
}

fn shards() -> Vec<PathBuf> {
    let dir = env_dir("MOLD_QWEN_IMAGE21_SHARED_DIR").join("text_encoder");
    (1..=4)
        .map(|i| dir.join(format!("model-0000{i}-of-00004.safetensors")))
        .collect()
}

fn device() -> Device {
    Device::cuda_if_available(0).unwrap()
}

/// llama.cpp's Qwen3 names → the HF names under `model.language_model`.
fn hf_name(gguf: &str) -> Option<String> {
    let root = "model.language_model";
    match gguf {
        "token_embd.weight" => return Some(format!("{root}.embed_tokens.weight")),
        "output_norm.weight" => return Some(format!("{root}.norm.weight")),
        "output.weight" => return Some("lm_head.weight".to_string()),
        _ => {}
    }
    let rest = gguf.strip_prefix("blk.")?;
    let (layer, leaf) = rest.split_once('.')?;
    let module = match leaf {
        "attn_norm.weight" => "input_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("{root}.layers.{layer}.{module}"))
}

fn relative_error(actual: &Tensor, expected: &Tensor) -> f64 {
    let sq = |t: &Tensor| {
        t.to_dtype(DType::F32)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64
    };
    let diff =
        (actual.to_dtype(DType::F32).unwrap() - expected.to_dtype(DType::F32).unwrap()).unwrap();
    (sq(&diff) / sq(expected)).sqrt()
}

#[test]
#[ignore = "needs the Qwen3-VL-8B GGUF files and the Qwen Image 2.1 text-encoder shards"]
fn every_gguf_tensor_dequantizes_to_the_bf16_shards() {
    let device = device();
    let refs = shards();
    let refs = refs.iter().map(PathBuf::as_path).collect::<Vec<_>>();
    // SAFETY: read-only mapping of verified model files.
    let reference = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&refs) }.unwrap();
    let tiers = env_dir("MOLD_QWEN_IMAGE21_TIERS_DIR");
    for (file, bound) in GGUF_TIERS {
        let path = tiers.join(file);
        let mut reader = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let (mut worst, mut worst_name, mut count, mut norms) = (0.0f64, String::new(), 0, 0);
        for name in content.tensor_infos.keys() {
            let hf = hf_name(name).unwrap_or_else(|| panic!("{file}: unmapped tensor {name}"));
            let expected = reference.load(&hf, &device).unwrap();
            let actual = content
                .tensor(&mut reader, name, &device)
                .unwrap()
                .dequantize(&device)
                .unwrap();
            assert_eq!(actual.dims(), expected.dims(), "{file} {name}");
            let error = relative_error(&actual, &expected);
            if expected.rank() == 1 {
                // Norm scales are stored in F32, widened from the BF16 values.
                assert_eq!(error, 0.0, "{file}: norm {name} moved");
                norms += 1;
                continue;
            }
            count += 1;
            if error > worst {
                worst = error;
                worst_name = name.clone();
            }
        }
        eprintln!(
            "TE-PARITY {file}: {count} matrices, max rel err {worst:.5} ({worst_name}); {norms} norms exact"
        );
        assert_eq!(count, 36 * 7 + 2, "{file}: every matrix compared");
        assert!(
            worst <= bound,
            "{file}: {worst_name} error {worst} > {bound}"
        );
    }
}

fn tokenizer_path() -> PathBuf {
    env_dir("MOLD_QWEN_IMAGE21_SHARED_DIR").join("processor/tokenizer.json")
}

/// Per-valid-token cosine between two conditioning tensors.
fn cosines(
    a: &crate::qwen_image21::QwenImage21TextConditioning,
    b: &crate::qwen_image21::QwenImage21TextConditioning,
) -> Vec<f64> {
    assert_eq!(a.valid_tokens, b.valid_tokens);
    let mut out = Vec::new();
    for (row, valid) in a.valid_tokens.iter().enumerate() {
        for (token, real) in valid.iter().enumerate() {
            if !real {
                continue;
            }
            let x = a
                .embeddings
                .i((row, token))
                .unwrap()
                .to_dtype(DType::F64)
                .unwrap();
            let y = b
                .embeddings
                .i((row, token))
                .unwrap()
                .to_dtype(DType::F64)
                .unwrap();
            let dot = (&x * &y)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f64>()
                .unwrap();
            let nx = x
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f64>()
                .unwrap()
                .sqrt();
            let ny = y
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f64>()
                .unwrap()
                .sqrt();
            out.push(dot / (nx * ny));
        }
    }
    out
}

#[test]
#[ignore = "needs the Qwen3-VL-8B GGUF files and the Qwen Image 2.1 text-encoder shards"]
fn gguf_conditioning_tracks_the_bf16_encoder_on_the_t2i_template() {
    let device = device();
    let progress = crate::progress::ProgressReporter::default();
    // Two prompts of different length: the batch is left-padded, so the
    // shorter row exercises the padded mask and the restarted RoPE positions.
    let prompts = vec![
        "A red fox curled asleep on a mossy stone in a misty pine forest at dawn, soft golden light, \
         shallow depth of field, a hand-painted wooden sign reading \"MOON CAFE\" in the background"
            .to_string(),
        "a cup of coffee".to_string(),
    ];
    // The oracle is the BF16 checkpoint run in F32 — the precision the GGUF
    // encoder computes in — so the gate measures the quantization, not the
    // BF16 reference's own rounding. The BF16-in-BF16 run is reported beside
    // it as the precision the shipped BF16 tier already renders at.
    let encode = |dtype: DType| {
        let mut encoder = Qwen3Encoder::load_bf16(
            &shards(),
            &tokenizer_path(),
            &device,
            dtype,
            &Qwen3BF16Config::qwen3_image_21_text_encoder(),
            &progress,
        )
        .unwrap();
        crate::qwen_image21::encode_t2i_prompts(&mut encoder, &prompts)
            .unwrap()
            .to_device_dtype(&Device::Cpu, DType::F32)
            .unwrap()
    };
    let reference = encode(DType::F32);
    let summary = |cos: &[f64]| {
        let min = cos.iter().copied().fold(f64::INFINITY, f64::min);
        let mean = cos.iter().sum::<f64>() / cos.len() as f64;
        (mean, min)
    };
    let bf16_cos = cosines(&encode(DType::BF16), &reference);
    let (bf16_mean, bf16_min) = summary(&bf16_cos);
    eprintln!(
        "TE-COSINE bf16-in-bf16: {} tokens, mean {bf16_mean:.5}, min {bf16_min:.5}",
        bf16_cos.len()
    );
    if std::env::var("MOLD_TEST_QWEN3_FORCE_DMMV").as_deref() == Ok("1") {
        crate::quantized_dmmv::set_force_dmmv(true);
    }
    let tiers = env_dir("MOLD_QWEN_IMAGE21_TIERS_DIR");
    for (file, _) in GGUF_TIERS {
        let variant = mold_core::manifest::known_qwen3_vl_8b_variants()
            .iter()
            .find(|variant| variant.hf_filename == file)
            .unwrap();
        let mut gguf = Qwen3Encoder::load_gguf(
            &tiers.join(file),
            &tokenizer_path(),
            &device,
            &Qwen3BF16Config::qwen3_image_21_text_encoder(),
        )
        .unwrap();
        let actual = crate::qwen_image21::encode_t2i_prompts(&mut gguf, &prompts)
            .unwrap()
            .to_device_dtype(&Device::Cpu, DType::F32)
            .unwrap();
        let cos = cosines(&actual, &reference);
        let (mean, min) = summary(&cos);
        eprintln!(
            "TE-COSINE {file}: {} tokens, mean {mean:.5}, min {min:.5}; below 0.995 at {:?}",
            cos.len(),
            cos.iter()
                .enumerate()
                .filter(|(_, c)| **c < 0.995)
                .map(|(i, c)| format!("{i}:{c:.4}"))
                .collect::<Vec<_>>()
        );
        // The gate: mean per-token cosine >= 0.995 against the F32 oracle, and
        // no token further from it than the shipped BF16 tier's worst token
        // (a single outlier-activation token sits at ~0.964 for BF16 itself).
        let qualifies = mean >= 0.995 && min >= bf16_min;
        if mold_core::manifest::qwen3_vl_8b_variant_auto_eligible(variant) {
            assert!(
                qualifies,
                "{file}: mean {mean} / worst {min} fail the gate (BF16 worst {bf16_min})"
            );
        } else {
            // An explicit-only variant is explicit-only BECAUSE it fails the
            // gate; if it ever passes, it should join the auto list.
            assert!(
                !qualifies,
                "{file} now qualifies (mean {mean}, worst {min}); make it auto-eligible"
            );
        }
    }
}
/// The GGUF language model's MULTIMODAL forward — interleaved MRoPE over
/// unequal T/H/W axes, the visual rows spliced in, DeepStack after the early
/// layers, and (at two references, ~2.1k tokens) more than one attention
/// chunk — against the BF16 shards on the same reference-conditioned
/// template, under the text test's gate: per-token cosine >= 0.995 mean
/// against the F32 oracle, and no token further from it than the BF16 tier's
/// own worst. The vision tower runs once, in its shipped F32
/// (`reference::vision_tower_dtype`), and every language model receives the
/// same rows — the tower always runs from the BF16 shards whatever the LM
/// tier, so this measures the LM alone.
///
/// Measured on an L40S (2 references, 2,097 tokens, 2,058 visual): the GGUF
/// code path fed UNQUANTIZED weights matches the oracle exactly (cosine
/// 1.000000 on every token), so what remains is quantization. Q8_0 holds the
/// gate on text rows (mean 0.99891, worst 0.99285) but NOT on visual rows
/// (mean 0.99412, worst 0.40068, 146 rows under 0.99) against BF16's own
/// 0.99635 / 0.45722 — this test fails on Q8_0 as it stands, and that is a
/// finding about Q8_0's auto-eligibility for reference-conditioned prompts,
/// not a tolerance to relax. Q4_K_M: mean 0.90302, worst 0.06854.
#[test]
#[ignore = "needs the Qwen3-VL-8B GGUF files and the Qwen Image 2.1 text-encoder shards"]
fn gguf_multimodal_conditioning_tracks_the_bf16_encoder() {
    use crate::qwen_image21::reference::{
        encode_prompt_with_images, encode_vision, load_vision_tower, prepare_reference,
        vision_tower_dtype,
    };
    let device = device();
    let progress = crate::progress::ProgressReporter::default();
    let testdata = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/qwen_image21");
    let references = ["ref_opaque.png", "ref_rgba.png"]
        .iter()
        .map(|name| prepare_reference(&std::fs::read(testdata.join(name)).unwrap()).unwrap())
        .collect::<Vec<_>>();
    let prompt =
        "Change the sky to a warm sunset with orange clouds, keep the house and the sign unchanged.";
    let vision = {
        let tower = load_vision_tower(&shards(), &device, vision_tower_dtype(), &progress).unwrap();
        encode_vision(&tower, &references, &device, &mut || Ok(())).unwrap()
    };
    let rows = vision.embeds.dim(0).unwrap();
    eprintln!(
        "TE-MM {} references, {rows} visual rows, {} DeepStack maps",
        references.len(),
        vision.deepstack.len()
    );
    let bf16_encode = |dtype: DType| {
        let mut encoder = Qwen3Encoder::load_bf16(
            &shards(),
            &tokenizer_path(),
            &device,
            dtype,
            &Qwen3BF16Config::qwen3_image_21_text_encoder(),
            &progress,
        )
        .unwrap();
        encode_prompt_with_images(&mut encoder, &vision, prompt)
            .unwrap()
            .to_device_dtype(&Device::Cpu, DType::F32)
            .unwrap()
    };
    let reference = bf16_encode(DType::F32);
    let tokens = reference.valid_tokens[0].len();
    assert!(
        tokens > 1024,
        "{tokens} tokens do not exercise more than one GGUF attention chunk"
    );
    let summary = |cos: &[f64]| {
        let min = cos.iter().copied().fold(f64::INFINITY, f64::min);
        let mean = cos.iter().sum::<f64>() / cos.len() as f64;
        (mean, min)
    };
    // Per-token cosines split by kind (text vs `<|image_pad|>` rows), with
    // the low tail, so a failure says where the error lives.
    let slots = reference.image_slots[0]
        .iter()
        .zip(&reference.valid_tokens[0])
        .filter(|(_, valid)| **valid)
        .map(|(slot, _)| *slot)
        .collect::<Vec<bool>>();
    let breakdown = |label: &str, cos: &[f64]| {
        for (kind, visual) in [("text", false), ("visual", true)] {
            let mut part = cos
                .iter()
                .zip(&slots)
                .filter(|(_, slot)| **slot == visual)
                .map(|(c, _)| *c)
                .collect::<Vec<f64>>();
            part.sort_by(f64::total_cmp);
            let at = |q: f64| part[((part.len() - 1) as f64 * q) as usize];
            eprintln!(
                "TE-MM-BREAKDOWN {label} {kind}: {} tokens, mean {:.5}, min {:.5}, p1 {:.5}, p5 {:.5}, p50 {:.5}, <0.99: {}",
                part.len(),
                part.iter().sum::<f64>() / part.len() as f64,
                part[0],
                at(0.01),
                at(0.05),
                at(0.5),
                part.iter().filter(|c| **c < 0.99).count()
            );
        }
    };
    let bf16_cos = cosines(&bf16_encode(DType::BF16), &reference);
    let (bf16_mean, bf16_min) = summary(&bf16_cos);
    eprintln!(
        "TE-MM-COSINE bf16-in-bf16: {} tokens, mean {bf16_mean:.5}, min {bf16_min:.5}",
        bf16_cos.len()
    );
    breakdown("bf16-in-bf16", &bf16_cos);
    let mut failures = Vec::new();
    let tiers = env_dir("MOLD_QWEN_IMAGE21_TIERS_DIR");

    // The GGUF CODE PATH fed the BF16 shards' values unquantized (F32
    // QTensors under the Q8 file's own names and header): whatever it moves
    // from the oracle is the path's own arithmetic, not the quantization.
    // Host-side: an F32 `QMatMul` keeps a dense copy beside its QTensor, so
    // 2 x 33 GB does not fit a 46 GB card.
    {
        let path = tiers.join(GGUF_TIERS[0].0);
        let mut reader = std::fs::File::open(&path).unwrap();
        let content = gguf_file::Content::read(&mut reader).unwrap();
        let refs = shards();
        let refs = refs.iter().map(PathBuf::as_path).collect::<Vec<_>>();
        // SAFETY: read-only mapping of verified model files.
        let shards = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&refs) }.unwrap();
        let tensors = content
            .tensor_infos
            .keys()
            .filter(|name| name.starts_with("blk.") || *name == "token_embd.weight")
            .map(|name| {
                let values = shards
                    .load(&hf_name(name).unwrap(), &Device::Cpu)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                let dense = candle_core::quantized::QTensor::quantize(
                    &values,
                    candle_core::quantized::GgmlDType::F32,
                )
                .unwrap();
                (name.clone(), std::sync::Arc::new(dense))
            })
            .collect();
        let model = super::qwen3_gguf::GgufQwen3Encoder::from_parked(
            &(tensors, content.metadata.clone()),
            &Device::Cpu,
        )
        .unwrap();
        let mut encoder = Qwen3Encoder::from_gguf_model(
            model,
            &tokenizer_path(),
            &Device::Cpu,
            &Qwen3BF16Config::qwen3_image_21_text_encoder(),
        )
        .unwrap();
        let actual = encode_prompt_with_images(&mut encoder, &vision, prompt)
            .unwrap()
            .to_device_dtype(&Device::Cpu, DType::F32)
            .unwrap();
        let cos = cosines(&actual, &reference);
        let (mean, min) = summary(&cos);
        eprintln!(
            "TE-MM-COSINE gguf-path-unquantized: {} tokens, mean {mean:.6}, min {min:.6}",
            cos.len()
        );
        breakdown("gguf-path-unquantized", &cos);
        // The path itself must be the oracle's arithmetic (F32 both sides).
        if !(mean >= 0.99999 && min >= 0.999) {
            failures.push(format!(
                "GGUF code path on unquantized weights: mean {mean} / worst {min}"
            ));
        }
    }
    let mut gated = 0;
    for (file, _) in GGUF_TIERS {
        let variant = mold_core::manifest::known_qwen3_vl_8b_variants()
            .iter()
            .find(|variant| variant.hf_filename == file)
            .unwrap();
        let mut gguf = Qwen3Encoder::load_gguf(
            &tiers.join(file),
            &tokenizer_path(),
            &device,
            &Qwen3BF16Config::qwen3_image_21_text_encoder(),
        )
        .unwrap();
        let actual = encode_prompt_with_images(&mut gguf, &vision, prompt)
            .unwrap()
            .to_device_dtype(&Device::Cpu, DType::F32)
            .unwrap();
        assert_eq!(
            actual.image_slots, reference.image_slots,
            "{file} image slots"
        );
        let cos = cosines(&actual, &reference);
        let (mean, min) = summary(&cos);
        let auto = mold_core::manifest::qwen3_vl_8b_variant_auto_eligible(variant);
        eprintln!(
            "TE-MM-COSINE {file} (auto-eligible: {auto}): {} tokens, mean {mean:.5}, min {min:.5}",
            cos.len()
        );
        breakdown(file, &cos);
        // The tiers the encoder picks on its own must hold the text gate on
        // the multimodal path too; an explicit-only tier is reported.
        if auto {
            gated += 1;
            if !(mean >= 0.995 && min >= bf16_min) {
                failures.push(format!(
                    "{file}: multimodal mean {mean} / worst {min} fail the gate (BF16 worst {bf16_min})"
                ));
            }
        }
    }
    assert!(gated > 0, "no auto-eligible GGUF tier was gated");
    assert!(failures.is_empty(), "{failures:#?}");
}
