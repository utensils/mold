//! Synthetic MiniMax H3 FlashAttention release-candidate qualification.
//!
//! This binary is available only with `dev-bins,h3-attention-rc`. It never
//! constructs an inference engine, resolves a model, or opens an artifact.

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use mold_candle::minimax_h3::{
    execute_h3_attention, execute_h3_attention_with_telemetry, H3AttentionDevice,
    H3AttentionModelContract, H3AttentionReleaseCandidateBuild, H3AttentionReleaseRequirement,
    H3AttentionTelemetry, H3AttentionWorkspace, H3FrozenAttentionPlan,
};
use serde::Serialize;

const PROBE_ROWS: usize = 257;
const SHORT_RELEASE_ROWS: usize = 37_296;
const LONG_RELEASE_ROWS: usize = 107_856;
const MAX_ABSOLUTE_DIFFERENCE: f32 = 0.02;
const MIN_QK_SWAP_DIFFERENCE: f32 = MAX_ABSOLUTE_DIFFERENCE;

#[derive(Serialize)]
struct PlannedShape {
    sequence_rows: usize,
    workspace: H3AttentionWorkspace,
}

impl PlannedShape {
    fn from_plan(plan: &H3FrozenAttentionPlan) -> Self {
        Self {
            sequence_rows: plan.rows(),
            workspace: plan.workspace().clone(),
        }
    }
}

#[derive(Serialize)]
struct QualificationReport {
    schema: &'static str,
    decision: &'static str,
    build: H3AttentionReleaseCandidateBuild,
    observed_device: H3AttentionDevice,
    authority_identity_sha256: String,
    probe: H3AttentionTelemetry,
    max_absolute_difference: f32,
    maximum_allowed_difference: f32,
    qk_swap_max_absolute_difference: f32,
    minimum_qk_swap_difference: f32,
    planned_shapes: Vec<PlannedShape>,
    release_activation: &'static str,
    release_requirements: Vec<H3AttentionReleaseRequirement>,
    model_artifacts_accessed: bool,
    runtime_activated: bool,
}

fn deterministic_values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let mut value = (index as u64).wrapping_add(seed);
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            (((value >> 48) as i32 - 32_768) as f32 / 65_536.0) * scale
        })
        .collect()
}

fn bf16_tensor(values: Vec<f32>, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(values, (1, PROBE_ROWS, 56, 128), device)?.to_dtype(DType::BF16)?)
}

fn max_abs_difference(left: &Tensor, right: &Tensor) -> Result<f32> {
    Ok((left - right)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?)
}

fn main() -> Result<()> {
    let cuda = Device::new_cuda(0).context(
        "MiniMax H3 attention qualification requires one visible CUDA device; no CPU fallback is allowed",
    )?;
    let observed_device = H3AttentionDevice::from_candle(&cuda);
    let build = H3AttentionReleaseCandidateBuild::current();
    let authority = build
        .qualify(observed_device)
        .map_err(|error| anyhow!(error))?;
    authority
        .verify_model(H3AttentionModelContract::released_bf16(), &cuda)
        .map_err(|error| anyhow!(error))?;

    let execution = authority
        .freeze_execution(1, 256, PROBE_ROWS)
        .map_err(|error| anyhow!(error))?;
    let cpu_authority =
        mold_candle::minimax_h3::H3AttentionRuntimeAuthority::bounded_synthetic_math(
            H3AttentionDevice::Cpu,
            H3AttentionModelContract::released_bf16(),
        )
        .map_err(|error| anyhow!(error))?;
    let cpu_plan = cpu_authority
        .freeze_execution(1, 256, PROBE_ROWS)
        .map_err(|error| anyhow!(error))?
        .packed_transformer;

    let count = PROBE_ROWS * 56 * 128;
    let q_cpu = bf16_tensor(
        deterministic_values(count, 0x243f_6a88_85a3_08d3, 2.0),
        &Device::Cpu,
    )?;
    let k_cpu = bf16_tensor(
        deterministic_values(count, 0x1319_8a2e_0370_7344, 2.0),
        &Device::Cpu,
    )?;
    let v_cpu = bf16_tensor(
        deterministic_values(count, 0xa409_3822_299f_31d0, 1.0),
        &Device::Cpu,
    )?;
    let expected = execute_h3_attention(&cpu_plan, &q_cpu, &k_cpu, &v_cpu)?.to_dtype(DType::F32)?;
    let qk_swapped =
        execute_h3_attention(&cpu_plan, &k_cpu, &q_cpu, &v_cpu)?.to_dtype(DType::F32)?;
    let qk_swap_max_absolute_difference = max_abs_difference(&expected, &qk_swapped)?;
    if !qk_swap_max_absolute_difference.is_finite()
        || qk_swap_max_absolute_difference <= MIN_QK_SWAP_DIFFERENCE
    {
        return Err(anyhow!(
            "MiniMax H3 qualification vectors cannot detect swapped Q/K wiring: max absolute difference {qk_swap_max_absolute_difference} does not exceed {MIN_QK_SWAP_DIFFERENCE}"
        ));
    }

    let q_cuda = q_cpu.to_device(&cuda)?;
    let k_cuda = k_cpu.to_device(&cuda)?;
    let v_cuda = v_cpu.to_device(&cuda)?;
    let (actual, probe) = execute_h3_attention_with_telemetry(
        &execution.packed_transformer,
        &q_cuda,
        &k_cuda,
        &v_cuda,
    )?;
    let actual = actual.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let max_absolute_difference = max_abs_difference(&expected, &actual)?;
    if !max_absolute_difference.is_finite() || max_absolute_difference > MAX_ABSOLUTE_DIFFERENCE {
        return Err(anyhow!(
            "MiniMax H3 FlashAttention parity failed: max absolute difference {max_absolute_difference} exceeds {MAX_ABSOLUTE_DIFFERENCE}"
        ));
    }

    let planned_shapes = [SHORT_RELEASE_ROWS, LONG_RELEASE_ROWS]
        .into_iter()
        .map(|rows| {
            authority
                .freeze_execution(1, 256, rows)
                .map(|execution| PlannedShape::from_plan(&execution.packed_transformer))
                .map_err(|error| anyhow!(error))
        })
        .collect::<Result<Vec<_>>>()?;
    let release_requirements = authority
        .require_release_activation()
        .expect_err("the H3 release candidate must never activate a shipping runtime")
        .requirements;
    let report = QualificationReport {
        schema: "mold.minimax-h3.attention-qualification.v1",
        decision: "qualified-release-candidate-only",
        build,
        observed_device,
        authority_identity_sha256: authority.identity_sha256().to_string(),
        probe,
        max_absolute_difference,
        maximum_allowed_difference: MAX_ABSOLUTE_DIFFERENCE,
        qk_swap_max_absolute_difference,
        minimum_qk_swap_difference: MIN_QK_SWAP_DIFFERENCE,
        planned_shapes,
        release_activation: "rejected",
        release_requirements,
        model_artifacts_accessed: false,
        runtime_activated: false,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
