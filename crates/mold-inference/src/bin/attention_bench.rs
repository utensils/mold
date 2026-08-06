//! A/B the attention backends at the shapes Mold's engines actually produce.
//!
//! Issue #598 asked for numbers behind the `flash-attn` feature. The shapes
//! below come from the engines themselves, not from round figures:
//! `[batch, heads, seq, head_dim]`, where `seq` is the fused `[txt; img]`
//! sequence for the image families.
//!
//! Only four call sites route through `crate::attention` — FLUX's offload and
//! GGUF-bypass transformers, all of Flux.2, and Z-Image's offload transformer.
//! Dense BF16 FLUX and Z-Image use upstream Candle's own attention and are
//! unaffected by this feature, so the FLUX rows here describe the offload path.
//!
//! ```text
//! cargo run --release -p mold-ai-inference --features dev-bins,flash-attn \
//!     --bin attention_bench
//! ```
//!
//! Without `flash-attn` the "flash" column measures the math fallback, which
//! is a useful control: the two columns should then be identical.

use candle_core::{DType, Device, Tensor};
use mold_inference::attention::{flash_attention, math_attention, AttentionBackend};
use std::time::Instant;

/// `(label, batch, heads, seq, head_dim)`.
const SHAPES: &[(&str, usize, usize, usize, usize)] = &[
    ("FLUX offload 768^2   ", 1, 24, 2560, 128),
    ("FLUX offload 1024^2  ", 1, 24, 4352, 128),
    ("FLUX offload 1536^2  ", 1, 24, 9472, 128),
    ("Flux.2 klein 1024^2  ", 1, 24, 4150, 128),
    ("Flux.2 dev 1024^2    ", 1, 48, 4150, 128),
    ("Z-Image offload 1024^2", 1, 30, 4160, 128),
    ("LTX-2 stage2 default ", 1, 32, 10868, 128),
    ("LTX-2 stage2 1024^2  ", 1, 32, 13312, 128),
];

const WARMUP: usize = 3;
const ITERS: usize = 10;

fn main() -> anyhow::Result<()> {
    let device = Device::new_cuda(0).map_err(|e| {
        anyhow::anyhow!("this benchmark needs a CUDA device (flash-attention is CUDA-only): {e}")
    })?;
    let dtype = DType::BF16;

    println!("backend resolved to: {:?}", AttentionBackend::resolve());
    println!(
        "dtype {dtype:?}, {WARMUP} warmup + {ITERS} timed iterations per shape\n\
         chunk policy for the math column is the production default (auto)\n"
    );
    println!(
        "{:<22} {:>12} {:>12} {:>9} {:>12}",
        "shape", "math (ms)", "flash (ms)", "speedup", "max |diff|"
    );

    for &(label, b, h, n, d) in SHAPES {
        let q = Tensor::randn(0f32, 1.0, (b, h, n, d), &device)?.to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1.0, (b, h, n, d), &device)?.to_dtype(dtype)?;
        let v = Tensor::randn(0f32, 1.0, (b, h, n, d), &device)?.to_dtype(dtype)?;
        let scale = 1.0 / (d as f64).sqrt() as f32;

        let math_ms = time(&device, || math_attention(&q, &k, &v, scale))?;
        let flash_ms = time(&device, || flash_attention(&q, &k, &v, scale))?;

        // Correctness alongside speed: FA2 keeps an fp32 online-softmax
        // accumulator while the math path reduces in the input dtype, so these
        // agree to bf16 precision, not to 1e-5.
        let a = math_attention(&q, &k, &v, scale)?.to_dtype(DType::F32)?;
        let c = flash_attention(&q, &k, &v, scale)?.to_dtype(DType::F32)?;
        let diff = (a - c)?.abs()?.max_all()?.to_scalar::<f32>()?;

        println!(
            "{label:<22} {math_ms:>12.2} {flash_ms:>12.2} {:>8.2}x {diff:>12.2e}",
            math_ms / flash_ms
        );
    }

    Ok(())
}

/// Median wall time in milliseconds. `synchronize` before and after each
/// sample, or CUDA's async launch queue makes every kernel look free.
fn time<F>(device: &Device, mut op: F) -> anyhow::Result<f64>
where
    F: FnMut() -> candle_core::Result<Tensor>,
{
    for _ in 0..WARMUP {
        op()?;
    }
    device.synchronize()?;

    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        let out = op()?;
        device.synchronize()?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        drop(out);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).expect("wall-clock samples are finite"));
    Ok(samples[samples.len() / 2])
}
