//! Developer benchmark for GGUF load throughput.
//!
//! Not built by default: `required-features = ["dev-bins", "cuda"]`.
//!
//! ```bash
//! CUDA_VISIBLE_DEVICES=3 cargo run --release -p mold-ai-candle \
//!   --features dev-bins,cuda --bin gguf_load_bench -- \
//!   /storage/mold/models/flux-dev-q8/flux1-dev-Q8_0.gguf stats mmap staged
//! ```

use candle::cuda::cudarc;
use candle::quantized::gguf_file;
use candle::{Device, Result};
use mold_candle::gguf_mmap::GgufMmap;
use std::os::unix::fs::FileExt;
use std::time::{Duration, Instant};

/// Minor and major fault counters for this process.
///
/// `comm` (field 2) can contain spaces and parentheses, so the tail is taken
/// from the LAST `)`, after which the fields are `state ppid pgrp session tty
/// tpgid flags minflt cminflt majflt`.
fn faults() -> (u64, u64) {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let Some(idx) = stat.rfind(')') else {
        return (0, 0);
    };
    let fields: Vec<&str> = stat[idx + 1..].split_whitespace().collect();
    let minflt = fields.get(7).and_then(|v| v.parse().ok()).unwrap_or(0);
    let majflt = fields.get(9).and_then(|v| v.parse().ok()).unwrap_or(0);
    (minflt, majflt)
}

fn report(label: &str, bytes: u64, elapsed: Duration, faults: (u64, u64)) {
    let secs = elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    println!(
        "{label:<16} {:>7.2} GB {:>8} ms {:>7.2} GB/s  minflt={:<10} majflt={}",
        bytes as f64 / 1e9,
        elapsed.as_millis(),
        bytes as f64 / 1e9 / secs,
        faults.0,
        faults.1,
    );
}

/// Run `body`, timing it and differencing the fault counters.
fn timed<T>(label: &str, bytes: u64, body: impl FnOnce() -> Result<T>) -> Result<T> {
    let before = faults();
    let started = Instant::now();
    let out = body()?;
    let elapsed = started.elapsed();
    let after = faults();
    report(
        label,
        bytes,
        elapsed,
        (
            after.0.saturating_sub(before.0),
            after.1.saturating_sub(before.1),
        ),
    );
    Ok(out)
}

/// Tensor names in ascending file order.
fn ordered(map: &GgufMmap) -> Vec<String> {
    let infos = map.tensor_infos();
    let mut names: Vec<String> = infos.keys().cloned().collect();
    names.sort_by_key(|name| infos[name].offset);
    names
}

fn payload_len(info: &gguf_file::TensorInfo) -> usize {
    info.shape.elem_count() / info.ggml_dtype.block_size() * info.ggml_dtype.type_size()
}

fn wrap(e: cudarc::driver::DriverError) -> candle::Error {
    candle::Error::Msg(e.to_string())
}

/// Staging buffer size for the pinned variants.
const STAGE: usize = 256 << 20;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: gguf_load_bench <file> [variant...]");
    let owned: Vec<String> = args.collect();
    let variants: Vec<&str> = if owned.is_empty() {
        vec!["stats", "mmap"]
    } else {
        owned.iter().map(String::as_str).collect()
    };

    let device = Device::new_cuda(0)?;
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let ctx = stream.context().clone();

    let map = GgufMmap::open(&path)?;
    let total = map.total_tensor_bytes();
    let names = ordered(&map);
    let sizes: Vec<usize> = names
        .iter()
        .map(|n| payload_len(&map.tensor_infos()[n]))
        .collect();
    let largest = sizes.iter().copied().max().unwrap_or(0);

    println!(
        "file={path}\n  tensors={}  payload={:.2} GB  largest={:.1} MB  mean={:.2} MB",
        names.len(),
        total as f64 / 1e9,
        largest as f64 / 1e6,
        total as f64 / 1e6 / names.len().max(1) as f64,
    );

    for variant in variants {
        match variant {
            "stats" => {
                let mut hist = [0usize; 6];
                for size in &sizes {
                    let bucket = match size {
                        0..=65_535 => 0,
                        65_536..=1_048_575 => 1,
                        1_048_576..=8_388_607 => 2,
                        8_388_608..=67_108_863 => 3,
                        67_108_864..=536_870_911 => 4,
                        _ => 5,
                    };
                    hist[bucket] += 1;
                }
                println!(
                    "  histogram  <64K={} <1M={} <8M={} <64M={} <512M={} bigger={}",
                    hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]
                );
            }
            // Host-side fault cost: one byte read per 4 KiB page, no upload.
            "touch" => {
                timed("touch", total, || {
                    let mut acc = 0u64;
                    for name in &names {
                        for page in map.tensor_bytes(name)?.chunks(4096) {
                            acc = acc.wrapping_add(page[0] as u64);
                        }
                    }
                    std::hint::black_box(acc);
                    Ok(())
                })?;
            }
            // memcpy every tensor into a reused PAGEABLE buffer.
            "copy_pageable" => {
                let mut buf = vec![0u8; STAGE];
                timed("copy_pageable", total, || {
                    for (name, len) in names.iter().zip(&sizes) {
                        if *len > STAGE {
                            continue;
                        }
                        buf[..*len].copy_from_slice(map.tensor_bytes(name)?);
                    }
                    Ok(())
                })?;
            }
            // memcpy every tensor into a reused PINNED write-combined buffer.
            "copy_pinned" => {
                let mut pinned = unsafe { ctx.alloc_pinned::<u8>(STAGE) }.map_err(wrap)?;
                timed("copy_pinned", total, || {
                    let dst = pinned.as_mut_slice().map_err(wrap)?;
                    for (name, len) in names.iter().zip(&sizes) {
                        if *len > STAGE {
                            continue;
                        }
                        dst[..*len].copy_from_slice(map.tensor_bytes(name)?);
                    }
                    Ok(())
                })?;
            }
            // PCIe ceiling from pinned memory.
            "h2d_pinned" => {
                let pinned = unsafe { ctx.alloc_pinned::<u8>(STAGE) }.map_err(wrap)?;
                let mut dev = cuda.alloc_zeros::<u8>(STAGE)?;
                let reps = (total as usize).div_ceil(STAGE);
                timed("h2d_pinned", (reps * STAGE) as u64, || {
                    for _ in 0..reps {
                        stream.memcpy_htod(&pinned, &mut dev).map_err(wrap)?;
                    }
                    stream.synchronize().map_err(wrap)?;
                    Ok(())
                })?;
            }
            // The same upload from an ALREADY FAULTED pageable buffer: this is
            // the driver's own pageable staging rate, with no fault cost.
            "h2d_pageable" => {
                let buf = vec![7u8; STAGE];
                let mut dev = cuda.alloc_zeros::<u8>(STAGE)?;
                let reps = (total as usize).div_ceil(STAGE);
                timed("h2d_pageable", (reps * STAGE) as u64, || {
                    for _ in 0..reps {
                        stream.memcpy_htod(buf.as_slice(), &mut dev).map_err(wrap)?;
                    }
                    stream.synchronize().map_err(wrap)?;
                    Ok(())
                })?;
            }
            // The pre-staging loader: every tensor read from the mapping.
            "mmap" => {
                let tensors = timed("mmap", total, || {
                    map.load_all_from_mapping(&device, &mut |_, _| {})
                })?;
                drop(tensors);
            }
            // Byte equality of the two paths on a REAL checkpoint, on the CPU
            // so the comparison reads host storage directly.
            "verify" => {
                let staged = timed("verify(staged)", total, || {
                    map.load_all_staged(&Device::Cpu, &mut |_, _| {}, STAGE)
                })?;
                let mapped = timed("verify(mapped)", total, || {
                    map.load_all_from_mapping(&Device::Cpu, &mut |_, _| {})
                })?;
                assert_eq!(staged.len(), mapped.len());
                let mut checked = 0usize;
                for (name, want) in &mapped {
                    let have = &staged[name];
                    assert_eq!(have.shape(), want.shape(), "{name}");
                    assert_eq!(have.dtype(), want.dtype(), "{name}");
                    assert_eq!(have.data()?, want.data()?, "{name}");
                    checked += 1;
                }
                println!("                 {checked} tensors byte-identical");
            }
            // The same check against what actually landed in DEVICE memory.
            // This is the one that catches a staging buffer refilled while an
            // upload out of it was still in flight; the CPU comparison cannot,
            // because there the copy is synchronous.
            "verify_cuda" => {
                let staged = timed("verify_cuda", total, || {
                    map.load_all(&device, &mut |_, _| {})
                })?;
                let mut checked = 0usize;
                for name in &names {
                    let want = map.tensor_bytes(name)?;
                    let have = staged[name].data()?;
                    assert_eq!(have.len(), want.len(), "{name}");
                    assert_eq!(&have[..], want, "{name}");
                    checked += 1;
                }
                println!("                 {checked} device tensors byte-identical to the file");
            }
            // The shipping loader: staged parallel read + page-locked upload.
            "load_all" => {
                let tensors = timed("load_all", total, || map.load_all(&device, &mut |_, _| {}))?;
                drop(tensors);
            }
            // MADV_WILLNEED over the whole mapping, then the shipping loader.
            "mmap_willneed" => {
                timed("  (willneed)", total, || {
                    map.advise_will_need()?;
                    Ok(())
                })?;
                let tensors = timed("mmap_willneed", total, || {
                    map.load_all_from_mapping(&device, &mut |_, _| {})
                })?;
                drop(tensors);
            }
            // pread every tensor into a reused PAGEABLE buffer, no upload:
            // the host read-path ceiling.
            "pread_pageable" => {
                let file = std::fs::File::open(&path)?;
                let mut buf = vec![0u8; STAGE];
                timed("pread_pageable", total, || {
                    for (name, len) in names.iter().zip(&sizes) {
                        if *len > STAGE {
                            continue;
                        }
                        let info = &map.tensor_infos()[name];
                        let at = map.content().tensor_data_offset + info.offset;
                        file.read_exact_at(&mut buf[..*len], at)?;
                    }
                    Ok(())
                })?;
            }
            // pread every tensor into a reused PINNED buffer, no upload.
            "pread_pinned_host" => {
                let file = std::fs::File::open(&path)?;
                let mut pinned = unsafe { ctx.alloc_pinned::<u8>(STAGE) }.map_err(wrap)?;
                timed("pread_pinned_host", total, || {
                    let dst = pinned.as_mut_slice().map_err(wrap)?;
                    for (name, len) in names.iter().zip(&sizes) {
                        if *len > STAGE {
                            continue;
                        }
                        let info = &map.tensor_infos()[name];
                        let at = map.content().tensor_data_offset + info.offset;
                        file.read_exact_at(&mut dst[..*len], at)?;
                    }
                    Ok(())
                })?;
            }
            // Full load: pread into one pinned buffer, then upload, serialized.
            "pread_pinned" => {
                let file = std::fs::File::open(&path)?;
                let mut pinned = unsafe { ctx.alloc_pinned::<u8>(STAGE) }.map_err(wrap)?;
                let mut read = Duration::ZERO;
                let mut upload = Duration::ZERO;
                let tensors = timed("pread_pinned", total, || {
                    let mut out = Vec::with_capacity(names.len());
                    for (name, len) in names.iter().zip(&sizes) {
                        let info = &map.tensor_infos()[name];
                        let dims = info.shape.dims().to_vec();
                        if *len > STAGE {
                            out.push(candle::quantized::ggml_file::qtensor_from_ggml(
                                info.ggml_dtype,
                                map.tensor_bytes(name)?,
                                dims,
                                &device,
                            )?);
                            continue;
                        }
                        let at = map.content().tensor_data_offset + info.offset;
                        let t0 = Instant::now();
                        let dst = pinned.as_mut_slice().map_err(wrap)?;
                        file.read_exact_at(&mut dst[..*len], at)?;
                        read += t0.elapsed();

                        let t1 = Instant::now();
                        // SAFETY: `dst` is the pinned allocation, alive across
                        // the upload, which the synchronize below completes.
                        let staged: &[u8] =
                            unsafe { std::slice::from_raw_parts(dst.as_ptr(), *len) };
                        out.push(candle::quantized::ggml_file::qtensor_from_ggml(
                            info.ggml_dtype,
                            staged,
                            dims,
                            &device,
                        )?);
                        stream.synchronize().map_err(wrap)?;
                        upload += t1.elapsed();
                    }
                    Ok(out)
                })?;
                println!(
                    "                 read {:>6} ms ({:>5.2} GB/s)   upload {:>6} ms ({:>5.2} GB/s)",
                    read.as_millis(),
                    total as f64 / 1e9 / read.as_secs_f64().max(f64::MIN_POSITIVE),
                    upload.as_millis(),
                    total as f64 / 1e9 / upload.as_secs_f64().max(f64::MIN_POSITIVE),
                );
                drop(tensors);
            }
            // The shipping loader over a MAP_POPULATE mapping.
            "mmap_populate" => {
                let populated = timed("  (populate)", total, || GgufMmap::open_populated(&path))?;
                let tensors = timed("mmap_populate", total, || {
                    populated.load_all_from_mapping(&device, &mut |_, _| {})
                })?;
                drop(tensors);
            }
            // One pinned staging buffer, memcpy then upload, serialized so the
            // two phases can be attributed separately.
            "staged" => {
                let mut pinned = unsafe { ctx.alloc_pinned::<u8>(STAGE) }.map_err(wrap)?;
                let mut copy = Duration::ZERO;
                let mut upload = Duration::ZERO;
                let tensors = timed("staged", total, || {
                    let mut out = Vec::with_capacity(names.len());
                    for (name, len) in names.iter().zip(&sizes) {
                        let info = &map.tensor_infos()[name];
                        let src = map.tensor_bytes(name)?;
                        if *len > STAGE {
                            out.push(candle::quantized::ggml_file::qtensor_from_ggml(
                                info.ggml_dtype,
                                src,
                                info.shape.dims().to_vec(),
                                &device,
                            )?);
                            continue;
                        }
                        let t0 = Instant::now();
                        let dst = pinned.as_mut_slice().map_err(wrap)?;
                        dst[..*len].copy_from_slice(src);
                        copy += t0.elapsed();

                        let t1 = Instant::now();
                        // SAFETY: `dst` is the pinned allocation, alive for the
                        // whole upload, which the synchronize below completes.
                        let staged: &[u8] =
                            unsafe { std::slice::from_raw_parts(dst.as_ptr(), *len) };
                        out.push(candle::quantized::ggml_file::qtensor_from_ggml(
                            info.ggml_dtype,
                            staged,
                            info.shape.dims().to_vec(),
                            &device,
                        )?);
                        stream.synchronize().map_err(wrap)?;
                        upload += t1.elapsed();
                    }
                    Ok(out)
                })?;
                println!(
                    "                 copy {:>6} ms ({:>5.2} GB/s)   upload {:>6} ms ({:>5.2} GB/s)",
                    copy.as_millis(),
                    total as f64 / 1e9 / copy.as_secs_f64().max(f64::MIN_POSITIVE),
                    upload.as_millis(),
                    total as f64 / 1e9 / upload.as_secs_f64().max(f64::MIN_POSITIVE),
                );
                drop(tensors);
            }
            other => println!("unknown variant {other}"),
        }
    }
    Ok(())
}
