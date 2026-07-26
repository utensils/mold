use std::fs;
use std::path::{Path, PathBuf};

fn rust_sources_below(root: &Path, sources: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).expect("read source directory") {
        let path = entry.expect("read source entry").path();
        if path.is_dir() {
            rust_sources_below(&path, sources);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            sources.push(path);
        }
    }
}

fn code_without_line_comments(source: &str) -> String {
    source
        .lines()
        .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn normal_runtime_has_no_primary_context_reset_or_reclaim_api() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("inference crate is below workspace crates directory");
    let mut sources = Vec::new();
    rust_sources_below(&workspace.join("crates/mold-inference/src"), &mut sources);
    rust_sources_below(&workspace.join("crates/mold-server/src"), &mut sources);

    let reset_symbol = ["cuDevicePrimaryCtx", "Reset_v2"].concat();
    let reclaim_symbol = ["reclaim_gpu", "_memory"].concat();
    let mut violations = Vec::new();
    for path in sources {
        let source = fs::read_to_string(&path).expect("read Rust source");
        let code = code_without_line_comments(&source);
        if code.contains(&reset_symbol) || code.contains(&reclaim_symbol) {
            violations.push(
                path.strip_prefix(workspace)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
            );
        }
    }

    assert!(
        violations.is_empty(),
        "normal runtime must be reset-free; forbidden references remain in:\n{}",
        violations.join("\n")
    );
}

#[cfg(feature = "cuda")]
#[test]
fn post_drop_sampling_preserves_live_cuda_handles() {
    use candle_core::{Device, Tensor};
    use mold_inference::device;

    if !candle_core::utils::cuda_is_available() {
        return;
    }

    let cuda = Device::new_cuda(0).expect("open CUDA device 0");
    let live =
        Tensor::from_slice(&[1.0_f32, 2.0, 3.0], 3, &cuda).expect("allocate live CUDA tensor");
    device::init_thread_gpu_ordinal(0);
    let sampled = device::post_drop_free_vram_bytes(0);
    device::clear_thread_gpu_ordinal();

    assert!(sampled.is_some(), "CUDA free-memory sample must succeed");
    let values = live
        .affine(2.0, 1.0)
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .expect("live tensor remains usable after post-drop sampling");
    assert_eq!(values, vec![3.0, 5.0, 7.0]);
}

#[cfg(feature = "cuda")]
#[test]
fn sampling_one_gpu_preserves_sibling_gpu_handles() {
    use candle_core::cuda_backend::cudarc::driver::CudaContext;
    use candle_core::{Device, Tensor};
    use mold_inference::device;

    if CudaContext::device_count().unwrap_or(0) < 2 {
        return;
    }

    let cuda0 = Device::new_cuda(0).expect("open CUDA device 0");
    let cuda1 = Device::new_cuda(1).expect("open CUDA device 1");
    let live0 = Tensor::from_slice(&[2.0_f32], 1, &cuda0).expect("GPU 0 tensor");
    let live1 = Tensor::from_slice(&[4.0_f32], 1, &cuda1).expect("GPU 1 tensor");

    device::init_thread_gpu_ordinal(0);
    assert!(device::post_drop_free_vram_bytes(0).is_some());
    device::clear_thread_gpu_ordinal();

    assert_eq!(
        live0.to_vec1::<f32>().expect("GPU 0 handle survives"),
        vec![2.0]
    );
    assert_eq!(
        live1.to_vec1::<f32>().expect("GPU 1 handle survives"),
        vec![4.0]
    );
}
