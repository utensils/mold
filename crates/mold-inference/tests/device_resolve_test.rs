//! Coverage for `mold_inference::device::resolve_device`. Runs as an
//! integration test — keeps the test binary small and avoids pulling candle's
//! full model-loading stack.

use mold_core::types::DeviceRef;
use mold_inference::device::resolve_device;

fn cpu_auto() -> anyhow::Result<candle_core::Device> {
    Ok(candle_core::Device::Cpu)
}

#[test]
fn resolve_device_none_calls_auto() {
    let dev = resolve_device(None, cpu_auto).unwrap();
    assert!(matches!(dev, candle_core::Device::Cpu));
}

#[test]
fn resolve_device_auto_calls_auto() {
    let dev = resolve_device(Some(DeviceRef::Auto), cpu_auto).unwrap();
    assert!(matches!(dev, candle_core::Device::Cpu));
}

#[test]
fn resolve_device_cpu_bypasses_auto() {
    let called = std::sync::atomic::AtomicBool::new(false);
    let dev = resolve_device(Some(DeviceRef::Cpu), || {
        called.store(true, std::sync::atomic::Ordering::SeqCst);
        cpu_auto()
    })
    .unwrap();
    assert!(matches!(dev, candle_core::Device::Cpu));
    assert!(
        !called.load(std::sync::atomic::Ordering::SeqCst),
        "Cpu override must not invoke the auto closure"
    );
}

#[test]
#[cfg(not(any(feature = "cuda", feature = "metal")))]
fn resolve_device_gpu_on_cpu_only_host_errors_clearly() {
    let err = resolve_device(Some(DeviceRef::gpu(0)), cpu_auto).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("GPU") || msg.contains("cuda") || msg.contains("metal"),
        "error should mention the missing backend, got: {msg}"
    );
}
