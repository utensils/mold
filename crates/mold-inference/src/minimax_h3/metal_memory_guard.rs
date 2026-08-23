//! Live memory-pressure guard for MiniMax H3 on Apple unified memory.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use candle_core::Device;

use crate::progress::InferenceCancellationToken;

const MINIMUM_AVAILABLE_FLOOR_BYTES: u64 = 6 << 30;
const MAXIMUM_ATTEMPT_SWAP_GROWTH_BYTES: u64 = 2 << 30;
const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);

type MemorySampler = Arc<dyn Fn() -> Result<H3MetalMemorySample> + Send + Sync>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct H3MetalMemorySample {
    available_bytes: u64,
    used_swap_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct H3MetalMemoryPolicy {
    minimum_available_bytes: u64,
    baseline_swap_bytes: u64,
    maximum_swap_growth_bytes: u64,
}

impl H3MetalMemoryPolicy {
    fn for_sample(sample: H3MetalMemorySample) -> Self {
        let minimum_available_bytes = minimum_available_floor_bytes(
            crate::device::total_system_memory_bytes().unwrap_or_default(),
        );
        Self {
            minimum_available_bytes,
            baseline_swap_bytes: sample.used_swap_bytes,
            maximum_swap_growth_bytes: MAXIMUM_ATTEMPT_SWAP_GROWTH_BYTES,
        }
    }

    fn violation(self, sample: H3MetalMemorySample) -> Option<String> {
        if sample.available_bytes < self.minimum_available_bytes {
            return Some(format!(
                "MiniMax H3 Metal memory guard stopped inference: {:.1} GiB reclaimable memory remains, below the {:.1} GiB host safety floor",
                sample.available_bytes as f64 / (1_u64 << 30) as f64,
                self.minimum_available_bytes as f64 / (1_u64 << 30) as f64,
            ));
        }
        let swap_growth = sample
            .used_swap_bytes
            .saturating_sub(self.baseline_swap_bytes);
        if swap_growth > self.maximum_swap_growth_bytes {
            return Some(format!(
                "MiniMax H3 Metal memory guard stopped inference: attempt swap grew by {:.1} GiB, above the {:.1} GiB safety limit",
                swap_growth as f64 / (1_u64 << 30) as f64,
                self.maximum_swap_growth_bytes as f64 / (1_u64 << 30) as f64,
            ));
        }
        None
    }
}

fn minimum_available_floor_bytes(total_system_bytes: u64) -> u64 {
    // Preserve at least 15% on larger unified-memory Macs while allowing the
    // allocator's short transformer-load crest to pass on 48 GiB machines.
    // The absolute 6 GiB floor still protects smaller supported systems.
    (total_system_bytes.saturating_mul(15) / 100).max(MINIMUM_AVAILABLE_FLOOR_BYTES)
}

fn sample_metal_memory() -> Result<H3MetalMemorySample> {
    let available_bytes = crate::device::available_system_memory_bytes()
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| anyhow!("macOS available-memory sample is unavailable"))?;
    let used_swap_bytes = crate::device::used_system_swap_bytes()
        .ok_or_else(|| anyhow!("macOS swap sample is unavailable"))?;
    Ok(H3MetalMemorySample {
        available_bytes,
        used_swap_bytes,
    })
}

/// Attempt-scoped watchdog. It cancels through the same token the pipeline
/// already polls, so model objects still unwind on their owner thread.
pub(crate) struct H3MetalMemoryGuard {
    stop: Arc<AtomicBool>,
    violation: Arc<Mutex<Option<String>>>,
    worker: Option<JoinHandle<()>>,
}

impl H3MetalMemoryGuard {
    pub(crate) fn start(device: &Device, cancellation: InferenceCancellationToken) -> Result<Self> {
        if !device.is_metal() {
            return Ok(Self {
                stop: Arc::new(AtomicBool::new(true)),
                violation: Arc::new(Mutex::new(None)),
                worker: None,
            });
        }

        let initial = sample_metal_memory()?;
        let policy = H3MetalMemoryPolicy::for_sample(initial);
        if let Some(message) = policy.violation(initial) {
            bail!(message);
        }
        Self::spawn(
            policy,
            cancellation,
            Arc::new(sample_metal_memory),
            SAMPLE_INTERVAL,
        )
    }

    fn spawn(
        policy: H3MetalMemoryPolicy,
        cancellation: InferenceCancellationToken,
        sampler: MemorySampler,
        sample_interval: Duration,
    ) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let violation = Arc::new(Mutex::new(None));
        let worker_stop = stop.clone();
        let worker_violation = violation.clone();
        let worker = thread::Builder::new()
            .name("mold-h3-metal-memory-guard".into())
            .spawn(move || {
                while !worker_stop.load(Ordering::Acquire) {
                    thread::sleep(sample_interval);
                    if worker_stop.load(Ordering::Acquire) {
                        break;
                    }
                    let message = match sampler() {
                        Ok(sample) => policy.violation(sample),
                        Err(error) => Some(format!(
                            "MiniMax H3 Metal memory guard stopped inference because pressure sampling failed: {error}"
                        )),
                    };
                    if let Some(message) = message {
                        *worker_violation
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(message);
                        cancellation.cancel();
                        break;
                    }
                }
            })?;
        Ok(Self {
            stop,
            violation,
            worker: Some(worker),
        })
    }

    pub(crate) fn finish(mut self) -> Result<Option<String>> {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| anyhow!("MiniMax H3 Metal memory guard thread panicked"))?;
        }
        Ok(self
            .violation
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone())
    }
}

impl Drop for H3MetalMemoryGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gib(value: u64) -> u64 {
        value << 30
    }

    #[test]
    fn guard_rejects_floor_and_attempt_swap_growth_independently() {
        let policy = H3MetalMemoryPolicy {
            minimum_available_bytes: gib(8),
            baseline_swap_bytes: gib(10),
            maximum_swap_growth_bytes: gib(2),
        };
        assert!(policy
            .violation(H3MetalMemorySample {
                available_bytes: gib(7),
                used_swap_bytes: gib(10),
            })
            .unwrap()
            .contains("host safety floor"));
        assert!(policy
            .violation(H3MetalMemorySample {
                available_bytes: gib(12),
                used_swap_bytes: gib(13),
            })
            .unwrap()
            .contains("attempt swap grew"));
        assert_eq!(
            policy.violation(H3MetalMemorySample {
                available_bytes: gib(12),
                used_swap_bytes: gib(12),
            }),
            None
        );
    }

    #[test]
    fn guard_floor_is_six_gib_or_fifteen_percent_whichever_is_larger() {
        assert_eq!(minimum_available_floor_bytes(gib(32)), gib(6));
        let forty_eight_gib_floor = minimum_available_floor_bytes(gib(48));
        assert!(forty_eight_gib_floor > gib(7));
        assert!(forty_eight_gib_floor < gib(8));
        assert_eq!(minimum_available_floor_bytes(gib(64)), gib(64) * 15 / 100);
    }

    #[test]
    fn non_metal_guard_starts_without_a_macos_memory_sampler() {
        let guard =
            H3MetalMemoryGuard::start(&Device::Cpu, InferenceCancellationToken::default()).unwrap();
        assert_eq!(guard.finish().unwrap(), None);
    }

    #[test]
    fn watchdog_cancels_the_attempt_when_pressure_crosses_the_floor() {
        let policy = H3MetalMemoryPolicy {
            minimum_available_bytes: gib(8),
            baseline_swap_bytes: gib(10),
            maximum_swap_growth_bytes: gib(2),
        };
        let cancellation = InferenceCancellationToken::default();
        let observer = cancellation.clone();
        let guard = H3MetalMemoryGuard::spawn(
            policy,
            cancellation,
            Arc::new(|| {
                Ok(H3MetalMemorySample {
                    available_bytes: gib(7),
                    used_swap_bytes: gib(10),
                })
            }),
            Duration::from_millis(1),
        )
        .unwrap();

        for _ in 0..1_000 {
            if observer.is_cancelled() {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        assert!(
            observer.is_cancelled(),
            "watchdog did not cancel the attempt"
        );
        assert!(guard
            .finish()
            .unwrap()
            .unwrap()
            .contains("host safety floor"));
    }
}
