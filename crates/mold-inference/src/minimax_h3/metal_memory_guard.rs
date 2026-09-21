//! Live memory-pressure guard for MiniMax H3 on Apple unified memory.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use candle_core::Device;

use crate::progress::InferenceCancellationToken;

const MINIMUM_AVAILABLE_FLOOR_BYTES: u64 = 8 << 30;
const MAXIMUM_ATTEMPT_SWAP_GROWTH_BYTES: u64 = 2 << 30;
const CAMPAIGN_MINIMUM_AVAILABLE_FLOOR_BYTES: u64 = 12 << 30;
const CAMPAIGN_MAXIMUM_SWAP_GROWTH_BYTES: u64 = 256 << 20;
const SAMPLE_INTERVAL: Duration = Duration::from_millis(250);

type MemorySampler = Arc<dyn Fn() -> Result<H3MetalMemorySample> + Send + Sync>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct H3MetalMemorySample {
    pub(crate) available_bytes: u64,
    pub(crate) used_swap_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct H3MetalMemoryPolicy {
    minimum_available_bytes: u64,
    baseline_swap_bytes: u64,
    maximum_swap_growth_bytes: u64,
    /// The derived native-allocation ceiling the campaign requires. Production
    /// runs with `None` and never samples the allocator; campaign runs derive
    /// the ceiling from current capacity and headroom and refuse to launch
    /// without one.
    maximum_native_bytes: Option<u64>,
}

impl H3MetalMemoryPolicy {
    fn for_sample(sample: H3MetalMemorySample) -> Self {
        Self {
            minimum_available_bytes: MINIMUM_AVAILABLE_FLOOR_BYTES,
            baseline_swap_bytes: sample.used_swap_bytes,
            maximum_swap_growth_bytes: MAXIMUM_ATTEMPT_SWAP_GROWTH_BYTES,
            maximum_native_bytes: None,
        }
    }

    /// The tighter policy the H3 Metal default-resolution campaign requires.
    ///
    /// This is not an output-semantics switch: it only narrows the same live
    /// host-floor and swap-growth invariants that the shipped guard already
    /// checks, and adds the derived native-allocation ceiling. Campaign
    /// evidence may therefore be compared against the shipped guard without
    /// introducing a second runtime authority.
    fn for_campaign_sample(sample: H3MetalMemorySample, maximum_native_bytes: Option<u64>) -> Self {
        Self {
            minimum_available_bytes: CAMPAIGN_MINIMUM_AVAILABLE_FLOOR_BYTES,
            baseline_swap_bytes: sample.used_swap_bytes,
            maximum_swap_growth_bytes: CAMPAIGN_MAXIMUM_SWAP_GROWTH_BYTES,
            maximum_native_bytes,
        }
    }

    fn violation(
        self,
        sample: H3MetalMemorySample,
        native_allocated_bytes: Option<u64>,
    ) -> Option<String> {
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
        if let (Some(ceiling), Some(native)) = (self.maximum_native_bytes, native_allocated_bytes) {
            if native > ceiling {
                return Some(format!(
                    "MiniMax H3 Metal memory guard stopped inference: native allocation {:.2} GiB exceeded the {:.2} GiB ceiling",
                    native as f64 / (1_u64 << 30) as f64,
                    ceiling as f64 / (1_u64 << 30) as f64,
                ));
            }
        }
        None
    }
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
        if let Some(message) = policy.violation(initial, None) {
            bail!(message);
        }
        Self::spawn(
            policy,
            cancellation,
            Arc::new(sample_metal_memory),
            SAMPLE_INTERVAL,
        )
    }

    /// Start the attempt-scoped Metal guard with the campaign-only thresholds.
    ///
    /// The default (shipped) guard remains the production default. The
    /// campaign policy is opt-in through `MOLD_H3_METAL_CAMPAIGN=1` and is
    /// deliberately narrower than the shipped guard; it must never be used to
    /// relax the production invariants.
    pub(crate) fn start_campaign(
        device: &Device,
        cancellation: InferenceCancellationToken,
    ) -> Result<Self> {
        if !device.is_metal() {
            return Ok(Self {
                stop: Arc::new(AtomicBool::new(true)),
                violation: Arc::new(Mutex::new(None)),
                worker: None,
            });
        }

        let ceiling = super::campaign_capture::campaign_ceiling_bytes()?;
        let initial = sample_metal_memory()?;
        let policy = H3MetalMemoryPolicy::for_campaign_sample(initial, Some(ceiling));
        let native = super::campaign_capture::native_allocated_bytes()?;
        if let Some(message) = policy.violation(initial, Some(native)) {
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
                        Ok(sample) => {
                            super::campaign_capture::record_memory_sample();
                            let native = if policy.maximum_native_bytes.is_some() {
                                match super::campaign_capture::native_allocated_bytes() {
                                    Ok(native) => policy.violation(sample, Some(native)),
                                    Err(error) => Some(format!("MiniMax H3 Metal memory guard stopped inference because native allocation sampling failed: {error}")),
                                }
                            } else {
                                policy.violation(sample, None)
                            };
                            native
                        }
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
    fn campaign_policy_rejects_stricter_floor_and_swap_growth() {
        let policy = H3MetalMemoryPolicy::for_campaign_sample(
            H3MetalMemorySample {
                available_bytes: gib(20),
                used_swap_bytes: gib(10),
            },
            Some(gib(8)),
        );
        assert_eq!(policy.minimum_available_bytes, gib(12));
        assert_eq!(policy.maximum_swap_growth_bytes, 256 << 20);
        assert_eq!(policy.maximum_native_bytes, Some(gib(8)));
        assert!(policy
            .violation(
                H3MetalMemorySample {
                    available_bytes: gib(11),
                    used_swap_bytes: gib(10),
                },
                Some(gib(1)),
            )
            .unwrap()
            .contains("host safety floor"));
        assert!(policy
            .violation(
                H3MetalMemorySample {
                    available_bytes: gib(12),
                    used_swap_bytes: gib(10) + (256 << 20) + 1,
                },
                Some(gib(1)),
            )
            .unwrap()
            .contains("attempt swap grew"));
        assert_eq!(
            policy.violation(
                H3MetalMemorySample {
                    available_bytes: gib(12),
                    used_swap_bytes: gib(10),
                },
                Some(gib(1)),
            ),
            None
        );
    }

    #[test]
    fn campaign_policy_rejects_a_native_allocation_above_the_ceiling() {
        let policy = H3MetalMemoryPolicy::for_campaign_sample(
            H3MetalMemorySample {
                available_bytes: gib(20),
                used_swap_bytes: gib(10),
            },
            Some(gib(8)),
        );
        let violation = policy
            .violation(
                H3MetalMemorySample {
                    available_bytes: gib(20),
                    used_swap_bytes: gib(10),
                },
                Some(gib(8) + 1),
            )
            .expect("the ceiling is enforced");
        assert!(violation.contains("native allocation"));
        assert!(violation.contains("ceiling"));
        // Production policy (no ceiling) ignores the native sample entirely.
        let production = H3MetalMemoryPolicy::for_sample(H3MetalMemorySample {
            available_bytes: gib(20),
            used_swap_bytes: gib(10),
        });
        assert_eq!(production.maximum_native_bytes, None);
        assert_eq!(
            production.violation(
                H3MetalMemorySample {
                    available_bytes: gib(20),
                    used_swap_bytes: gib(10),
                },
                Some(gib(64)),
            ),
            None
        );
    }

    #[test]
    fn guard_rejects_floor_and_attempt_swap_growth_independently() {
        let policy = H3MetalMemoryPolicy {
            minimum_available_bytes: gib(8),
            baseline_swap_bytes: gib(10),
            maximum_swap_growth_bytes: gib(2),
            maximum_native_bytes: None,
        };
        assert!(policy
            .violation(
                H3MetalMemorySample {
                    available_bytes: gib(7),
                    used_swap_bytes: gib(10),
                },
                None,
            )
            .unwrap()
            .contains("host safety floor"));
        assert!(policy
            .violation(
                H3MetalMemorySample {
                    available_bytes: gib(12),
                    used_swap_bytes: gib(13),
                },
                None,
            )
            .unwrap()
            .contains("attempt swap grew"));
        assert_eq!(
            policy.violation(
                H3MetalMemorySample {
                    available_bytes: gib(12),
                    used_swap_bytes: gib(12),
                },
                None,
            ),
            None
        );
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
            maximum_native_bytes: None,
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
