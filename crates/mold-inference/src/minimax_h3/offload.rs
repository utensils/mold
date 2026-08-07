//! Cancellation-safe MiniMax H3 block-streaming state.
//!
//! The H3 DiT primitive is still on a sibling stack, so this module owns only
//! the scheduler-issued route, residency, and bounded prefetch lifecycle. A
//! future integration supplies real block objects through [`H3BlockLoader`].
//! Synthetic fixtures exercise the complete ownership contract without model
//! headers or weights.

use anyhow::{anyhow, Result};
use std::collections::VecDeque;

const H3_MAIN_BLOCK_COUNT: usize = 50;
const H3_MAX_PREFETCH_DEPTH: usize = 2;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FrozenH3BlockStreamingPlan {
    pub(crate) device_id: String,
    pub(crate) execution_fingerprint: String,
    pub(crate) resident_block_count: usize,
    pub(crate) prefetch_depth: usize,
}

impl FrozenH3BlockStreamingPlan {
    pub(crate) fn new(
        device_id: impl Into<String>,
        execution_fingerprint: impl Into<String>,
        resident_block_count: usize,
        prefetch_depth: usize,
    ) -> Result<Self> {
        let device_id = device_id.into();
        let execution_fingerprint = execution_fingerprint.into();
        if device_id.trim().is_empty() || execution_fingerprint.trim().is_empty() {
            return Err(anyhow!(
                "H3 block streaming requires frozen device and execution identities"
            ));
        }
        if resident_block_count > H3_MAIN_BLOCK_COUNT {
            return Err(anyhow!(
                "H3 resident block count {resident_block_count} exceeds {H3_MAIN_BLOCK_COUNT}"
            ));
        }
        if prefetch_depth > H3_MAX_PREFETCH_DEPTH {
            return Err(anyhow!(
                "H3 prefetch depth {prefetch_depth} exceeds bounded maximum {H3_MAX_PREFETCH_DEPTH}"
            ));
        }
        Ok(Self {
            device_id,
            execution_fingerprint,
            resident_block_count,
            prefetch_depth,
        })
    }
}

/// One Scheduler V2 lease. There is intentionally no sibling-device accessor:
/// one H3 render has exactly one execution route.
pub(crate) trait H3BlockLease {
    fn device_id(&self) -> &str;
    fn execution_fingerprint(&self) -> &str;
}

pub(crate) trait H3BlockLoader {
    type Block;

    /// Load one block on the already-frozen route. The returned block must own
    /// its resources so dropping it releases device/host allocations.
    fn load_block(
        &mut self,
        index: usize,
        device_id: &str,
        execution_fingerprint: &str,
    ) -> Result<Self::Block>;

    /// Synchronize a prefetched block before execution. A future CUDA loader
    /// can wait on its transfer event here; the synthetic loader is immediate.
    fn wait_for_prefetch(&mut self, _index: usize) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum H3StreamCheckpoint {
    BeforeResidentLoad(usize),
    AfterResidentLoad(usize),
    BeforePrefetch(usize),
    AfterPrefetch(usize),
    BeforePrefetchWait(usize),
    AfterPrefetchWait(usize),
    BeforeBlock(usize),
    AfterBlock(usize),
}

pub(crate) trait H3StreamCancellation {
    fn checkpoint(&self, point: H3StreamCheckpoint) -> Result<()>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum H3BlockStreamEvent {
    ResidentLoaded {
        index: usize,
    },
    PrefetchStarted {
        index: usize,
        queued: usize,
        capacity: usize,
    },
    PrefetchReady {
        index: usize,
        queued: usize,
        capacity: usize,
    },
    BlockStarted {
        index: usize,
        resident: bool,
    },
    BlockFinished {
        index: usize,
        resident: bool,
    },
}

/// Prepared resident blocks survive across denoise steps; streamed/prefetched
/// blocks are scoped to one step and are dropped on success, cancellation, or
/// any loader/forward error.
pub(crate) struct H3BlockStreamState<L: H3BlockLoader> {
    plan: FrozenH3BlockStreamingPlan,
    loader: L,
    resident: Vec<L::Block>,
    prepared: bool,
}

impl<L: H3BlockLoader> H3BlockStreamState<L> {
    pub(crate) fn new(
        plan: FrozenH3BlockStreamingPlan,
        lease: &impl H3BlockLease,
        loader: L,
    ) -> Result<Self> {
        if lease.device_id() != plan.device_id {
            return Err(anyhow!(
                "H3 block lease targets {}, frozen plan targets {}; sibling-device borrowing is forbidden",
                lease.device_id(),
                plan.device_id
            ));
        }
        if lease.execution_fingerprint() != plan.execution_fingerprint {
            return Err(anyhow!(
                "H3 block lease execution fingerprint differs from the frozen plan"
            ));
        }
        Ok(Self {
            plan,
            loader,
            resident: Vec::new(),
            prepared: false,
        })
    }

    pub(crate) fn resident_block_count(&self) -> usize {
        self.resident.len()
    }

    pub(crate) fn prepare(
        &mut self,
        cancellation: &impl H3StreamCancellation,
        mut observe: impl FnMut(H3BlockStreamEvent),
    ) -> Result<()> {
        if self.prepared {
            return Ok(());
        }
        let mut resident = Vec::with_capacity(self.plan.resident_block_count);
        for index in 0..self.plan.resident_block_count {
            cancellation.checkpoint(H3StreamCheckpoint::BeforeResidentLoad(index))?;
            let block = self.loader.load_block(
                index,
                &self.plan.device_id,
                &self.plan.execution_fingerprint,
            )?;
            cancellation.checkpoint(H3StreamCheckpoint::AfterResidentLoad(index))?;
            resident.push(block);
            observe(H3BlockStreamEvent::ResidentLoaded { index });
        }
        self.resident = resident;
        self.prepared = true;
        Ok(())
    }

    pub(crate) fn run_step(
        &mut self,
        cancellation: &impl H3StreamCancellation,
        mut observe: impl FnMut(H3BlockStreamEvent),
        mut forward: impl FnMut(usize, &L::Block) -> Result<()>,
    ) -> Result<()> {
        if !self.prepared {
            return Err(anyhow!(
                "H3 block stream must prepare resident blocks before a denoise step"
            ));
        }
        for (index, block) in self.resident.iter().enumerate() {
            cancellation.checkpoint(H3StreamCheckpoint::BeforeBlock(index))?;
            observe(H3BlockStreamEvent::BlockStarted {
                index,
                resident: true,
            });
            forward(index, block)?;
            cancellation.checkpoint(H3StreamCheckpoint::AfterBlock(index))?;
            observe(H3BlockStreamEvent::BlockFinished {
                index,
                resident: true,
            });
        }

        let first_streamed = self.plan.resident_block_count;
        if first_streamed == H3_MAIN_BLOCK_COUNT {
            return Ok(());
        }
        let capacity = self.plan.prefetch_depth.saturating_add(1);
        let mut queue = VecDeque::with_capacity(capacity);
        let mut next = first_streamed;
        while queue.len() < capacity && next < H3_MAIN_BLOCK_COUNT {
            prefetch_one(
                &self.plan,
                &mut self.loader,
                cancellation,
                &mut observe,
                &mut queue,
                next,
                capacity,
            )?;
            next += 1;
        }

        while let Some((index, block)) = queue.pop_front() {
            cancellation.checkpoint(H3StreamCheckpoint::BeforePrefetchWait(index))?;
            self.loader.wait_for_prefetch(index)?;
            cancellation.checkpoint(H3StreamCheckpoint::AfterPrefetchWait(index))?;
            cancellation.checkpoint(H3StreamCheckpoint::BeforeBlock(index))?;
            observe(H3BlockStreamEvent::BlockStarted {
                index,
                resident: false,
            });
            forward(index, &block)?;
            cancellation.checkpoint(H3StreamCheckpoint::AfterBlock(index))?;
            observe(H3BlockStreamEvent::BlockFinished {
                index,
                resident: false,
            });
            drop(block);

            if next < H3_MAIN_BLOCK_COUNT {
                prefetch_one(
                    &self.plan,
                    &mut self.loader,
                    cancellation,
                    &mut observe,
                    &mut queue,
                    next,
                    capacity,
                )?;
                next += 1;
            }
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn prefetch_one<L: H3BlockLoader>(
    plan: &FrozenH3BlockStreamingPlan,
    loader: &mut L,
    cancellation: &impl H3StreamCancellation,
    observe: &mut impl FnMut(H3BlockStreamEvent),
    queue: &mut VecDeque<(usize, L::Block)>,
    index: usize,
    capacity: usize,
) -> Result<()> {
    cancellation.checkpoint(H3StreamCheckpoint::BeforePrefetch(index))?;
    observe(H3BlockStreamEvent::PrefetchStarted {
        index,
        queued: queue.len(),
        capacity,
    });
    let block = loader.load_block(index, &plan.device_id, &plan.execution_fingerprint)?;
    cancellation.checkpoint(H3StreamCheckpoint::AfterPrefetch(index))?;
    queue.push_back((index, block));
    observe(H3BlockStreamEvent::PrefetchReady {
        index,
        queued: queue.len(),
        capacity,
    });
    debug_assert!(queue.len() <= capacity);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    };

    #[derive(Default)]
    struct Counters {
        live: AtomicUsize,
        peak: AtomicUsize,
        loaded: Mutex<Vec<usize>>,
        dropped: Mutex<Vec<usize>>,
    }

    struct SyntheticBlock {
        index: usize,
        counters: Arc<Counters>,
    }

    impl Drop for SyntheticBlock {
        fn drop(&mut self) {
            self.counters.live.fetch_sub(1, Ordering::SeqCst);
            self.counters.dropped.lock().unwrap().push(self.index);
        }
    }

    struct SyntheticLoader {
        counters: Arc<Counters>,
        fail_load: Option<usize>,
        fail_wait: Option<usize>,
    }

    impl H3BlockLoader for SyntheticLoader {
        type Block = SyntheticBlock;

        fn load_block(
            &mut self,
            index: usize,
            device_id: &str,
            execution_fingerprint: &str,
        ) -> Result<Self::Block> {
            assert_eq!(device_id, "gpu-0");
            assert_eq!(execution_fingerprint, "plan-v1");
            if self.fail_load == Some(index) {
                return Err(anyhow!("synthetic load failure at block {index}"));
            }
            self.counters.loaded.lock().unwrap().push(index);
            let live = self.counters.live.fetch_add(1, Ordering::SeqCst) + 1;
            self.counters.peak.fetch_max(live, Ordering::SeqCst);
            Ok(SyntheticBlock {
                index,
                counters: Arc::clone(&self.counters),
            })
        }

        fn wait_for_prefetch(&mut self, index: usize) -> Result<()> {
            if self.fail_wait == Some(index) {
                Err(anyhow!("synthetic prefetch wait failure at block {index}"))
            } else {
                Ok(())
            }
        }
    }

    struct SyntheticLease<'a> {
        device: &'a str,
        fingerprint: &'a str,
    }

    impl H3BlockLease for SyntheticLease<'_> {
        fn device_id(&self) -> &str {
            self.device
        }

        fn execution_fingerprint(&self) -> &str {
            self.fingerprint
        }
    }

    struct CancelAt(Option<H3StreamCheckpoint>);

    impl H3StreamCancellation for CancelAt {
        fn checkpoint(&self, point: H3StreamCheckpoint) -> Result<()> {
            if self.0 == Some(point) {
                Err(anyhow!("synthetic cancellation at {point:?}"))
            } else {
                Ok(())
            }
        }
    }

    fn state(
        resident: usize,
        prefetch: usize,
        counters: Arc<Counters>,
        fail_load: Option<usize>,
        fail_wait: Option<usize>,
    ) -> H3BlockStreamState<SyntheticLoader> {
        let plan = FrozenH3BlockStreamingPlan::new("gpu-0", "plan-v1", resident, prefetch).unwrap();
        H3BlockStreamState::new(
            plan,
            &SyntheticLease {
                device: "gpu-0",
                fingerprint: "plan-v1",
            },
            SyntheticLoader {
                counters,
                fail_load,
                fail_wait,
            },
        )
        .unwrap()
    }

    #[test]
    fn resident_blocks_survive_steps_while_prefetch_is_strictly_bounded() {
        let counters = Arc::new(Counters::default());
        let mut state = state(3, 1, Arc::clone(&counters), None, None);
        state.prepare(&CancelAt(None), |_| {}).unwrap();
        assert_eq!(state.resident_block_count(), 3);
        let mut first = Vec::new();
        state
            .run_step(
                &CancelAt(None),
                |_| {},
                |index, _| {
                    first.push(index);
                    Ok(())
                },
            )
            .unwrap();
        let mut second = Vec::new();
        state
            .run_step(
                &CancelAt(None),
                |_| {},
                |index, _| {
                    second.push(index);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(first, (0..H3_MAIN_BLOCK_COUNT).collect::<Vec<_>>());
        assert_eq!(second, first);
        assert_eq!(counters.live.load(Ordering::SeqCst), 3);
        assert!(
            counters.peak.load(Ordering::SeqCst) <= 3 + 2,
            "resident plus current/prefetched blocks must remain bounded"
        );
        let loaded = counters.loaded.lock().unwrap();
        assert_eq!(loaded.iter().filter(|&&index| index < 3).count(), 3);
        assert_eq!(loaded.iter().filter(|&&index| index >= 3).count(), 47 * 2);
        drop(loaded);
        drop(state);
        assert_eq!(counters.live.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cancellation_during_resident_load_releases_partial_preparation() {
        let counters = Arc::new(Counters::default());
        let mut state = state(4, 1, Arc::clone(&counters), None, None);
        let error = state
            .prepare(
                &CancelAt(Some(H3StreamCheckpoint::AfterResidentLoad(1))),
                |_| {},
            )
            .unwrap_err();
        assert!(error.to_string().contains("synthetic cancellation"));
        assert_eq!(state.resident_block_count(), 0);
        assert_eq!(counters.live.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cancellation_at_prefetch_wait_drops_every_transient_block() {
        let counters = Arc::new(Counters::default());
        let mut state = state(2, 2, Arc::clone(&counters), None, None);
        state.prepare(&CancelAt(None), |_| {}).unwrap();
        let error = state
            .run_step(
                &CancelAt(Some(H3StreamCheckpoint::BeforePrefetchWait(2))),
                |_| {},
                |_, _| Ok(()),
            )
            .unwrap_err();
        assert!(error.to_string().contains("synthetic cancellation"));
        assert_eq!(
            counters.live.load(Ordering::SeqCst),
            2,
            "only prepared resident blocks may survive the failed step"
        );
        drop(state);
        assert_eq!(counters.live.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn loader_wait_and_forward_failures_release_prefetch_queue() {
        for (fail_wait, fail_forward) in [(Some(4), None), (None, Some(5))] {
            let counters = Arc::new(Counters::default());
            let mut state = state(1, 2, Arc::clone(&counters), None, fail_wait);
            state.prepare(&CancelAt(None), |_| {}).unwrap();
            assert!(state
                .run_step(
                    &CancelAt(None),
                    |_| {},
                    |index, _| if fail_forward == Some(index) {
                        Err(anyhow!("synthetic forward failure"))
                    } else {
                        Ok(())
                    },
                )
                .is_err());
            assert_eq!(counters.live.load(Ordering::SeqCst), 1);
            drop(state);
            assert_eq!(counters.live.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn route_and_prefetch_authorities_fail_before_loading() {
        let counters = Arc::new(Counters::default());
        let plan = FrozenH3BlockStreamingPlan::new("gpu-0", "plan-v1", 1, 1).unwrap();
        let loader = SyntheticLoader {
            counters: Arc::clone(&counters),
            fail_load: None,
            fail_wait: None,
        };
        let error = H3BlockStreamState::new(
            plan.clone(),
            &SyntheticLease {
                device: "gpu-1",
                fingerprint: "plan-v1",
            },
            loader,
        )
        .err()
        .expect("sibling device must fail");
        assert!(error.to_string().contains("sibling-device"));
        assert_eq!(counters.live.load(Ordering::SeqCst), 0);

        assert!(FrozenH3BlockStreamingPlan::new(
            "gpu-0",
            "plan-v1",
            H3_MAIN_BLOCK_COUNT,
            H3_MAX_PREFETCH_DEPTH + 1,
        )
        .is_err());
    }
}
