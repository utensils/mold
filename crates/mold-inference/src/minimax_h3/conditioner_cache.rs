//! Process-global reuse of the MiniMax H3 Qwen3-VL conditioner output.
//!
//! One repeated prompt on one frozen conditioner route produces the same BF16
//! `[1, rows, 5120]` states every time: `H3PrivateQwenAdapter::encode_prepared`
//! already ends with `states.to_device(execution_device)`, so the runtime's own
//! answer is a device copy of a host-shaped tensor and one more
//! `to_device(Cpu)` / `to_device(exec)` pair adds no dtype change and no
//! numerics. Holding that copy removes the 15.7 GB conditioner load and its
//! encode from a repeated render.
//!
//! Three properties make this safe to keep:
//!
//! * The key (see [`derive_key`]) carries the code identity, the model, task
//!   and mode, the prompt, every conditioning and reference byte, the row
//!   counts, the conditioner ROUTE (placement plus the exact device id — CUDA
//!   and CPU conditioner outputs are not bit-identical), and the conditioner's
//!   own artifact pins. Seed, guidance and step count are excluded on purpose:
//!   none of them reaches the conditioner. Nothing minted PER ATTEMPT may join
//!   it either — a per-attempt identity cannot produce a stale hit, but it
//!   produces a cache that never hits, which is what the frozen component's
//!   `validation_sha256` did: it carries the attempt's runtime qualification
//!   identity, and that identity hashes the request envelope, so every clip
//!   length, step count and canvas presented a different key for one
//!   conditioner file.
//! * The cache is byte-bounded, not entry-bounded (a nine-image Ref2VA
//!   presentation is ~380 MB while FL2VA tops out near 20 MiB), and its CPU
//!   tensors are anonymous pages the host ledger already subtracts from
//!   `MemAvailable`.
//! * It is inert under `h3-private-uat`, so a capture-scope run always
//!   exercises the conditioner and always produces the runtime-bound
//!   observation its manifest requires.
//!
//! Built beside `crate::cache`, not on it: [`crate::cache::LruCache`] is entry
//! COUNT bounded and its `CachedTensor::restore` applies a `to_dtype`, and
//! neither is what this needs.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use mold_candle::minimax_h3::{
    H3ModalityTag, H3_QWEN_NVFP4_AWQ_POLICY_SHA256, H3_QWEN_NVFP4_AWQ_SHA256,
};
use mold_core::minimax_h3::{Mode, Task, REFERENCE_PREPROCESS_VERSION};
use sha2::{Digest, Sha256};

use crate::h3_factory::{
    FrozenH3FactoryAuthority, H3FactoryConditionerPlacement, H3FactoryPreparedRequestInput,
};

use super::pipeline::H3TextConditioning;

/// Default budget. Far below the host ledger's own `max(15 %, 8 GiB)` safety
/// floor, so a full cache never moves an admission decision.
pub(crate) const H3_CONDITIONER_CACHE_DEFAULT_BUDGET_BYTES: u64 = 512 << 20;
pub(crate) const H3_CONDITIONER_CACHE_ENV: &str = "MOLD_H3_CONDITIONER_CACHE";
/// Ceiling on the parsed budget. A byte bound that no entry can ever reach is
/// not a bound: `u64::MAX` MiB saturates, and after that neither the per-entry
/// refusal nor the eviction loop can fire again. 64 GiB is far above any
/// conditioning set the family can produce (a nine-image Ref2VA presentation
/// is ~380 MB) and far below a host budget worth silently claiming.
pub(crate) const H3_CONDITIONER_CACHE_MAX_BUDGET_MIB: u64 = 65_536;
/// The one conditioner state width the H3 pipeline accepts.
const H3_CONDITIONER_STATE_WIDTH: usize = 5_120;
const H3_CONDITIONER_CACHE_KEY_DOMAIN: &[u8] = b"mold.minimax-h3.conditioner-cache.v1\0";

/// Placement plus the exact device the conditioner ran on.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3ConditionerRouteIdentity {
    pub(crate) placement: H3FactoryConditionerPlacement,
    pub(crate) device_id: String,
}

impl H3ConditionerRouteIdentity {
    pub(crate) fn describe(&self) -> String {
        format!(
            "{}:{}",
            placement_tag(self.placement),
            self.device_id.as_str()
        )
    }
}

/// Every varying axis of the cache key, gathered so a test can move exactly
/// one of them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct H3ConditionerCacheKeyInput<'a> {
    pub(crate) runtime_code_identity_sha256: &'a str,
    pub(crate) canonical_model: &'a str,
    pub(crate) task: Task,
    pub(crate) mode: Mode,
    pub(crate) prompt_sha256: &'a str,
    pub(crate) conditioning_fingerprint: &'a str,
    pub(crate) reference_fingerprint: &'a str,
    /// Presentation frame count. Hashed for Ref2VA ONLY: its reference shapes
    /// are derived from the target frame count and its presentation carries
    /// per-block `<t.t seconds>` stamps, while FL2VA's conditioner input is a
    /// pure function of the prompt and the endpoints' normalized CPU pixels.
    pub(crate) frames: u32,
    pub(crate) qwen_output_text_rows: u64,
    pub(crate) qwen_vision_rows: u64,
    pub(crate) placement: H3FactoryConditionerPlacement,
    pub(crate) conditioner_device_id: &'a str,
    /// The frozen conditioner component's CONTENT digest, and deliberately not
    /// its validation digest.
    ///
    /// The content digest is the artifact identity: it hashes every member's
    /// relative path, source revision, `sha256`, structural contract, header
    /// identity, policy identity, size and tensor count, so it answers
    /// "which conditioner file ran" completely. The validation digest answers
    /// a different question — `private_h3_component_digests` folds the
    /// attempt's RUNTIME QUALIFICATION identity into it, and
    /// `runtime_qualification_identity` hashes the request envelope
    /// (`public_runtime_envelope_for_shape(canvas, frames, steps)`), so it
    /// moves with the clip length, the step count, the canvas and the row
    /// counts. Hashing it here silently made every shape change a miss.
    pub(crate) conditioner_component_content_sha256: &'a str,
    pub(crate) support_identity_sha256: &'a str,
}

const fn placement_tag(placement: H3FactoryConditionerPlacement) -> &'static str {
    match placement {
        H3FactoryConditionerPlacement::AssignedCudaThenDrop => "assigned-cuda-then-drop",
        H3FactoryConditionerPlacement::AssignedMetalThenDrop => "assigned-metal-then-drop",
        H3FactoryConditionerPlacement::HostCpuThenDrop => "host-cpu-then-drop",
    }
}

const fn task_tag(task: Task) -> &'static str {
    match task {
        Task::Fl2va => "fl2va",
        Task::Ref2va => "ref2va",
    }
}

const fn mode_tag(mode: Mode) -> &'static str {
    match mode {
        Mode::TextToAudioVideo => "t2va",
        Mode::FirstFrameToAudioVideo => "first-frame-fl2va",
        Mode::LastFrameToAudioVideo => "last-frame-fl2va",
        Mode::FirstAndLastFrameToAudioVideo => "first-last-frame-fl2va",
        Mode::ReferenceToAudioVideo => "ref2va",
    }
}

/// SHA-256 over the domain plus length-prefixed fields, the same composition
/// the runtime observer and the conditioning fingerprint already use.
pub(crate) fn derive_key(input: H3ConditionerCacheKeyInput<'_>) -> String {
    let mut digest = Sha256::new();
    digest.update(H3_CONDITIONER_CACHE_KEY_DOMAIN);
    let mut field = |bytes: &[u8]| {
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    };
    field(input.runtime_code_identity_sha256.as_bytes());
    field(input.canonical_model.as_bytes());
    field(task_tag(input.task).as_bytes());
    field(mode_tag(input.mode).as_bytes());
    field(input.prompt_sha256.as_bytes());
    field(input.conditioning_fingerprint.as_bytes());
    field(input.reference_fingerprint.as_bytes());
    field(&REFERENCE_PREPROCESS_VERSION.to_le_bytes());
    match input.task {
        Task::Ref2va => field(&input.frames.to_le_bytes()),
        Task::Fl2va => field(b"fl2va-presentation-is-frame-independent"),
    }
    field(&input.qwen_output_text_rows.to_le_bytes());
    field(&input.qwen_vision_rows.to_le_bytes());
    field(placement_tag(input.placement).as_bytes());
    field(input.conditioner_device_id.as_bytes());
    field(H3_QWEN_NVFP4_AWQ_SHA256.as_bytes());
    field(H3_QWEN_NVFP4_AWQ_POLICY_SHA256.as_bytes());
    field(input.conditioner_component_content_sha256.as_bytes());
    field(input.support_identity_sha256.as_bytes());
    format!("{:x}", digest.finalize())
}

/// Derive the key and the route identity from the frozen attempt.
pub(crate) fn key_for(
    request: &H3FactoryPreparedRequestInput,
    authority: &FrozenH3FactoryAuthority,
    support_identity_sha256: &str,
    conditioner_device_id: &str,
) -> (String, H3ConditionerRouteIdentity) {
    // Only the content half: the validation half carries the attempt's own
    // runtime qualification identity, which hashes the request envelope.
    let (content_sha256, _validation_sha256) = authority.conditioner_component_authority();
    let placement = authority.conditioner_placement();
    let key = derive_key(H3ConditionerCacheKeyInput {
        runtime_code_identity_sha256: super::PRIVATE_RUNTIME_CODE_IDENTITY_SHA256,
        canonical_model: &request.canonical_model,
        task: request.task,
        mode: request.mode,
        prompt_sha256: &request.prompt_sha256,
        conditioning_fingerprint: &request.conditioning_fingerprint,
        reference_fingerprint: &request.reference_fingerprint,
        frames: request.frames,
        qwen_output_text_rows: request.rows.qwen_output_text_rows,
        qwen_vision_rows: request.rows.qwen_vision_rows,
        placement,
        conditioner_device_id,
        conditioner_component_content_sha256: content_sha256,
        support_identity_sha256,
    });
    (
        key,
        H3ConditionerRouteIdentity {
            placement,
            device_id: conditioner_device_id.to_owned(),
        },
    )
}

/// One conditioner answer, held on the host.
pub(crate) struct H3CachedConditioning {
    states_cpu: Tensor,
    tags: Vec<H3ModalityTag>,
    text_rows: u64,
    vision_rows: u64,
    route: H3ConditionerRouteIdentity,
    bytes: u64,
}

impl H3CachedConditioning {
    /// Copy one encode result to the host, refusing anything the H3 pipeline
    /// would not itself accept back.
    pub(crate) fn capture(
        text: &H3TextConditioning,
        text_rows: u64,
        vision_rows: u64,
        route: H3ConditionerRouteIdentity,
    ) -> Result<Self> {
        let (batch, rows, width) = text.states.dims3()?;
        if batch != 1 || rows == 0 || width != H3_CONDITIONER_STATE_WIDTH {
            bail!(
                "private H3 conditioner cache refuses a [{batch},{rows},{width}] state tensor; it must be [1,rows,{H3_CONDITIONER_STATE_WIDTH}]"
            )
        }
        if text.states.dtype() != DType::BF16 {
            bail!(
                "private H3 conditioner cache refuses a {:?} state tensor; the released conditioner output is BF16",
                text.states.dtype()
            )
        }
        if text.tags.len() != rows {
            bail!(
                "private H3 conditioner cache refuses {} tags for {rows} state rows",
                text.tags.len()
            )
        }
        if u64::try_from(rows)? != text_rows {
            bail!(
                "private H3 conditioner cache refuses {rows} state rows for {text_rows} frozen text rows"
            )
        }
        let states_cpu = text.states.to_device(&Device::Cpu)?.contiguous()?;
        let bytes = u64::try_from(states_cpu.elem_count())?
            .checked_mul(u64::try_from(DType::BF16.size_in_bytes())?)
            .and_then(|bytes| bytes.checked_add(u64::try_from(text.tags.len()).ok()?))
            .ok_or_else(|| anyhow::anyhow!("private H3 conditioner cache entry bytes overflow"))?;
        Ok(Self {
            states_cpu,
            tags: text.tags.clone(),
            text_rows,
            vision_rows,
            route,
            bytes,
        })
    }

    pub(crate) const fn text_rows(&self) -> u64 {
        self.text_rows
    }

    pub(crate) const fn vision_rows(&self) -> u64 {
        self.vision_rows
    }

    pub(crate) const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) const fn route(&self) -> &H3ConditionerRouteIdentity {
        &self.route
    }

    /// Restore onto the execution device with no dtype conversion at all.
    pub(crate) fn restore(&self, device: &Device) -> Result<H3TextConditioning> {
        let states = self.states_cpu.to_device(device)?;
        Ok(H3TextConditioning {
            states,
            tags: self.tags.clone(),
            #[cfg(test)]
            lifetime_probe: None,
        })
    }
}

/// Byte-bounded LRU. `order` is least-recently-used first.
struct H3ConditionerCache {
    budget_bytes: u64,
    resident_bytes: u64,
    order: VecDeque<String>,
    entries: HashMap<String, Arc<H3CachedConditioning>>,
}

impl H3ConditionerCache {
    fn with_budget(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            resident_bytes: 0,
            order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    fn get(&mut self, key: &str) -> Option<Arc<H3CachedConditioning>> {
        let entry = self.entries.get(key)?.clone();
        if let Some(position) = self.order.iter().position(|held| held == key) {
            let promoted = self.order.remove(position)?;
            self.order.push_back(promoted);
        }
        Some(entry)
    }

    fn insert(&mut self, key: String, entry: H3CachedConditioning) -> bool {
        if entry.bytes > self.budget_bytes {
            tracing::debug!(
                target: "mold::minimax_h3::conditioner_cache",
                entry_bytes = entry.bytes,
                budget_bytes = self.budget_bytes,
                "MiniMax H3 conditioner output exceeds the whole cache budget and is not stored"
            );
            return false;
        }
        self.remove(&key);
        while self.resident_bytes.saturating_add(entry.bytes) > self.budget_bytes {
            let Some(evicted) = self.order.front().cloned() else {
                break;
            };
            self.remove(&evicted);
        }
        self.resident_bytes = self.resident_bytes.saturating_add(entry.bytes);
        self.order.push_back(key.clone());
        self.entries.insert(key, Arc::new(entry));
        true
    }

    fn remove(&mut self, key: &str) {
        if let Some(entry) = self.entries.remove(key) {
            self.resident_bytes = self.resident_bytes.saturating_sub(entry.bytes);
            if let Some(position) = self.order.iter().position(|held| held == key) {
                self.order.remove(position);
            }
        }
    }

    fn clear(&mut self) -> u64 {
        let released = self.resident_bytes;
        self.entries.clear();
        self.order.clear();
        self.resident_bytes = 0;
        released
    }
}

/// `unset` -> 512 MiB, `0|off|false|no|disabled` -> disabled, `<n>` -> n MiB
/// clamped to [`H3_CONDITIONER_CACHE_MAX_BUDGET_MIB`].
///
/// An unparseable value keeps the default rather than silently disabling the
/// cache: a typo must not change residency without saying so.
pub(crate) fn budget_from_env(value: Option<&str>) -> Option<u64> {
    let Some(value) = value else {
        return Some(H3_CONDITIONER_CACHE_DEFAULT_BUDGET_BYTES);
    };
    let value = value.trim().to_ascii_lowercase();
    if value.is_empty() {
        return Some(H3_CONDITIONER_CACHE_DEFAULT_BUDGET_BYTES);
    }
    if matches!(value.as_str(), "0" | "off" | "false" | "no" | "disabled") {
        return None;
    }
    match value.parse::<u64>() {
        Ok(0) => None,
        Ok(mib) if mib > H3_CONDITIONER_CACHE_MAX_BUDGET_MIB => {
            tracing::warn!(
                target: "mold::minimax_h3::conditioner_cache",
                requested_mib = mib,
                clamped_mib = H3_CONDITIONER_CACHE_MAX_BUDGET_MIB,
                "{H3_CONDITIONER_CACHE_ENV} exceeds the accepted ceiling; clamping"
            );
            Some(H3_CONDITIONER_CACHE_MAX_BUDGET_MIB << 20)
        }
        Ok(mib) => Some(mib << 20),
        Err(_) => {
            tracing::warn!(
                target: "mold::minimax_h3::conditioner_cache",
                value = %value,
                "{H3_CONDITIONER_CACHE_ENV} is not `off` or a MiB count; keeping the default budget"
            );
            Some(H3_CONDITIONER_CACHE_DEFAULT_BUDGET_BYTES)
        }
    }
}

/// Resolved once per process.
///
/// Read straight from the environment rather than through
/// [`crate::runtime_env`]: the knob moves residency and wall clock but not
/// device choice, weights, or numerics, exactly like `MOLD_LTX2_KEEP_SESSION`.
/// It must never join `ENGINE_SHAPING_VARIABLES`.
pub(crate) fn budget_bytes() -> Option<u64> {
    // Capture-scope builds always exercise the conditioner: the private
    // runtime-bound observation the capture manifest requires describes a real
    // Qwen encode, and a served hit has none to describe.
    if cfg!(feature = "h3-private-uat") {
        return None;
    }
    static BUDGET: OnceLock<Option<u64>> = OnceLock::new();
    *BUDGET.get_or_init(|| budget_from_env(std::env::var(H3_CONDITIONER_CACHE_ENV).ok().as_deref()))
}

pub(crate) fn enabled() -> bool {
    budget_bytes().is_some()
}

fn cache() -> &'static Mutex<H3ConditionerCache> {
    static CACHE: OnceLock<Mutex<H3ConditionerCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(H3ConditionerCache::with_budget(budget_bytes().unwrap_or(0))))
}

fn locked() -> std::sync::MutexGuard<'static, H3ConditionerCache> {
    cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The map's OWN budget is the one authority on what it holds.
///
/// A disabled process (env `off`, or any `h3-private-uat` build) initializes
/// the global with a zero budget, and [`H3ConditionerCache::insert`] refuses
/// every entry against it — [`H3CachedConditioning::capture`] refuses a
/// zero-row state, so no entry can ever weigh zero. Reading the budget here
/// rather than re-deriving `enabled()` is what lets a test install a known
/// budget on the global map instead of depending on the developer's shell.
pub(crate) fn lookup(key: &str) -> Option<Arc<H3CachedConditioning>> {
    locked().get(key)
}

pub(crate) fn insert(key: String, entry: H3CachedConditioning) {
    let bytes = entry.bytes;
    if locked().insert(key, entry) {
        tracing::debug!(
            target: "mold::minimax_h3::conditioner_cache",
            entry_bytes = bytes,
            "retained MiniMax H3 conditioner output for reuse"
        );
    }
}

/// Release every cached conditioner output and report the bytes handed back.
///
/// The reclaim contract, in full: H3 never enters `ModelCache`, so
/// `host_reclaim::reclaim_headroom` reaches
/// `routes::release_host_memory_after_unload` only on a host that also runs
/// another family. `gpu_worker::prepare_private_h3_allocation_boundary`
/// therefore calls this directly when the sampled host headroom is below the
/// attempt's charge, which is the only reclaim an H3-only host ever performs.
pub fn h3_conditioner_cache_clear() -> u64 {
    locked().clear()
}

/// What the process-global map currently holds.
///
/// Crate-visible on purpose: the server's reclaim paths call
/// [`h3_conditioner_cache_clear`], which already reports the bytes it handed
/// back, so exporting a second reader that nothing calls would advertise an
/// operator surface that does not exist.
pub(crate) fn h3_conditioner_cache_resident_bytes() -> u64 {
    locked().resident_bytes
}

/// Serialize every test that drives the PROCESS-GLOBAL map.
///
/// Holding this guard is what keeps such a test from racing a sibling; it
/// deliberately does not touch the budget, so a capture-scope build's inert
/// map stays inert while it is held.
#[cfg(test)]
pub(crate) fn process_global_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Install a known budget on the process-global map and empty it.
///
/// `budget_bytes()` resolves the environment once per process through a
/// `OnceLock`, so a test that wants the global map enabled cannot get there by
/// setting an environment variable — and one that asserts on absolute resident
/// bytes must not inherit whatever the developer's shell says. Call under
/// [`process_global_test_guard`].
#[cfg(test)]
pub(crate) fn install_process_global_budget_for_test(budget_bytes: u64) {
    let mut cache = locked();
    cache.clear();
    cache.budget_bytes = budget_bytes;
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::h3_factory::{
        H3FactoryAuthorityInput, H3FactoryComponentAuthority, H3FactoryComponentRole,
        H3FactoryEndpointAnchor, H3FactoryEndpointInput, H3FactoryEndpointPreprocess,
        H3FactoryPreparedRowsInput, H3FactoryQuantizationAuthority,
    };

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    /// Owned key material so each test can move exactly one axis.
    struct Baseline {
        code: String,
        model: String,
        prompt: String,
        conditioning: String,
        reference: String,
        device: String,
        content: String,
        support: String,
    }

    impl Baseline {
        fn new() -> Self {
            Self {
                code: sha('7'),
                model: "minimax-h3-fl2va:comfy-pruned-int8".into(),
                prompt: sha('1'),
                conditioning: sha('2'),
                reference: sha('3'),
                device: "cuda:0".into(),
                content: sha('4'),
                support: sha('6'),
            }
        }

        fn input(&self) -> H3ConditionerCacheKeyInput<'_> {
            H3ConditionerCacheKeyInput {
                runtime_code_identity_sha256: &self.code,
                canonical_model: &self.model,
                task: Task::Fl2va,
                mode: Mode::FirstFrameToAudioVideo,
                prompt_sha256: &self.prompt,
                conditioning_fingerprint: &self.conditioning,
                reference_fingerprint: &self.reference,
                frames: 124,
                qwen_output_text_rows: 1_024,
                qwen_vision_rows: 4_096,
                placement: H3FactoryConditionerPlacement::AssignedCudaThenDrop,
                conditioner_device_id: &self.device,
                conditioner_component_content_sha256: &self.content,
                support_identity_sha256: &self.support,
            }
        }
    }

    #[test]
    fn key_changes_on_every_axis() {
        let baseline = Baseline::new();
        let base = baseline.input();
        let expected = derive_key(base);
        assert_eq!(expected.len(), 64);
        assert_eq!(
            derive_key(baseline.input()),
            expected,
            "equal inputs must give equal keys"
        );

        let other = sha('a');
        let mut moved: Vec<(&str, String)> = Vec::new();
        let mut mutate = |label: &'static str, input: H3ConditionerCacheKeyInput<'_>| {
            moved.push((label, derive_key(input)));
        };
        mutate(
            "code identity",
            H3ConditionerCacheKeyInput {
                runtime_code_identity_sha256: &other,
                ..base
            },
        );
        mutate(
            "canonical model",
            H3ConditionerCacheKeyInput {
                canonical_model: "minimax-h3-ref2va:comfy-pruned-int8",
                ..base
            },
        );
        mutate(
            "task",
            H3ConditionerCacheKeyInput {
                task: Task::Ref2va,
                ..base
            },
        );
        mutate(
            "mode",
            H3ConditionerCacheKeyInput {
                mode: Mode::LastFrameToAudioVideo,
                ..base
            },
        );
        mutate(
            "prompt",
            H3ConditionerCacheKeyInput {
                prompt_sha256: &other,
                ..base
            },
        );
        mutate(
            "conditioning fingerprint",
            H3ConditionerCacheKeyInput {
                conditioning_fingerprint: &other,
                ..base
            },
        );
        mutate(
            "reference fingerprint",
            H3ConditionerCacheKeyInput {
                reference_fingerprint: &other,
                ..base
            },
        );
        mutate(
            "text rows",
            H3ConditionerCacheKeyInput {
                qwen_output_text_rows: 1_025,
                ..base
            },
        );
        mutate(
            "vision rows",
            H3ConditionerCacheKeyInput {
                qwen_vision_rows: 4_097,
                ..base
            },
        );
        mutate(
            "placement",
            H3ConditionerCacheKeyInput {
                placement: H3FactoryConditionerPlacement::HostCpuThenDrop,
                ..base
            },
        );
        mutate(
            "device id",
            H3ConditionerCacheKeyInput {
                conditioner_device_id: "cuda:1",
                ..base
            },
        );
        mutate(
            "component content",
            H3ConditionerCacheKeyInput {
                conditioner_component_content_sha256: &other,
                ..base
            },
        );
        mutate(
            "support identity",
            H3ConditionerCacheKeyInput {
                support_identity_sha256: &other,
                ..base
            },
        );

        for (label, key) in &moved {
            assert_ne!(*key, expected, "{label} must change the conditioner key");
        }
        let mut distinct = moved.iter().map(|(_, key)| key.clone()).collect::<Vec<_>>();
        distinct.sort();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            moved.len(),
            "every axis must be independently addressable"
        );
    }

    #[test]
    fn fl2va_key_ignores_frames_and_ref2va_key_does_not() {
        let baseline = Baseline::new();
        let fl2va = baseline.input();
        assert_eq!(
            derive_key(fl2va),
            derive_key(H3ConditionerCacheKeyInput {
                frames: 141,
                ..fl2va
            }),
            "FL2VA conditioning is a pure function of the prompt and endpoint pixels"
        );

        let ref2va = H3ConditionerCacheKeyInput {
            task: Task::Ref2va,
            mode: Mode::ReferenceToAudioVideo,
            ..fl2va
        };
        assert_ne!(
            derive_key(ref2va),
            derive_key(H3ConditionerCacheKeyInput {
                frames: 141,
                ..ref2va
            }),
            "Ref2VA reference shapes and second stamps are derived from the frame count"
        );
    }

    /// One frozen authority, with the CONDITIONER component pair moved.
    ///
    /// Every other field is what a second render of the same shot keeps: this
    /// is the contract-only authority `key_for` reads.
    fn frozen_authority(
        conditioner_content_sha256: &str,
        conditioner_validation_sha256: &str,
    ) -> FrozenH3FactoryAuthority {
        let components = [
            (
                H3FactoryComponentRole::Conditioner,
                conditioner_content_sha256.to_owned(),
                conditioner_validation_sha256.to_owned(),
            ),
            (H3FactoryComponentRole::Transformer, sha('b'), sha('c')),
            (H3FactoryComponentRole::VisualVae, sha('d'), sha('e')),
            (H3FactoryComponentRole::AudioVae, sha('f'), sha('0')),
        ]
        .into_iter()
        .map(|(role, content, validation)| {
            H3FactoryComponentAuthority::new(role, content, validation).unwrap()
        })
        .collect();
        FrozenH3FactoryAuthority::new_contract_only(H3FactoryAuthorityInput {
            model: mold_core::minimax_h3::FL2VA_COMFY.into(),
            device_id: "cuda:0".into(),
            device_ordinal: 0,
            compute_capability: Some((8, 9)),
            execution_fingerprint: sha('8'),
            conditioner_placement: H3FactoryConditionerPlacement::AssignedCudaThenDrop,
            qwen_parameter_bytes: 2_048,
            qwen_host_resident_parameter_bytes: 1_024,
            qwen_device_resident_parameter_bytes: 1_024,
            qwen_activation_workspace_bytes: 1_024,
            qwen_maximum_tensor_staging_bytes: 512,
            qwen_retained_raw_header_bytes: 64,
            qwen_output_text_rows: 594,
            qwen_vision_rows: 2_304,
            condition_visual_rows: 576,
            resident_block_count: 0,
            prefetch_depth: 0,
            attention_backend: crate::attention::AttentionBackend::Flash,
            attention_chunk: crate::attention::AttentionChunkPolicy::Off,
            attention_kernel_identity: "synthetic-qualified-kernel".into(),
            attention_qualification_sha256: sha('9'),
            attention_full_noncausal: true,
            attention_lossless: true,
            attention_head_count: 56,
            attention_head_dim: 128,
            attention_runtime: None,
            block_offload: true,
            quantization: H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256: sha('c'),
                qwen_policy_sha256: sha('d'),
                pruned_adaln_table_sha256: sha('e'),
                turbo_adapter: None,
            },
            prepared_attempt: None,
            execution_budget_echo: None,
            components,
        })
        .unwrap()
    }

    /// The frozen FL2VA request for one clip length. The conditioner's own
    /// inputs — prompt, endpoint pixels, and both Qwen row counts — are the
    /// SAME at every length, exactly as plato measured them (594 text rows and
    /// 2,304 vision patches at 768x768, at both 124 and 141 frames).
    fn frozen_fl2va_request(frames: u32) -> H3FactoryPreparedRequestInput {
        let video_latent_frames = u64::from(frames).div_ceil(4);
        let target_video_rows = video_latent_frames * 576;
        H3FactoryPreparedRequestInput {
            identity_sha256: sha('1'),
            canonical_model: mold_core::minimax_h3::FL2VA_COMFY.into(),
            task: Task::Fl2va,
            mode: Mode::FirstFrameToAudioVideo,
            prompt_sha256: sha('2'),
            seed: 770_021,
            grid_points: 5,
            denoise_forward_count: 4,
            guidance_f64_bits: 0.0f64.to_bits(),
            strength_f64_bits: 1.0f64.to_bits(),
            batch_size: 1,
            width: 768,
            height: 768,
            frames,
            fps: 24,
            synchronized_audio: true,
            mp4_output: true,
            video_latent_frames,
            audio_latents_per_channel: u64::from(frames),
            audio_samples_per_channel: u64::from(frames) * 800,
            conditioning_fingerprint: sha('3'),
            reference_fingerprint: sha('4'),
            endpoints: vec![H3FactoryEndpointInput {
                anchor: H3FactoryEndpointAnchor::First,
                encoded_bytes: 128,
                encoded_content_sha256: sha('5'),
                preprocess: H3FactoryEndpointPreprocess::PillowLanczosRgbU8CpuV1,
                normalized_shape: [1, 3, 1, 768, 768],
                normalized_cpu_bytes: 768 * 768 * 3,
                normalized_cpu_content_sha256: sha('6'),
            }],
            references: Vec::new(),
            rows: H3FactoryPreparedRowsInput {
                qwen_output_text_rows: 594,
                qwen_vision_rows: 2_304,
                condition_visual_rows: 576,
                condition_audio_rows: 0,
                target_video_rows,
                target_audio_rows: u64::from(frames) * 10 / 3,
                total_packed_rows: 594 + 576 + target_video_rows,
            },
        }
    }

    /// A second render that changes only the clip length must reuse the answer.
    ///
    /// The frozen component's `validation_sha256` is not an artifact identity:
    /// `private_h3_component_digests` folds the attempt's runtime
    /// qualification identity into it, and `runtime_qualification_identity`
    /// hashes the request ENVELOPE — the shipping profile is
    /// `public_runtime_envelope_for_shape(canvas, frames, steps)` — so the
    /// same conditioner file presents a different validation digest at every
    /// frame count, fps, step count and canvas. Hashing it made every such
    /// change a miss: on plato (2026-09-02, `mold-pr2-45e67af2`) a 141-frame
    /// FL2VA render stored key `ccbfa21f…`, an otherwise identical 124-frame
    /// render then MISSED while that entry was still resident and nothing had
    /// cleared it, and a second 141-frame render hit `ccbfa21f…` again.
    #[test]
    fn fl2va_key_ignores_the_per_attempt_component_validation_identity() {
        let content = sha('a');
        let (frames_124, _) = key_for(
            &frozen_fl2va_request(124),
            &frozen_authority(&content, &sha('1')),
            &sha('7'),
            "cuda:0",
        );
        let (frames_141, route) = key_for(
            &frozen_fl2va_request(141),
            &frozen_authority(&content, &sha('2')),
            &sha('7'),
            "cuda:0",
        );
        assert_eq!(
            frames_124, frames_141,
            "one conditioner file, one prompt and one first frame is one key at every clip length"
        );
        assert_eq!(
            route.placement,
            H3FactoryConditionerPlacement::AssignedCudaThenDrop
        );

        // The conditioner's own artifact identity still moves the key: the
        // content digest hashes every member's path, sha256, header identity
        // and policy identity, so nothing about WHICH conditioner ran is lost
        // by refusing the per-attempt validation digest.
        let (other_conditioner, _) = key_for(
            &frozen_fl2va_request(124),
            &frozen_authority(&sha('b'), &sha('1')),
            &sha('7'),
            "cuda:0",
        );
        assert_ne!(
            frames_124, other_conditioner,
            "a different conditioner component content digest is a different key"
        );
    }

    /// The key is the conditioner's inputs, weights and route — and nothing
    /// that is minted per attempt. A per-attempt identity in here does not
    /// produce a stale hit; it produces a cache that never hits, which is how
    /// #814's reuse was lost for every shape change.
    #[test]
    fn the_key_carries_no_per_attempt_authority_identity() {
        let source = include_str!("conditioner_cache.rs");
        // Assembled at runtime so this test's own text does not match.
        let banned = format!("valid{}", "ation_sha256");
        let start = source
            .find("pub(crate) fn derive_key(")
            .expect("the key derivation");
        let body = &source[start..];
        let end = body.find("\n}\n").expect("derive_key body end");
        assert!(
            !body[..end].contains(&banned),
            "derive_key must not hash a per-attempt authority identity"
        );
    }

    fn route() -> H3ConditionerRouteIdentity {
        H3ConditionerRouteIdentity {
            placement: H3FactoryConditionerPlacement::AssignedCudaThenDrop,
            device_id: "cuda:0".into(),
        }
    }

    fn conditioning(rows: usize, dtype: DType, tags: usize) -> H3TextConditioning {
        H3TextConditioning {
            states: Tensor::zeros((1, rows, H3_CONDITIONER_STATE_WIDTH), dtype, &Device::Cpu)
                .unwrap(),
            tags: vec![H3ModalityTag::Text; tags],
            lifetime_probe: None,
        }
    }

    fn entry(rows: usize) -> H3CachedConditioning {
        H3CachedConditioning::capture(
            &conditioning(rows, DType::BF16, rows),
            rows as u64,
            0,
            route(),
        )
        .unwrap()
    }

    const ROW_BYTES: u64 = (H3_CONDITIONER_STATE_WIDTH as u64) * 2 + 1;

    #[test]
    fn capture_records_exact_bytes_rows_and_route() {
        let captured = entry(2);
        assert_eq!(captured.bytes(), 2 * ROW_BYTES);
        assert_eq!(captured.text_rows(), 2);
        assert_eq!(captured.vision_rows(), 0);
        assert_eq!(
            captured.route().describe(),
            "assigned-cuda-then-drop:cuda:0"
        );
        let restored = captured.restore(&Device::Cpu).unwrap();
        assert_eq!(restored.states.dtype(), DType::BF16);
        assert_eq!(restored.states.dims3().unwrap(), (1, 2, 5_120));
        assert_eq!(restored.tags.len(), 2);
    }

    #[test]
    fn capture_refuses_shape_dtype_and_tag_mismatch() {
        assert!(
            H3CachedConditioning::capture(&conditioning(2, DType::F32, 2), 2, 0, route()).is_err(),
            "a non-BF16 state tensor is not the released conditioner output"
        );
        assert!(
            H3CachedConditioning::capture(&conditioning(2, DType::BF16, 3), 2, 0, route()).is_err(),
            "one tag per row or nothing"
        );
        assert!(
            H3CachedConditioning::capture(&conditioning(2, DType::BF16, 2), 3, 0, route()).is_err(),
            "the frozen text-row count must equal the state rows"
        );
        let narrow = H3TextConditioning {
            states: Tensor::zeros((1, 2, 4_096), DType::BF16, &Device::Cpu).unwrap(),
            tags: vec![H3ModalityTag::Text; 2],
            lifetime_probe: None,
        };
        assert!(
            H3CachedConditioning::capture(&narrow, 2, 0, route()).is_err(),
            "the H3 state width is 5120 and nothing else"
        );
    }

    #[test]
    fn entry_above_budget_is_not_stored() {
        let mut cache = H3ConditionerCache::with_budget(ROW_BYTES - 1);
        assert!(!cache.insert("a".into(), entry(1)));
        assert!(cache.get("a").is_none());
        assert_eq!(cache.resident_bytes, 0);
    }

    #[test]
    fn budget_evicts_least_recently_used_by_bytes() {
        let mut cache = H3ConditionerCache::with_budget(ROW_BYTES * 2 + 1);
        assert!(cache.insert("a".into(), entry(1)));
        assert!(cache.insert("b".into(), entry(1)));
        assert_eq!(cache.resident_bytes, ROW_BYTES * 2);
        assert!(cache.insert("c".into(), entry(1)));
        assert!(cache.get("a").is_none(), "the oldest entry leaves first");
        assert!(cache.get("b").is_some());
        assert!(cache.get("c").is_some());
        assert_eq!(cache.resident_bytes, ROW_BYTES * 2);

        // One entry twice the size evicts as many neighbours as it needs.
        assert!(cache.insert("d".into(), entry(2)));
        assert_eq!(cache.resident_bytes, 2 * ROW_BYTES);
        assert!(cache.get("b").is_none());
        assert!(cache.get("c").is_none());
        assert!(cache.get("d").is_some());
    }

    #[test]
    fn get_promotes_entry() {
        let mut cache = H3ConditionerCache::with_budget(ROW_BYTES * 2 + 1);
        cache.insert("a".into(), entry(1));
        cache.insert("b".into(), entry(1));
        assert!(cache.get("a").is_some());
        cache.insert("c".into(), entry(1));
        assert!(
            cache.get("b").is_none(),
            "the promoted entry must outlive its older neighbour"
        );
        assert!(cache.get("a").is_some());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn reinsert_of_a_live_key_does_not_double_charge() {
        let mut cache = H3ConditionerCache::with_budget(ROW_BYTES * 4);
        cache.insert("a".into(), entry(1));
        cache.insert("a".into(), entry(1));
        assert_eq!(cache.resident_bytes, ROW_BYTES);
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.order.len(), 1);
    }

    #[test]
    fn clear_returns_resident_bytes() {
        let mut cache = H3ConditionerCache::with_budget(ROW_BYTES * 4);
        cache.insert("a".into(), entry(1));
        cache.insert("b".into(), entry(2));
        assert_eq!(cache.clear(), ROW_BYTES * 3);
        assert_eq!(cache.resident_bytes, 0);
        assert!(cache.get("a").is_none());
        assert_eq!(cache.clear(), 0);
    }

    #[test]
    fn budget_from_env_parses_off_and_mib() {
        assert_eq!(
            budget_from_env(None),
            Some(H3_CONDITIONER_CACHE_DEFAULT_BUDGET_BYTES)
        );
        assert_eq!(
            budget_from_env(Some("  ")),
            Some(H3_CONDITIONER_CACHE_DEFAULT_BUDGET_BYTES)
        );
        for disabled in ["0", "off", "OFF", " false ", "no", "disabled"] {
            assert_eq!(budget_from_env(Some(disabled)), None, "{disabled}");
        }
        assert_eq!(budget_from_env(Some("1")), Some(1 << 20));
        assert_eq!(budget_from_env(Some(" 256 ")), Some(256 << 20));
        assert_eq!(
            budget_from_env(Some("512 MiB")),
            Some(H3_CONDITIONER_CACHE_DEFAULT_BUDGET_BYTES),
            "an unparseable value keeps the default rather than silently disabling reuse"
        );
    }

    /// A budget nothing can reach is not a budget: at `u64::MAX` bytes neither
    /// the per-entry refusal nor the eviction loop can fire again, so an
    /// operator typo turns a bounded cache into an unbounded one.
    #[test]
    fn an_over_large_budget_is_clamped_rather_than_saturated() {
        assert_eq!(
            budget_from_env(Some("99999999999")),
            Some(H3_CONDITIONER_CACHE_MAX_BUDGET_MIB << 20),
            "a huge MiB count clamps to the ceiling instead of saturating to ~u64::MAX"
        );
        assert_eq!(
            budget_from_env(Some(&u64::MAX.to_string())),
            Some(H3_CONDITIONER_CACHE_MAX_BUDGET_MIB << 20)
        );
        assert_eq!(
            budget_from_env(Some("65536")),
            Some(H3_CONDITIONER_CACHE_MAX_BUDGET_MIB << 20),
            "the ceiling itself is accepted unchanged"
        );
        let ceiling = budget_from_env(Some("99999999999")).expect("clamped, not disabled");
        let mut cache = H3ConditionerCache::with_budget(ceiling);
        assert!(cache.resident_bytes.checked_add(ceiling).is_some());
        cache.insert("held".into(), entry(1));
        assert_eq!(cache.resident_bytes, ROW_BYTES);
    }

    /// Capture scope is inert whatever the environment says, and the inertness
    /// is structural: the global map is built with a zero budget, which every
    /// entry exceeds because `capture` refuses a zero-row state.
    #[test]
    #[cfg(feature = "h3-private-uat")]
    fn cache_is_inert_in_capture_builds() {
        let _serialized = process_global_test_guard();
        assert_eq!(
            budget_bytes(),
            None,
            "capture scope must always exercise the conditioner"
        );
        assert!(!enabled());
        insert("capture".into(), entry(1));
        assert!(lookup("capture").is_none());
        assert_eq!(h3_conditioner_cache_resident_bytes(), 0);
        assert_eq!(h3_conditioner_cache_clear(), 0);
    }

    /// `lookup`, `insert` and `h3_conditioner_cache_clear` address ONE map.
    ///
    /// The budget is installed explicitly rather than inherited from the
    /// environment: `budget_bytes()` resolves `MOLD_H3_CONDITIONER_CACHE` once
    /// per process through a `OnceLock`, so a developer with the knob set to
    /// `off` would otherwise fail this test, and a later test driving the same
    /// map would race it.
    #[test]
    #[cfg(not(feature = "h3-private-uat"))]
    fn process_global_reuse_serves_and_clears() {
        let _serialized = process_global_test_guard();
        install_process_global_budget_for_test(ROW_BYTES * 4);
        assert_eq!(h3_conditioner_cache_resident_bytes(), 0);
        insert("process-global".into(), entry(1));
        let served = lookup("process-global").expect("a stored entry is served back");
        assert_eq!(served.text_rows(), 1);
        assert_eq!(h3_conditioner_cache_resident_bytes(), ROW_BYTES);
        assert_eq!(h3_conditioner_cache_clear(), ROW_BYTES);
        assert!(lookup("process-global").is_none());
        assert_eq!(h3_conditioner_cache_resident_bytes(), 0);
    }

    /// A zero budget refuses every entry, which is how the disabled process and
    /// every capture-scope build stay inert without a second `enabled()` gate
    /// inside `lookup`/`insert`.
    #[test]
    fn a_zero_budget_map_holds_nothing() {
        let _serialized = process_global_test_guard();
        install_process_global_budget_for_test(0);
        insert("disabled".into(), entry(1));
        assert!(lookup("disabled").is_none());
        assert_eq!(h3_conditioner_cache_resident_bytes(), 0);
        assert_eq!(h3_conditioner_cache_clear(), 0);
    }
}
