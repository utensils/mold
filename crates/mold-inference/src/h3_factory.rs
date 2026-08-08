//! Contract-only MiniMax H3 factory authority.
//!
//! This module is the typed seam between server admission and the private H3
//! Candle backend. It carries only digests and immutable scheduling facts; it
//! never receives artifact paths or bytes. Construction deliberately creates
//! the backend's unavailable plan, so neither a qualified admission record nor
//! this public type can activate the family by itself.

use anyhow::{anyhow, bail, Result};
use mold_core::minimax_h3::{self as contract, Layout, Task};
use sha2::{Digest, Sha256};

use crate::attention::{AttentionBackend, AttentionChunkPolicy};
use crate::minimax_h3::backend::{
    FrozenH3Fl2VaCandlePlan, H3CandleBackendDevice, H3ComponentRole, H3ValidatedComponentAuthority,
    H3ValidatedComponentSet,
};
use crate::minimax_h3::offload::FrozenH3BlockStreamingPlan;
use crate::minimax_h3::vae_runtime::expected_h3_comfy_vae_artifact_plan_identity;
use crate::minimax_h3::{FrozenH3ConditionerPlacement, H3ConditionerExecution};

const H3_FACTORY_AUTHORITY_SCHEMA_VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum H3FactoryComponentRole {
    Conditioner,
    Transformer,
    VisualVae,
    AudioVae,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryComponentAuthority {
    role: H3FactoryComponentRole,
    content_sha256: String,
    validation_sha256: String,
}

impl H3FactoryComponentAuthority {
    pub fn new(
        role: H3FactoryComponentRole,
        content_sha256: impl Into<String>,
        validation_sha256: impl Into<String>,
    ) -> Result<Self> {
        let authority = Self {
            role,
            content_sha256: content_sha256.into(),
            validation_sha256: validation_sha256.into(),
        };
        authority.validate()?;
        Ok(authority)
    }

    fn validate(&self) -> Result<()> {
        require_sha256(&self.content_sha256, "H3 component content")?;
        require_sha256(&self.validation_sha256, "H3 component validation")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum H3FactoryConditionerPlacement {
    AssignedCudaThenDrop,
    HostCpuThenDrop,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum H3FactoryQuantizationAuthority {
    OfficialBf16,
    ComfyPrunedInt8ConvrotNvfp4Awq {
        transformer_policy_sha256: String,
        qwen_policy_sha256: String,
        pruned_adaln_table_sha256: String,
    },
}

impl H3FactoryQuantizationAuthority {
    fn validate(&self, layout: Layout) -> Result<()> {
        match (self, layout) {
            (Self::OfficialBf16, Layout::OfficialBf16) => Ok(()),
            (
                Self::ComfyPrunedInt8ConvrotNvfp4Awq {
                    transformer_policy_sha256,
                    qwen_policy_sha256,
                    pruned_adaln_table_sha256,
                },
                Layout::ComfyPrunedInt8ConvrotNvfp4Awq,
            ) => {
                require_sha256(transformer_policy_sha256, "H3 transformer quantization")?;
                require_sha256(qwen_policy_sha256, "H3 Qwen quantization")?;
                require_sha256(pruned_adaln_table_sha256, "H3 pruned AdaLN table")
            }
            _ => bail!("MiniMax H3 layout and quantization authorities disagree"),
        }
    }

    fn update_identity(&self, hash: &mut Sha256) {
        match self {
            Self::OfficialBf16 => hash.update(b"official-bf16"),
            Self::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256,
                qwen_policy_sha256,
                pruned_adaln_table_sha256,
            } => {
                hash.update(b"comfy-pruned-int8-convrot-nvfp4-awq\0");
                hash.update(transformer_policy_sha256.as_bytes());
                hash.update([0]);
                hash.update(qwen_policy_sha256.as_bytes());
                hash.update([0]);
                hash.update(pruned_adaln_table_sha256.as_bytes());
            }
        }
    }
}

/// Server-owned inputs that may cross into the H3 factory boundary.
///
/// The constructor validates and fingerprints every field before retaining
/// it. In particular, component authorities are digests only; paths and model
/// bytes remain on the compliance-gated side of the boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct H3FactoryAuthorityInput {
    pub model: String,
    pub device_id: String,
    pub device_ordinal: usize,
    pub compute_capability: (u16, u16),
    pub execution_fingerprint: String,
    pub conditioner_placement: H3FactoryConditionerPlacement,
    pub qwen_parameter_bytes: u64,
    pub qwen_host_resident_parameter_bytes: u64,
    pub qwen_device_resident_parameter_bytes: u64,
    pub qwen_activation_workspace_bytes: u64,
    pub qwen_output_text_rows: u64,
    pub qwen_vision_rows: u64,
    pub resident_block_count: u32,
    pub prefetch_depth: u32,
    pub attention_backend: AttentionBackend,
    pub attention_chunk: AttentionChunkPolicy,
    pub attention_kernel_identity: String,
    pub attention_qualification_sha256: String,
    pub attention_full_noncausal: bool,
    pub attention_lossless: bool,
    pub attention_head_count: u32,
    pub attention_head_dim: u32,
    pub block_offload: bool,
    pub quantization: H3FactoryQuantizationAuthority,
    pub components: Vec<H3FactoryComponentAuthority>,
}

/// Immutable, contract-only authority carried by [`crate::FrozenEngineConfig`].
///
/// Its private backend plan always lacks executable attention/quantization
/// authorities. `validate_for_dispatch` also requires both the public capability
/// contract and production family registry before a future loader can be reached.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenH3FactoryAuthority {
    schema_version: u32,
    backend_plan: FrozenH3Fl2VaCandlePlan,
    comfy_vae_artifact_plan_identity_sha256: Option<String>,
    device_ordinal: usize,
    conditioner_placement: H3FactoryConditionerPlacement,
    qwen_parameter_bytes: u64,
    qwen_host_resident_parameter_bytes: u64,
    qwen_device_resident_parameter_bytes: u64,
    qwen_activation_workspace_bytes: u64,
    qwen_output_text_rows: u64,
    qwen_vision_rows: u64,
    attention_backend: AttentionBackend,
    attention_chunk: AttentionChunkPolicy,
    attention_kernel_identity: String,
    attention_qualification_sha256: String,
    attention_full_noncausal: bool,
    attention_lossless: bool,
    attention_head_count: u32,
    attention_head_dim: u32,
    block_offload: bool,
    quantization: H3FactoryQuantizationAuthority,
    identity_sha256: String,
}

/// Private-only projection of the exact admission record needed to compose
/// the opened-file Comfy VAEs with one already-authorized component backend.
#[cfg(feature = "h3-private-uat")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct H3PrivateVaeFactoryAuthority {
    pub(crate) factory_identity_sha256: String,
    pub(crate) backend_plan_identity_sha256: String,
    pub(crate) vae_artifact_plan_identity_sha256: String,
    pub(crate) component_set_identity_sha256: String,
    pub(crate) canonical_model: String,
    pub(crate) task: Task,
    pub(crate) device_id: String,
    pub(crate) execution_fingerprint: String,
}

impl FrozenH3FactoryAuthority {
    pub fn new_contract_only(input: H3FactoryAuthorityInput) -> Result<Self> {
        let model_contract = contract::capability_contract_for_model(&input.model)
            .ok_or_else(|| anyhow!("{:?} has no MiniMax H3 capability contract", input.model))?;
        if model_contract.generation.runtime_available {
            bail!(
                "contract-only MiniMax H3 factory authority cannot be created for a runnable contract"
            );
        }
        if input.device_id.trim().is_empty() || input.compute_capability.0 == 0 {
            bail!("MiniMax H3 factory authority requires one concrete CUDA route");
        }
        require_sha256(&input.execution_fingerprint, "H3 scheduler execution")?;
        if input.qwen_parameter_bytes == 0
            || input
                .qwen_host_resident_parameter_bytes
                .checked_add(input.qwen_device_resident_parameter_bytes)
                .is_none_or(|bytes| bytes == 0)
            || input.qwen_activation_workspace_bytes == 0
            || input.qwen_output_text_rows == 0
        {
            bail!("MiniMax H3 factory authority requires exact nonzero Qwen memory facts");
        }
        if input.attention_kernel_identity.trim().is_empty()
            || !input.attention_full_noncausal
            || !input.attention_lossless
            || input.attention_head_count != 56
            || input.attention_head_dim != 128
        {
            bail!("MiniMax H3 factory authority requires qualified lossless 56x128 full attention");
        }
        require_sha256(
            &input.attention_qualification_sha256,
            "H3 attention qualification",
        )?;
        if !input.block_offload {
            bail!("MiniMax H3 factory authority requires admitted block streaming");
        }
        input.quantization.validate(model_contract.layout)?;

        let components = H3ValidatedComponentSet::new(
            input
                .components
                .into_iter()
                .map(|component| {
                    component.validate()?;
                    let role = match component.role {
                        H3FactoryComponentRole::Conditioner => H3ComponentRole::Conditioner,
                        H3FactoryComponentRole::Transformer => H3ComponentRole::Transformer,
                        H3FactoryComponentRole::VisualVae => H3ComponentRole::VisualVae,
                        H3FactoryComponentRole::AudioVae => H3ComponentRole::AudioVae,
                    };
                    H3ValidatedComponentAuthority::new(
                        role,
                        component.content_sha256,
                        component.validation_sha256,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        )?;
        let conditioner_execution = match input.conditioner_placement {
            H3FactoryConditionerPlacement::AssignedCudaThenDrop => {
                H3ConditionerExecution::CudaResident
            }
            H3FactoryConditionerPlacement::HostCpuThenDrop => H3ConditionerExecution::CpuOffloaded,
        };
        let conditioner_device = match input.conditioner_placement {
            H3FactoryConditionerPlacement::AssignedCudaThenDrop => input.device_id.clone(),
            H3FactoryConditionerPlacement::HostCpuThenDrop => "cpu".to_string(),
        };
        let conditioner_placement = FrozenH3ConditionerPlacement::new(
            conditioner_device,
            conditioner_execution,
            input.execution_fingerprint.clone(),
            input.qwen_host_resident_parameter_bytes,
            input.qwen_device_resident_parameter_bytes,
            input.qwen_activation_workspace_bytes,
        )?;
        let block_streaming = FrozenH3BlockStreamingPlan::new(
            input.device_id.clone(),
            input.execution_fingerprint.clone(),
            usize::try_from(input.resident_block_count)
                .map_err(|_| anyhow!("MiniMax H3 resident block count exceeds usize"))?,
            usize::try_from(input.prefetch_depth)
                .map_err(|_| anyhow!("MiniMax H3 prefetch depth exceeds usize"))?,
        )?;
        let backend_plan = FrozenH3Fl2VaCandlePlan::new_unavailable(
            model_contract.canonical_model,
            input.device_id,
            H3CandleBackendDevice::Cuda {
                compute_capability: input.compute_capability,
            },
            input.execution_fingerprint,
            conditioner_placement,
            block_streaming,
            components,
        )?;
        let comfy_vae_artifact_plan_identity_sha256 =
            if model_contract.layout == Layout::ComfyPrunedInt8ConvrotNvfp4Awq {
                Some(
                    expected_h3_comfy_vae_artifact_plan_identity(model_contract.canonical_model)
                        .map_err(|error| anyhow!(error.to_string()))?,
                )
            } else {
                None
            };
        let mut frozen = Self {
            schema_version: H3_FACTORY_AUTHORITY_SCHEMA_VERSION,
            backend_plan,
            comfy_vae_artifact_plan_identity_sha256,
            device_ordinal: input.device_ordinal,
            conditioner_placement: input.conditioner_placement,
            qwen_parameter_bytes: input.qwen_parameter_bytes,
            qwen_host_resident_parameter_bytes: input.qwen_host_resident_parameter_bytes,
            qwen_device_resident_parameter_bytes: input.qwen_device_resident_parameter_bytes,
            qwen_activation_workspace_bytes: input.qwen_activation_workspace_bytes,
            qwen_output_text_rows: input.qwen_output_text_rows,
            qwen_vision_rows: input.qwen_vision_rows,
            attention_backend: input.attention_backend,
            attention_chunk: input.attention_chunk,
            attention_kernel_identity: input.attention_kernel_identity,
            attention_qualification_sha256: input.attention_qualification_sha256,
            attention_full_noncausal: input.attention_full_noncausal,
            attention_lossless: input.attention_lossless,
            attention_head_count: input.attention_head_count,
            attention_head_dim: input.attention_head_dim,
            block_offload: input.block_offload,
            quantization: input.quantization,
            identity_sha256: String::new(),
        };
        frozen.identity_sha256 = frozen_identity(&frozen);
        frozen.validate_frozen()?;
        Ok(frozen)
    }

    pub fn identity_sha256(&self) -> &str {
        &self.identity_sha256
    }

    pub fn component_set_identity_sha256(&self) -> &str {
        self.backend_plan.component_set_identity()
    }

    #[cfg(feature = "h3-private-uat")]
    pub(crate) fn backend_plan_identity_sha256(&self) -> &str {
        self.backend_plan.identity_sha256()
    }

    #[cfg(feature = "h3-private-uat")]
    pub(crate) fn private_vae_adapter_authority(&self) -> Result<H3PrivateVaeFactoryAuthority> {
        self.validate_frozen()?;
        let vae_artifact_plan_identity_sha256 = self
            .comfy_vae_artifact_plan_identity_sha256
            .as_deref()
            .ok_or_else(|| anyhow!("MiniMax H3 factory authority has no private Comfy VAE plan"))?;
        if self.task() != Task::Fl2va {
            bail!("private MiniMax H3 VAE adapter currently requires the FL2VA task authority");
        }
        Ok(H3PrivateVaeFactoryAuthority {
            factory_identity_sha256: self.identity_sha256.clone(),
            backend_plan_identity_sha256: self.backend_plan_identity_sha256().into(),
            vae_artifact_plan_identity_sha256: vae_artifact_plan_identity_sha256.into(),
            component_set_identity_sha256: self.component_set_identity_sha256().into(),
            canonical_model: self.canonical_model().into(),
            task: self.task(),
            device_id: self.device_id().into(),
            execution_fingerprint: self.execution_fingerprint().into(),
        })
    }

    /// Exact logical conditioner authority frozen by server admission.
    ///
    /// Private runtime adapters use this to cross-check the independently
    /// authenticated Qwen/support lease. It deliberately exposes only
    /// digests, never artifact paths or bytes.
    #[cfg(feature = "h3-private-uat")]
    pub(crate) fn conditioner_component_authority(&self) -> (&str, &str) {
        let authority = self
            .backend_plan
            .components()
            .authority(H3ComponentRole::Conditioner)
            .expect("validated H3 component set always contains the conditioner");
        (authority.content_sha256(), authority.validation_sha256())
    }

    pub fn canonical_model(&self) -> &str {
        self.backend_plan.canonical_model()
    }

    pub const fn task(&self) -> Task {
        self.backend_plan.task()
    }

    pub fn device_id(&self) -> &str {
        self.backend_plan.device_id()
    }

    pub const fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub fn compute_capability(&self) -> (u16, u16) {
        match self.backend_plan.backend() {
            H3CandleBackendDevice::Cuda { compute_capability } => compute_capability,
        }
    }

    pub fn execution_fingerprint(&self) -> &str {
        self.backend_plan.execution_fingerprint()
    }

    pub fn attention_qualification_sha256(&self) -> &str {
        &self.attention_qualification_sha256
    }

    pub const fn attention_backend(&self) -> AttentionBackend {
        self.attention_backend
    }

    pub const fn attention_chunk(&self) -> AttentionChunkPolicy {
        self.attention_chunk
    }

    pub const fn conditioner_placement(&self) -> H3FactoryConditionerPlacement {
        self.conditioner_placement
    }

    pub const fn qwen_parameter_bytes(&self) -> u64 {
        self.qwen_parameter_bytes
    }

    pub const fn qwen_host_resident_parameter_bytes(&self) -> u64 {
        self.qwen_host_resident_parameter_bytes
    }

    pub const fn qwen_device_resident_parameter_bytes(&self) -> u64 {
        self.qwen_device_resident_parameter_bytes
    }

    pub const fn qwen_activation_workspace_bytes(&self) -> u64 {
        self.qwen_activation_workspace_bytes
    }

    pub const fn qwen_output_text_rows(&self) -> u64 {
        self.qwen_output_text_rows
    }

    pub const fn qwen_vision_rows(&self) -> u64 {
        self.qwen_vision_rows
    }

    pub fn resident_block_count(&self) -> usize {
        self.backend_plan.block_streaming().resident_block_count
    }

    pub fn prefetch_depth(&self) -> usize {
        self.backend_plan.block_streaming().prefetch_depth
    }

    pub const fn block_offload(&self) -> bool {
        self.block_offload
    }

    /// Validate the immutable authority carried into the legal-neutral engine
    /// seam without claiming that H3 is runnable.
    ///
    /// Production factory dispatch must still call `validate_for_dispatch`,
    /// which additionally requires the public runtime capability and family
    /// registry. This narrower check exists only so an injected, weight-free
    /// runtime can exercise the engine/worker transaction while those gates
    /// remain closed.
    pub(crate) fn validate_engine_seam(
        &self,
        model: &str,
        gpu_ordinal: usize,
        offload: bool,
    ) -> Result<()> {
        self.validate_frozen()?;
        let request_contract = contract::capability_contract_for_model(model)
            .ok_or_else(|| anyhow!("{model:?} has no MiniMax H3 capability contract"))?;
        if request_contract.canonical_model != self.canonical_model()
            || request_contract.task != self.task()
            || gpu_ordinal != self.device_ordinal
            || offload != self.block_offload
        {
            bail!("MiniMax H3 frozen engine authority changed before construction");
        }
        Ok(())
    }

    pub fn quantization(&self) -> &H3FactoryQuantizationAuthority {
        &self.quantization
    }

    fn validate_frozen(&self) -> Result<()> {
        if self.schema_version != H3_FACTORY_AUTHORITY_SCHEMA_VERSION {
            bail!("MiniMax H3 factory authority uses an unsupported schema version");
        }
        self.backend_plan.validate()?;
        let expected_vae_plan =
            if self.backend_plan.layout() == Layout::ComfyPrunedInt8ConvrotNvfp4Awq {
                Some(
                    expected_h3_comfy_vae_artifact_plan_identity(self.canonical_model())
                        .map_err(|error| anyhow!(error.to_string()))?,
                )
            } else {
                None
            };
        if self.comfy_vae_artifact_plan_identity_sha256 != expected_vae_plan {
            bail!("MiniMax H3 factory VAE artifact plan changed after admission");
        }
        let conditioner_memory = &self.backend_plan.conditioner_placement().memory;
        if conditioner_memory.resident_parameter_bytes
            != self
                .qwen_host_resident_parameter_bytes
                .checked_add(self.qwen_device_resident_parameter_bytes)
                .ok_or_else(|| anyhow!("MiniMax H3 Qwen resident bytes overflow"))?
            || conditioner_memory.activation_workspace_bytes != self.qwen_activation_workspace_bytes
        {
            bail!("MiniMax H3 conditioner placement differs from frozen Qwen memory facts");
        }
        if self.backend_plan.block_streaming().resident_block_count > 50
            || self.backend_plan.block_streaming().prefetch_depth > 2
        {
            bail!("MiniMax H3 factory authority changed its streaming bounds");
        }
        if self.attention_kernel_identity.trim().is_empty()
            || !self.attention_full_noncausal
            || !self.attention_lossless
            || self.attention_head_count != 56
            || self.attention_head_dim != 128
            || !self.block_offload
            || self.qwen_parameter_bytes == 0
            || self
                .qwen_host_resident_parameter_bytes
                .checked_add(self.qwen_device_resident_parameter_bytes)
                .is_none_or(|bytes| bytes == 0)
            || self.qwen_activation_workspace_bytes == 0
            || self.qwen_output_text_rows == 0
        {
            bail!("MiniMax H3 factory attention or offload authority changed after admission");
        }
        require_sha256(
            &self.attention_qualification_sha256,
            "H3 attention qualification",
        )?;
        self.quantization.validate(self.backend_plan.layout())?;
        require_sha256(&self.identity_sha256, "H3 factory authority")?;
        if self.identity_sha256 != frozen_identity(self) {
            bail!("MiniMax H3 factory authority changed after admission");
        }
        Ok(())
    }

    pub(crate) fn validate_for_dispatch(
        &self,
        model: &str,
        family: &str,
        gpu_ordinal: usize,
        offload: bool,
        attention_backend: AttentionBackend,
        attention_chunk: AttentionChunkPolicy,
    ) -> Result<()> {
        self.validate_frozen()?;
        let request_contract = contract::capability_contract_for_model(model)
            .ok_or_else(|| anyhow!("{model:?} has no MiniMax H3 capability contract"))?;
        if !contract::is_family(family)
            || request_contract.canonical_model != self.canonical_model()
            || request_contract.task != self.task()
            || gpu_ordinal != self.device_ordinal
            || offload != self.block_offload
            || attention_backend != self.attention_backend
            || attention_chunk != self.attention_chunk
        {
            bail!(
                "MiniMax H3 frozen route, attention, or offload authority changed before dispatch"
            );
        }
        if contract::runnable_capability_contract_for_model(model).is_none()
            || crate::production_family_capability_for_family(family).is_none()
        {
            bail!(
                "MiniMax H3 public capability or production factory registry remains runtime unavailable"
            );
        }
        Ok(())
    }
}

fn frozen_identity(authority: &FrozenH3FactoryAuthority) -> String {
    let mut hash = Sha256::new();
    hash.update(b"mold.minimax-h3.factory-authority.v3\0");
    hash.update(authority.schema_version.to_le_bytes());
    hash.update(authority.backend_plan.identity_sha256().as_bytes());
    hash.update([0]);
    hash.update(
        authority
            .comfy_vae_artifact_plan_identity_sha256
            .as_deref()
            .unwrap_or("no-comfy-vae-plan")
            .as_bytes(),
    );
    hash.update(authority.device_ordinal.to_le_bytes());
    hash.update(match authority.conditioner_placement {
        H3FactoryConditionerPlacement::AssignedCudaThenDrop => b"qwen-cuda".as_slice(),
        H3FactoryConditionerPlacement::HostCpuThenDrop => b"qwen-cpu".as_slice(),
    });
    hash.update(authority.qwen_parameter_bytes.to_le_bytes());
    hash.update(authority.qwen_host_resident_parameter_bytes.to_le_bytes());
    hash.update(authority.qwen_device_resident_parameter_bytes.to_le_bytes());
    hash.update(authority.qwen_activation_workspace_bytes.to_le_bytes());
    hash.update(authority.qwen_output_text_rows.to_le_bytes());
    hash.update(authority.qwen_vision_rows.to_le_bytes());
    hash.update(match authority.attention_backend {
        AttentionBackend::Math => b"math".as_slice(),
        AttentionBackend::Flash => b"flash".as_slice(),
    });
    match authority.attention_chunk {
        AttentionChunkPolicy::Auto => hash.update(b"chunk-auto"),
        AttentionChunkPolicy::Off => hash.update(b"chunk-off"),
        AttentionChunkPolicy::Size(size) => {
            hash.update(b"chunk-size\0");
            hash.update(size.to_le_bytes());
        }
    }
    hash.update(authority.attention_kernel_identity.as_bytes());
    hash.update([0]);
    hash.update(authority.attention_qualification_sha256.as_bytes());
    hash.update([
        u8::from(authority.attention_full_noncausal),
        u8::from(authority.attention_lossless),
        u8::from(authority.block_offload),
    ]);
    hash.update(authority.attention_head_count.to_le_bytes());
    hash.update(authority.attention_head_dim.to_le_bytes());
    authority.quantization.update_identity(&mut hash);
    format!("{:x}", hash.finalize())
}

fn require_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        bail!("{label} fingerprint is not SHA-256")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn authority_for(model: &str) -> FrozenH3FactoryAuthority {
        FrozenH3FactoryAuthority::new_contract_only(H3FactoryAuthorityInput {
            model: model.into(),
            device_id: "gpu-0".into(),
            device_ordinal: 0,
            compute_capability: (8, 9),
            execution_fingerprint: sha('a'),
            conditioner_placement: H3FactoryConditionerPlacement::HostCpuThenDrop,
            qwen_parameter_bytes: 2048,
            qwen_host_resident_parameter_bytes: 2048,
            qwen_device_resident_parameter_bytes: 0,
            qwen_activation_workspace_bytes: 1024,
            qwen_output_text_rows: 1,
            qwen_vision_rows: 0,
            resident_block_count: 8,
            prefetch_depth: 1,
            attention_backend: AttentionBackend::Flash,
            attention_chunk: AttentionChunkPolicy::Off,
            attention_kernel_identity: "flash-attention-v2-sm89".into(),
            attention_qualification_sha256: sha('b'),
            attention_full_noncausal: true,
            attention_lossless: true,
            attention_head_count: 56,
            attention_head_dim: 128,
            block_offload: true,
            quantization: H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256: sha('c'),
                qwen_policy_sha256: sha('d'),
                pruned_adaln_table_sha256: sha('e'),
            },
            components: [
                H3FactoryComponentRole::Conditioner,
                H3FactoryComponentRole::Transformer,
                H3FactoryComponentRole::VisualVae,
                H3FactoryComponentRole::AudioVae,
            ]
            .into_iter()
            .enumerate()
            .map(|(index, role)| {
                H3FactoryComponentAuthority::new(
                    role,
                    sha(char::from(b'1' + index as u8)),
                    sha(char::from(b'5' + index as u8)),
                )
                .unwrap()
            })
            .collect(),
        })
        .unwrap()
    }

    fn authority() -> FrozenH3FactoryAuthority {
        authority_for(contract::FL2VA_COMFY)
    }

    #[test]
    fn factory_authority_binds_backend_route_components_and_runtime_evidence() {
        let authority = authority();
        assert_eq!(authority.canonical_model(), contract::FL2VA_COMFY);
        assert_eq!(authority.device_id(), "gpu-0");
        assert_eq!(authority.device_ordinal(), 0);
        assert_eq!(authority.execution_fingerprint(), sha('a'));
        assert_eq!(authority.identity_sha256().len(), 64);
        assert_eq!(authority.component_set_identity_sha256().len(), 64);
        #[cfg(feature = "h3-private-uat")]
        {
            let vae = authority.private_vae_adapter_authority().unwrap();
            assert_eq!(vae.factory_identity_sha256, authority.identity_sha256());
            assert_eq!(
                vae.backend_plan_identity_sha256,
                authority.backend_plan_identity_sha256()
            );
            assert_eq!(
                vae.component_set_identity_sha256,
                authority.component_set_identity_sha256()
            );
            assert_eq!(
                vae.vae_artifact_plan_identity_sha256,
                expected_h3_comfy_vae_artifact_plan_identity(contract::FL2VA_COMFY).unwrap()
            );
        }
        assert!(authority.block_offload());
        assert!(matches!(
            authority.quantization(),
            H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq { .. }
        ));
    }

    #[test]
    fn post_admission_mutations_are_rejected() {
        for mutate in [
            (|value: &mut FrozenH3FactoryAuthority| value.device_ordinal += 1)
                as fn(&mut FrozenH3FactoryAuthority),
            |value| value.attention_kernel_identity.push_str("-changed"),
            |value| value.block_offload = false,
            |value| {
                value.quantization = H3FactoryQuantizationAuthority::OfficialBf16;
            },
            |value| {
                value.comfy_vae_artifact_plan_identity_sha256 = Some(sha('f'));
            },
        ] {
            let mut changed = authority();
            mutate(&mut changed);
            assert!(changed.validate_frozen().is_err());
        }
    }

    #[test]
    fn exact_authority_still_rejects_while_runtime_and_factory_registry_are_closed() {
        let authority = authority();
        let error = authority
            .validate_for_dispatch(
                contract::FL2VA_COMFY,
                contract::FAMILY,
                0,
                true,
                AttentionBackend::Flash,
                AttentionChunkPolicy::Off,
            )
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("registry remains runtime unavailable"));
    }

    #[test]
    fn ref2va_contract_authority_is_distinct_while_missing_components_fail_closed() {
        let ref2va = authority_for(contract::REF2VA_COMFY);
        assert_eq!(ref2va.task(), Task::Ref2va);
        assert_eq!(ref2va.canonical_model(), contract::REF2VA_COMFY);
        assert_ne!(ref2va.identity_sha256(), authority().identity_sha256());
        let error = ref2va
            .validate_for_dispatch(
                contract::REF2VA_COMFY,
                contract::FAMILY,
                0,
                true,
                AttentionBackend::Flash,
                AttentionChunkPolicy::Off,
            )
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("registry remains runtime unavailable"));

        let input = H3FactoryAuthorityInput {
            model: contract::REF2VA_COMFY.into(),
            device_id: "gpu-0".into(),
            device_ordinal: 0,
            compute_capability: (8, 9),
            execution_fingerprint: sha('a'),
            conditioner_placement: H3FactoryConditionerPlacement::HostCpuThenDrop,
            qwen_parameter_bytes: 2048,
            qwen_host_resident_parameter_bytes: 2048,
            qwen_device_resident_parameter_bytes: 0,
            qwen_activation_workspace_bytes: 1024,
            qwen_output_text_rows: 1,
            qwen_vision_rows: 0,
            resident_block_count: 8,
            prefetch_depth: 1,
            attention_backend: AttentionBackend::Flash,
            attention_chunk: AttentionChunkPolicy::Off,
            attention_kernel_identity: "flash-attention-v2-sm89".into(),
            attention_qualification_sha256: sha('b'),
            attention_full_noncausal: true,
            attention_lossless: true,
            attention_head_count: 56,
            attention_head_dim: 128,
            block_offload: true,
            quantization: H3FactoryQuantizationAuthority::ComfyPrunedInt8ConvrotNvfp4Awq {
                transformer_policy_sha256: sha('c'),
                qwen_policy_sha256: sha('d'),
                pruned_adaln_table_sha256: sha('e'),
            },
            components: Vec::new(),
        };
        assert!(FrozenH3FactoryAuthority::new_contract_only(input).is_err());
    }
}
