//! Authorization-bound MiniMax H3 transformer primitive capture adapter.
//!
//! This development-only binary authenticates the complete reviewed FL2VA and
//! Ref2VA artifact set, loads only the selected task's first token-refiner and
//! main block plus final heads, and observes the production Candle equations.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use candle_core::{DType, Device, Tensor};
use mold_candle::minimax_h3::{
    capture_h3_private_transformer_primitives, H3CheckpointComponent, H3CheckpointSource,
    H3PrivateTransformerCaptureInput, H3TransformerConfig, H3TransformerTask,
};
use mold_core::secure_file::{open_regular_file_no_follow, sha256_open_file};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[cfg(unix)]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

const RAW_SCHEMA: &str = "mold.minimax-h3.transformer-raw-output.v1";
const AUTHORIZATION_SCHEMA: &str = "mold.minimax-h3.authorization.v1";
const AUTHORITY_SET_SCHEMA: &str = "mold.minimax-h3.component-authority-set.v1";
const AUTHORITY_SET_SHA256: &str =
    "b6a4e455ea28a0acbd3b4e2180bda495bb35b32e98da131682590a254148135b";
const AUTHORIZATION_SOURCE_SHA256: &str =
    "8cd4d6e52cff34d7d39721ebab13b8c1187aa87aafc1c4ae2a16609186f22f1d";
const CLAIM_MARKER: &str = "mold.minimax-h3.private-uat-transformer-capture.v1";
const MODEL_REVISION: &str = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86";
const LICENSE_SHA256: &str = "59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44";
const MANIFEST_SHA256: &str = "f6fa8bbc78ca2ee1e4811aedb51413e176f71ca009595cebbd3ef23845d9d183";
const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;
const MAX_AUTHORIZATION_BYTES: u64 = 64 * 1024;

#[derive(Debug)]
struct Arguments {
    model_root: PathBuf,
    manifest: PathBuf,
    authorization_record: PathBuf,
    output: PathBuf,
    task: H3TransformerTask,
    component: H3CheckpointComponent,
    component_dir: &'static str,
    device_index: usize,
    device_label: String,
    source_revision: String,
    input_sha256: &'static str,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorizationRecord {
    schema_version: String,
    family: String,
    decision: String,
    license_revision: String,
    license_sha256: String,
    approved_scopes: Vec<String>,
    source_document_path: PathBuf,
    source_document_sha256: String,
    review_reference: String,
}

#[derive(Debug)]
struct AuthenticatedComponent {
    record: ManifestComponent,
    file: File,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestComponent {
    id: String,
    source_id: String,
    relative_path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct CheckpointIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct RawOutput {
    schema_version: &'static str,
    claim_marker: &'static str,
    task: &'static str,
    component: &'static str,
    device: String,
    source_revision: String,
    authorization_source_sha256: &'static str,
    input_sha256: &'static str,
    authority_set_sha256: &'static str,
    tensors: Vec<RawTensor>,
}

#[derive(Serialize)]
struct RawTensor {
    key: &'static str,
    dtype: &'static str,
    shape: Vec<usize>,
    little_endian_base64: String,
}

fn usage() -> &'static str {
    "usage: h3_transformer_capture --model-root <absolute-model-root> \
     --manifest <absolute-manifest.json> --authorization-record <absolute-record.json> \
     --source-revision <40-lower-hex> --task <fl2va|ref2va> --device <cuda:index> \
     --raw-output <absolute-new.json>"
}

fn absolute(value: String, label: &str) -> Result<PathBuf> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        bail!("{label} must be absolute")
    }
    Ok(path)
}

fn parse_args() -> Result<Arguments> {
    let mut values = env::args().skip(1);
    let mut named = BTreeMap::new();
    while let Some(flag) = values.next() {
        if !flag.starts_with("--") || named.contains_key(&flag) {
            bail!(usage())
        }
        named.insert(flag, values.next().ok_or_else(|| anyhow!(usage()))?);
    }
    let required = |name: &str| {
        named
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("missing {name}; {}", usage()))
    };
    if named.len() != 7 {
        bail!(usage())
    }
    let (task, component, component_dir, input_sha256) = match required("--task")?.as_str() {
        "fl2va" => (
            H3TransformerTask::T2VaFl2Va,
            H3CheckpointComponent::Transformer,
            "transformer",
            "f139e36713bb2d0a843cf944db59a832eb31f561fc5a33e3877683564183f91f",
        ),
        "ref2va" => (
            H3TransformerTask::Ref2Va,
            H3CheckpointComponent::TransformerRef,
            "transformer_ref",
            "af7f4983e006ea6426719a6dccfbfaa0415abe526dc91d9525f5f431ff0c614f",
        ),
        _ => bail!("task must be fl2va or ref2va"),
    };
    let source_revision = required("--source-revision")?;
    if source_revision.len() != 40
        || !source_revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("source revision must be 40 lowercase hexadecimal characters")
    }
    let device_label = required("--device")?;
    let device_index = device_label
        .strip_prefix("cuda:")
        .ok_or_else(|| anyhow!("capture device must be indexed CUDA"))?
        .parse::<usize>()
        .context("CUDA device index is invalid")?;
    Ok(Arguments {
        model_root: absolute(required("--model-root")?, "model root")?,
        manifest: absolute(required("--manifest")?, "manifest")?,
        authorization_record: absolute(
            required("--authorization-record")?,
            "authorization record",
        )?,
        output: absolute(required("--raw-output")?, "raw output")?,
        task,
        component,
        component_dir,
        device_index,
        device_label,
        source_revision,
        input_sha256,
    })
}

fn read_bounded(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let metadata =
        fs::symlink_metadata(path).with_context(|| format!("failed to inspect {label}"))?;
    if !metadata.file_type().is_file() || metadata.len() > maximum {
        bail!("{label} must be a bounded direct regular file")
    }
    #[cfg(unix)]
    if metadata.mode() & 0o022 != 0 {
        bail!("{label} must not be group/other writable")
    }
    let mut file = open_regular_file_no_follow(path)
        .with_context(|| format!("failed to open {label} without following links"))?;
    let opened = file.metadata()?;
    #[cfg(unix)]
    if metadata.dev() != opened.dev() || metadata.ino() != opened.ino() {
        bail!("{label} changed while opening")
    }
    let mut bytes = Vec::with_capacity(opened.len() as usize);
    file.read_to_end(&mut bytes)?;
    if file.metadata()?.len() != opened.len() {
        bail!("{label} changed while reading")
    }
    Ok(bytes)
}

fn validate_authorization(path: &Path) -> Result<()> {
    let record: AuthorizationRecord = serde_json::from_slice(&read_bounded(
        path,
        MAX_AUTHORIZATION_BYTES,
        "authorization record",
    )?)
    .context("authorization record is malformed")?;
    if record.schema_version != AUTHORIZATION_SCHEMA
        || record.family != "minimax-h3"
        || record.decision != "approved"
        || record.license_revision != MODEL_REVISION
        || record.license_sha256 != LICENSE_SHA256
        || record.source_document_sha256 != AUTHORIZATION_SOURCE_SHA256
        || record.review_reference.trim() != record.review_reference
        || record.review_reference.is_empty()
    {
        bail!("authorization record differs from reviewed evidence")
    }
    let scopes = record
        .approved_scopes
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if scopes.len() != record.approved_scopes.len() {
        bail!("authorization record repeats an approved scope")
    }
    for required in [
        "checkpoint-execution",
        "fixture-capture",
        "generated-output-retention",
    ] {
        if !scopes.contains(required) {
            bail!("authorization record lacks required scope {required}")
        }
    }
    let source = record.source_document_path.canonicalize()?;
    if source == path || source.parent() != path.parent() {
        bail!("authorization source must be a distinct sibling of its record")
    }
    let source_bytes = read_bounded(&source, 1024 * 1024, "authorization source")?;
    let source_sha = Sha256::digest(source_bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if source_sha != AUTHORIZATION_SOURCE_SHA256 {
        bail!("authorization source differs from reviewed evidence")
    }
    Ok(())
}

fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}

fn authority_digest(components: &[ManifestComponent]) -> Result<String> {
    let authority = serde_json::json!({
        "components": components.iter().map(|component| serde_json::json!({
            "id": component.id,
            "sha256": component.sha256,
        })).collect::<Vec<_>>(),
        "schema_version": AUTHORITY_SET_SCHEMA,
    });
    Ok(Sha256::digest(canonical_json(&authority)?)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn validate_relative_path(value: &str) -> Result<()> {
    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        bail!("manifest component path is not a canonical relative path")
    }
    Ok(())
}

fn authenticate_components(args: &Arguments) -> Result<Vec<AuthenticatedComponent>> {
    let manifest_bytes = read_bounded(&args.manifest, MAX_MANIFEST_BYTES, "conformance manifest")?;
    let manifest_sha = Sha256::digest(&manifest_bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if manifest_sha != MANIFEST_SHA256 {
        bail!("conformance manifest differs from the reviewed capture manifest")
    }
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    let mut components = manifest
        .get("component_indexes")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("manifest lacks component indexes"))?
        .iter()
        .map(|value| serde_json::from_value::<ManifestComponent>(value.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    components.retain(|component| {
        component.source_id == "minimax-official-model"
            && (component.relative_path.starts_with("transformer/")
                || component.relative_path.starts_with("transformer_ref/"))
    });
    components.sort_by(|left, right| left.id.cmp(&right.id));
    if components.len() != 32 || authority_digest(&components)? != AUTHORITY_SET_SHA256 {
        bail!("manifest transformer authority differs from the reviewed set")
    }
    let model_root = args.model_root.canonicalize()?;
    let mut authenticated = Vec::with_capacity(components.len());
    for component in components {
        validate_relative_path(&component.relative_path)?;
        let path = model_root.join(&component.relative_path);
        let resolved = path
            .canonicalize()
            .with_context(|| format!("missing component {}", component.id))?;
        if !resolved.starts_with(&model_root) {
            bail!("component {} escapes the model root", component.id)
        }
        let file = open_regular_file_no_follow(&resolved)?;
        let observed = sha256_open_file(&file)?;
        if observed != component.sha256 {
            bail!(
                "component {} differs from its reviewed SHA-256",
                component.id
            )
        }
        authenticated.push(AuthenticatedComponent {
            record: component,
            file,
        });
    }
    Ok(authenticated)
}

fn descriptor_path(file: &File) -> PathBuf {
    PathBuf::from(format!("/dev/fd/{}", file.as_raw_fd()))
}

fn retained_bytes(file: &File, maximum: u64, label: &str) -> Result<Vec<u8>> {
    let mut reader = file.try_clone()?;
    reader.rewind()?;
    if reader.metadata()?.len() > maximum {
        bail!("{label} exceeds its size bound")
    }
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn load_tensor(
    mapped: &candle_core::safetensors::MmapedSafetensors,
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    mapped
        .load(name, device)
        .with_context(|| format!("failed to load official tensor {name}"))
}

fn insert_linear(
    tensors: &mut HashMap<String, Tensor>,
    mapped: &candle_core::safetensors::MmapedSafetensors,
    source: &str,
    target: &str,
    device: &Device,
) -> Result<()> {
    tensors.insert(target.to_owned(), load_tensor(mapped, source, device)?);
    Ok(())
}

fn insert_attention(
    tensors: &mut HashMap<String, Tensor>,
    mapped: &candle_core::safetensors::MmapedSafetensors,
    source: &str,
    target: &str,
    config: &H3TransformerConfig,
    device: &Device,
) -> Result<()> {
    let q = load_tensor(mapped, &format!("{source}.to_q.weight"), device)?;
    let k = load_tensor(mapped, &format!("{source}.to_k.weight"), device)?;
    let v = load_tensor(mapped, &format!("{source}.to_v.weight"), device)?;
    let grouped = Tensor::stack(
        &[
            q.reshape((
                config.num_attention_heads,
                config.attention_head_dim,
                config.hidden_size,
            ))?,
            k.reshape((
                config.num_attention_heads,
                config.attention_head_dim,
                config.hidden_size,
            ))?,
            v.reshape((
                config.num_attention_heads,
                config.attention_head_dim,
                config.hidden_size,
            ))?,
        ],
        1,
    )?
    .reshape((3 * config.attention_inner_dim(), config.hidden_size))?;
    tensors.insert(format!("{target}.qkv_proj.weight"), grouped);
    for (source_suffix, target_suffix) in [
        ("norm_q.weight", "q_norm.weight"),
        ("norm_k.weight", "k_norm.weight"),
        ("to_out.0.weight", "out_proj.weight"),
    ] {
        insert_linear(
            tensors,
            mapped,
            &format!("{source}.{source_suffix}"),
            &format!("{target}.{target_suffix}"),
            device,
        )?;
    }
    Ok(())
}

fn insert_feed_forward_input(
    tensors: &mut HashMap<String, Tensor>,
    mapped: &candle_core::safetensors::MmapedSafetensors,
    source: &str,
    target: &str,
    config: &H3TransformerConfig,
    device: &Device,
) -> Result<()> {
    let value_gate = load_tensor(mapped, source, device)?;
    let value = value_gate.narrow(0, 0, config.ffn_hidden_size)?;
    let gate = value_gate.narrow(0, config.ffn_hidden_size, config.ffn_hidden_size)?;
    tensors.insert(target.to_owned(), Tensor::cat(&[gate, value], 0)?);
    Ok(())
}

fn canonical_tensors(
    mapped: &candle_core::safetensors::MmapedSafetensors,
    config: &H3TransformerConfig,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();
    for (source_prefix, target_prefix, with_adaln) in [
        (
            "token_refiner.refiner_blocks.0",
            "token_refiner.blocks.0",
            false,
        ),
        ("transformer_blocks.0", "blocks.0", true),
    ] {
        for suffix in ["norm1.weight", "norm2.weight"] {
            insert_linear(
                &mut tensors,
                mapped,
                &format!("{source_prefix}.{suffix}"),
                &format!("{target_prefix}.{suffix}"),
                device,
            )?;
        }
        insert_attention(
            &mut tensors,
            mapped,
            &format!("{source_prefix}.attn"),
            &format!("{target_prefix}.attn"),
            config,
            device,
        )?;
        insert_feed_forward_input(
            &mut tensors,
            mapped,
            &format!("{source_prefix}.ff.net.0.proj.weight"),
            &format!("{target_prefix}.mlp.fc1.weight"),
            config,
            device,
        )?;
        insert_linear(
            &mut tensors,
            mapped,
            &format!("{source_prefix}.ff.net.2.weight"),
            &format!("{target_prefix}.mlp.fc2.weight"),
            device,
        )?;
        if with_adaln {
            for suffix in ["weight", "bias"] {
                insert_linear(
                    &mut tensors,
                    mapped,
                    &format!("{source_prefix}.adaln_proj.linear.{suffix}"),
                    &format!("{target_prefix}.adaln_proj.linear.{suffix}"),
                    device,
                )?;
            }
        }
    }
    for (source, target) in [
        ("norm_out.norm.weight", "final_layer.norm.weight"),
        (
            "norm_out.linear.weight",
            "final_layer.adaln_proj.linear.weight",
        ),
        ("norm_out.linear.bias", "final_layer.adaln_proj.linear.bias"),
        ("proj_out.weight", "final_layer.video_out.weight"),
        ("proj_out.bias", "final_layer.video_out.bias"),
        ("audio_proj_out.weight", "final_layer.audio_out.weight"),
        ("audio_proj_out.bias", "final_layer.audio_out.bias"),
    ] {
        insert_linear(&mut tensors, mapped, source, target, device)?;
    }
    let inv_freq = (0..config.rope_inv_freq_len)
        .map(|index| 1f32 / 10_000f32.powf(index as f32 / config.rope_inv_freq_len as f32))
        .collect::<Vec<_>>();
    tensors.insert(
        "rope.inv_freq".into(),
        Tensor::from_vec(inv_freq, config.rope_inv_freq_len, device)?,
    );
    Ok(tensors)
}

fn deterministic(count: usize, modulus: usize, offset: f32, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        (0..count)
            .map(|index| (index % modulus) as f32 / 16.0 - offset)
            .collect(),
        count,
        device,
    )?)
}

fn raw_tensor(key: &'static str, tensor: &Tensor) -> Result<RawTensor> {
    let shape = tensor.dims().to_vec();
    let tensor = tensor.flatten_all()?.to_device(&Device::Cpu)?;
    let (dtype, bytes) = match tensor.dtype() {
        DType::BF16 => (
            "bfloat16",
            tensor
                .to_vec1::<half::bf16>()?
                .into_iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .collect::<Vec<_>>(),
        ),
        DType::F32 => (
            "float32",
            tensor
                .to_vec1::<f32>()?
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        ),
        dtype => bail!("capture tensor {key} has unsupported dtype {dtype:?}"),
    };
    Ok(RawTensor {
        key,
        dtype,
        shape,
        little_endian_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
    })
}

fn execute(args: &Arguments) -> Result<RawOutput> {
    validate_authorization(&args.authorization_record)?;
    let components = authenticate_components(args)?;
    let index_relative = format!(
        "{}/diffusion_pytorch_model.safetensors.index.json",
        args.component_dir
    );
    let index_component = components
        .iter()
        .find(|component| component.record.relative_path == index_relative)
        .ok_or_else(|| anyhow!("authenticated authority lacks selected checkpoint index"))?;
    let index: CheckpointIndex = serde_json::from_slice(&retained_bytes(
        &index_component.file,
        MAX_MANIFEST_BYTES,
        "transformer checkpoint index",
    )?)?;
    let required_names = [
        "token_refiner.refiner_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.to_q.weight",
        "proj_out.weight",
        "audio_proj_out.weight",
    ];
    let shard_names = required_names
        .iter()
        .map(|name| {
            index
                .weight_map
                .get(*name)
                .cloned()
                .ok_or_else(|| anyhow!("checkpoint index lacks {name}"))
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let shard_paths = shard_names
        .iter()
        .map(|name| {
            let relative = format!("{}/{name}", args.component_dir);
            components
                .iter()
                .find(|component| component.record.relative_path == relative)
                .map(|component| descriptor_path(&component.file))
                .ok_or_else(|| anyhow!("authenticated authority lacks selected shard {name}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let device = Device::new_cuda(args.device_index)?;
    // SAFETY: every mapped file was authenticated above and remains open and
    // unchanged until all loaded tensors and the mapping are dropped.
    let mapped = unsafe { candle_core::safetensors::MmapedSafetensors::multi(&shard_paths)? };
    let config = H3TransformerConfig {
        num_layers: 1,
        token_refiner_num_layers: 1,
        ..H3TransformerConfig::default()
    };
    let tensors = canonical_tensors(&mapped, &config, &device)?;
    let source = H3CheckpointSource::from_tensors(args.component, tensors, &device)?;
    let token_hidden = deterministic(2 * config.hidden_size, 31, 0.75, &device)?
        .reshape((1, 2, config.hidden_size))?
        .to_dtype(DType::BF16)?;
    let packed_hidden = deterministic(6 * config.hidden_size, 37, 1.0, &device)?
        .reshape((1, 6, config.hidden_size))?
        .to_dtype(DType::BF16)?;
    let adaln_input = deterministic(2 * config.time_embed_dim, 29, 0.5, &device)?
        .reshape((2, config.time_embed_dim))?
        .to_dtype(DType::BF16)?;
    let positions = Tensor::new(
        &[
            [0f32, 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 0., 1.],
        ],
        &device,
    )?;
    let adaln_indices = Tensor::new(&[1u32, 0, 2, 3, 5, 3], &device)?;
    let timestep_indices = Tensor::new(&[0u32, 0, 0, 1, 1, 1], &device)?;
    let video_indices = Tensor::new(&[1u32, 3], &device)?;
    let audio_indices = Tensor::new(&[2u32, 4], &device)?;
    let capture = capture_h3_private_transformer_primitives(
        &config,
        args.task,
        source,
        H3PrivateTransformerCaptureInput {
            token_hidden: &token_hidden,
            packed_hidden: &packed_hidden,
            adaln_input: &adaln_input,
            positions: &positions,
            adaln_indices: &adaln_indices,
            timestep_indices: &timestep_indices,
            video_indices: &video_indices,
            audio_indices: &audio_indices,
        },
    )?;
    device.synchronize()?;
    for component in &components {
        let observed = sha256_open_file(&component.file)?;
        if observed != component.record.sha256 {
            bail!("component {} changed during execution", component.record.id)
        }
    }
    let task = match capture.task {
        H3TransformerTask::T2VaFl2Va => "fl2va",
        H3TransformerTask::Ref2Va => "ref2va",
    };
    let component = match capture.component {
        H3CheckpointComponent::Transformer => "transformer",
        H3CheckpointComponent::TransformerRef => "transformer_ref",
    };
    Ok(RawOutput {
        schema_version: RAW_SCHEMA,
        claim_marker: CLAIM_MARKER,
        task,
        component,
        device: args.device_label.clone(),
        source_revision: args.source_revision.clone(),
        authorization_source_sha256: AUTHORIZATION_SOURCE_SHA256,
        input_sha256: args.input_sha256,
        authority_set_sha256: AUTHORITY_SET_SHA256,
        tensors: vec![
            raw_tensor("token-refiner", &capture.token_refiner)?,
            raw_tensor("normalized-q", &capture.normalized_q)?,
            raw_tensor("normalized-k", &capture.normalized_k)?,
            raw_tensor("rotated-q", &capture.rotated_q)?,
            raw_tensor("rotated-k", &capture.rotated_k)?,
            raw_tensor("adaln", &capture.adaln)?,
            raw_tensor("block-output", &capture.block_output)?,
            raw_tensor("video-head", &capture.video_output)?,
            raw_tensor("audio-head", &capture.audio_output)?,
        ],
    })
}

fn write_output(path: &Path, output: &RawOutput) -> Result<()> {
    if path.exists() {
        bail!("raw output already exists")
    }
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("raw output lacks parent"))?;
    let parent = parent.canonicalize()?;
    #[cfg(unix)]
    if fs::metadata(&parent)?.mode() & 0o077 != 0 {
        bail!("raw output parent must be owner-only")
    }
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options.open(path)?;
    serde_json::to_writer(&mut file, output)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let output = execute(&args)?;
    write_output(&args.output, &output)
}
