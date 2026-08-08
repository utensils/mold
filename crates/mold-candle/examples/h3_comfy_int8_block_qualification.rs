//! Private, release-unregistered qualification probe for one authenticated
//! Comfy MiniMax H3 INT8 ConvRot block.

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candle::{DType, Device, Tensor};
    use mold_candle::minimax_h3::{
        open_h3_comfy_published_int8_checkpoint, H3ComfyNeverCancel, H3ForwardInput,
        H3PackedLayout, H3TransformerTask,
    };

    let mut arguments = std::env::args().skip(1);
    let path = arguments.next().ok_or(
        "usage: h3_comfy_int8_block_qualification <checkpoint> <fl2va|ref2va> [gpu] [block]",
    )?;
    let task = arguments.next().ok_or("missing task")?;
    let gpu = arguments
        .next()
        .as_deref()
        .unwrap_or("0")
        .parse::<usize>()?;
    let block_index = arguments
        .next()
        .as_deref()
        .unwrap_or("0")
        .parse::<usize>()?;
    if block_index != 0 {
        return Err("the isolated execution probe must start with exact block 0".into());
    }
    if arguments.next().is_some() {
        return Err("unexpected extra qualification arguments".into());
    }
    let (task, artifact) = match task.as_str() {
        "fl2va" => (
            H3TransformerTask::T2VaFl2Va,
            mold_candle::minimax_h3::H3ComfyPublishedArtifact::Fl2VaPrunedInt8ConvRot,
        ),
        "ref2va" => (
            H3TransformerTask::Ref2Va,
            mold_candle::minimax_h3::H3ComfyPublishedArtifact::Ref2VaPrunedInt8ConvRot,
        ),
        _ => return Err("task must be fl2va or ref2va".into()),
    };
    let opened = open_h3_comfy_published_int8_checkpoint(
        std::path::Path::new(&path),
        task,
        artifact,
        &H3ComfyNeverCancel,
    )?;
    let checkpoint_identity = opened.checkpoint_identity_sha256().to_owned();
    let device = Device::new_cuda(gpu)?;
    let (transformer, mut loader) = opened.load(&device)?;
    let memory = loader.block_memory(block_index)?;
    let config = transformer.config();
    let layout = H3PackedLayout::new(
        vec![[0.0, 0.0, 0.0], [0.0, -1.0, 0.5], [0.0, 0.0, -1.0]],
        vec![0, 0, 0],
        vec![1, 0, 2],
        vec![1],
        vec![2],
        vec![0],
    )?
    .freeze(&device)?;
    let video = Tensor::zeros((1, 1, config.video_patch_dim()), DType::F32, &device)?;
    let audio = Tensor::zeros((1, 1, config.audio_latent_channels), DType::F32, &device)?;
    let text = Tensor::zeros((1, 1, config.text_dim), DType::F32, &device)?;
    let timesteps = Tensor::new(&[0.5f32], &device)?;
    let mut step = transformer.begin_step(
        H3ForwardInput {
            video_rows: &video,
            audio_rows: &audio,
            text_states: &text,
            timesteps: &timesteps,
        },
        &layout,
    )?;
    let block = loader.load_block(block_index)?;
    step.forward_block(block_index, &block)?;
    drop(block);
    let report = serde_json::json!({
        "schema": "mold.minimax-h3.comfy-int8-block-qualification.v1",
        "task": format!("{task:?}"),
        "block_index": block_index,
        "checkpoint_identity_sha256": checkpoint_identity,
        "content_sha256": loader.content_sha256(),
        "runtime_bytes_read": loader.bytes_read(),
        "encoded_host_bytes": memory.encoded_host_bytes,
        "max_host_read_staging_bytes": memory.max_host_read_staging_bytes,
        "max_device_weight_staging_bytes": memory.max_device_weight_staging_bytes,
        "protected_device_bytes": memory.protected_device_bytes,
        "live_blocks_after_drop": loader.live_block_count(),
        "executed_block_count": 1,
        "factory_activated": false,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("h3_comfy_int8_block_qualification requires --features cuda");
    std::process::exit(2);
}
