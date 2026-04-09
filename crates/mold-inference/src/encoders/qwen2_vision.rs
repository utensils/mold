use anyhow::{bail, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{linear, rms_norm, Conv2d, Conv2dConfig, Linear, Module, RmsNorm, VarBuilder};
use std::cmp::{max, min};

const CONDITION_IMAGE_AREA: u32 = 384 * 384;
const PATCH_SIZE: usize = 14;
const TEMPORAL_PATCH_SIZE: usize = 2;
const SPATIAL_MERGE_SIZE: usize = 2;
const WINDOW_SIZE: usize = 112;
const MIN_PIXELS: usize = 56 * 56;
const MAX_PIXELS: usize = 28 * 28 * 1280;
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.2613026, 0.2757771];

#[derive(Debug, Clone)]
pub(crate) struct Qwen2VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub out_hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_chans: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub window_size: usize,
    pub fullatt_block_indexes: [usize; 4],
}

impl Qwen2VisionConfig {
    pub(crate) fn qwen25_vl() -> Self {
        Self {
            depth: 32,
            hidden_size: 1280,
            out_hidden_size: 3584,
            intermediate_size: 3420,
            num_heads: 16,
            in_chans: 3,
            patch_size: PATCH_SIZE,
            spatial_merge_size: SPATIAL_MERGE_SIZE,
            temporal_patch_size: TEMPORAL_PATCH_SIZE,
            window_size: WINDOW_SIZE,
            fullatt_block_indexes: [7, 15, 23, 31],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Conv3dConfig {
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        }
    }
}

struct Conv3dNoBias {
    conv2d_1: Conv2d,
    conv2d_2: Conv2d,
}

impl Conv3dNoBias {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_sizes: [usize; 3],
        cfg: Conv3dConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let ws = vb.get(
            (
                out_channels,
                in_channels / cfg.groups,
                kernel_sizes[0],
                kernel_sizes[1],
                kernel_sizes[2],
            ),
            "weight",
        )?;
        let w1 = ws.i((.., .., 0, .., ..))?;
        let w2 = ws.i((.., .., 1, .., ..))?;
        let cfg = Conv2dConfig {
            padding: cfg.padding,
            stride: cfg.stride,
            dilation: cfg.dilation,
            groups: cfg.groups,
            cudnn_fwd_algo: None,
        };
        Ok(Self {
            conv2d_1: Conv2d::new(w1.contiguous()?, None, cfg),
            conv2d_2: Conv2d::new(w2.contiguous()?, None, cfg),
        })
    }
}

impl Module for Conv3dNoBias {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs1 = xs.i((.., .., 0, .., ..))?;
        let xs2 = xs.i((.., .., 1, .., ..))?;
        (self.conv2d_1.forward(&xs1)? + self.conv2d_2.forward(&xs2)?)?.unsqueeze(2)
    }
}

struct PatchEmbed {
    proj: Conv3dNoBias,
    in_channels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    hidden_size: usize,
}

impl PatchEmbed {
    fn new(cfg: &Qwen2VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj: Conv3dNoBias::new(
                cfg.in_chans,
                cfg.hidden_size,
                [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size],
                Conv3dConfig {
                    stride: cfg.patch_size,
                    ..Default::default()
                },
                vb.pp("proj"),
            )?,
            in_channels: cfg.in_chans,
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
            hidden_size: cfg.hidden_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.reshape((
            (),
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ))?;
        self.proj
            .forward(&xs)?
            .reshape(((), self.hidden_size))
            .map_err(Into::into)
    }
}

struct VisionMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl VisionMlp {
    fn new(cfg: &Qwen2VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gated = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gated * up)?).map_err(Into::into)
    }
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1).map_err(Into::into)
}

fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(D::Minus2)?;
    let sin = sin.unsqueeze(D::Minus2)?;
    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(cfg: &Qwen2VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: linear(cfg.hidden_size, cfg.hidden_size * 3, vb.pp("qkv"))?,
            proj: linear(cfg.hidden_size, cfg.hidden_size, vb.pp("proj"))?,
            num_heads: cfg.num_heads,
            head_dim: cfg.hidden_size / cfg.num_heads,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        let hidden_states = self.qkv.forward(xs)?;
        let qkv = hidden_states
            .reshape((seq_len, 3, self.num_heads, self.head_dim))?
            .permute((1, 0, 2, 3))?;
        let mut q = qkv.i(0)?.squeeze(0)?;
        let mut k = qkv.i(1)?.squeeze(0)?;
        let mut v = qkv.i(2)?.squeeze(0)?;

        let cos = cos.to_dtype(DType::F32)?;
        let sin = sin.to_dtype(DType::F32)?;
        q = q.to_dtype(DType::F32)?;
        k = k.to_dtype(DType::F32)?;
        v = v.to_dtype(DType::F32)?;
        (q, k) = apply_rotary_pos_emb_vision(&q, &k, &cos, &sin)?;

        let mut outputs = Vec::new();
        for window in cu_seqlens.windows(2) {
            let start = window[0];
            let end = window[1];
            if end <= start {
                continue;
            }
            let len = end - start;
            let q_chunk = q.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let k_chunk = k.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let v_chunk = v.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let chunk_out = {
                let q = q_chunk.unsqueeze(0)?;
                let k = k_chunk.unsqueeze(0)?;
                let v = v_chunk.unsqueeze(0)?;
                let attn_weights =
                    (q.matmul(&k.transpose(2, 3)?)? / (self.head_dim as f64).sqrt())?;
                let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                attn_weights.matmul(&v)?
            };
            outputs.push(
                chunk_out
                    .squeeze(0)?
                    .transpose(0, 1)?
                    .reshape((len, self.num_heads * self.head_dim))?
                    .to_dtype(xs.dtype())?,
            );
        }
        let attn_output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 0)?;
        self.proj.forward(&attn_output).map_err(Into::into)
    }
}

struct VisionBlock {
    norm1: RmsNorm,
    norm2: RmsNorm,
    attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(cfg: &Qwen2VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: rms_norm(cfg.hidden_size, 1e-6, vb.pp("norm1"))?,
            norm2: rms_norm(cfg.hidden_size, 1e-6, vb.pp("norm2"))?,
            attn: VisionAttention::new(cfg, vb.pp("attn"))?,
            mlp: VisionMlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let attn_out = self
            .attn
            .forward(&self.norm1.forward(xs)?, cu_seqlens, cos, sin)?;
        let xs = xs.add(&attn_out)?;
        let mlp_out = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        xs.add(&mlp_out).map_err(Into::into)
    }
}

struct PatchMerger {
    norm: RmsNorm,
    spatial_merge_unit: usize,
    merged_hidden_size: usize,
    fc1: Linear,
    fc2: Linear,
}

impl PatchMerger {
    fn new(cfg: &Qwen2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let merged_hidden_size = cfg.hidden_size * cfg.spatial_merge_size.pow(2);
        Ok(Self {
            norm: rms_norm(cfg.hidden_size, 1e-6, vb.pp("ln_q"))?,
            spatial_merge_unit: cfg.spatial_merge_size.pow(2),
            merged_hidden_size,
            fc1: linear(merged_hidden_size, merged_hidden_size, vb.pp("mlp.0"))?,
            fc2: linear(merged_hidden_size, cfg.out_hidden_size, vb.pp("mlp.2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        if seq_len % self.spatial_merge_unit != 0 {
            bail!(
                "sequence length {seq_len} is not divisible by spatial merge unit {}",
                self.spatial_merge_unit
            );
        }
        let grouped = seq_len / self.spatial_merge_unit;
        let xs = self
            .norm
            .forward(xs)?
            .reshape((grouped, self.merged_hidden_size))?;
        let xs = self.fc1.forward(&xs)?.gelu()?;
        self.fc2.forward(&xs).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    const THETA: f32 = 10000.;

    fn new(dim: usize, device: &Device) -> Result<Self> {
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / Self::THETA.powf(i as f32 / dim as f32))
            .collect::<Vec<_>>();
        let inv_freq_len = inv_freq.len();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?,
        })
    }

    fn make_embeds(&self, seqlen: usize) -> Result<Tensor> {
        let seq =
            Tensor::arange(0f32, seqlen as f32, self.inv_freq.device())?.unsqueeze(D::Minus1)?;
        seq.broadcast_matmul(&self.inv_freq).map_err(Into::into)
    }
}

pub(crate) struct Qwen2VisionModel {
    patch_embed: PatchEmbed,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    rotary_pos_emb: VisionRotaryEmbedding,
    spatial_merge_size: usize,
    spatial_merge_unit: usize,
    patch_size: usize,
    window_size: usize,
    fullatt_block_indexes: [usize; 4],
    hidden_size: usize,
    dtype: DType,
}

impl Qwen2VisionModel {
    pub(crate) fn new(cfg: &Qwen2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = PatchEmbed::new(cfg, vb.pp("patch_embed"))?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(VisionBlock::new(cfg, vb.pp(format!("blocks.{i}")))?);
        }
        let merger = PatchMerger::new(cfg, vb.pp("merger"))?;
        let head_dim = cfg.hidden_size / cfg.num_heads;
        Ok(Self {
            patch_embed,
            blocks,
            merger,
            rotary_pos_emb: VisionRotaryEmbedding::new(head_dim / 2, vb.device())?,
            spatial_merge_size: cfg.spatial_merge_size,
            spatial_merge_unit: cfg.spatial_merge_size.pow(2),
            patch_size: cfg.patch_size,
            window_size: cfg.window_size,
            fullatt_block_indexes: cfg.fullatt_block_indexes,
            hidden_size: cfg.hidden_size,
            dtype: vb.dtype(),
        })
    }

    fn rot_pos_emb(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let device = self.rotary_pos_emb.inv_freq.device();
        let grid = grid_thw.to_vec2::<u32>()?;
        let max_grid_size = grid.iter().flat_map(|g| [g[1], g[2]]).max().unwrap_or(0) as usize;
        let freq_table = self.rotary_pos_emb.make_embeds(max_grid_size)?;

        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for g in &grid {
            let t = g[0] as usize;
            let h = g[1] as usize;
            let w = g[2] as usize;
            let merged_h = h / self.spatial_merge_size;
            let merged_w = w / self.spatial_merge_size;
            let mut base_rows = Vec::with_capacity(h * w);
            let mut base_cols = Vec::with_capacity(h * w);
            for block_h in 0..merged_h {
                for block_w in 0..merged_w {
                    for inner_h in 0..self.spatial_merge_size {
                        for inner_w in 0..self.spatial_merge_size {
                            base_rows.push((block_h * self.spatial_merge_size + inner_h) as i64);
                            base_cols.push((block_w * self.spatial_merge_size + inner_w) as i64);
                        }
                    }
                }
            }
            for _ in 0..t {
                rows.extend(base_rows.iter().copied());
                cols.extend(base_cols.iter().copied());
            }
        }

        let row_len = rows.len();
        let col_len = cols.len();
        let rows = Tensor::from_vec(rows, (row_len,), device)?;
        let cols = Tensor::from_vec(cols, (col_len,), device)?;
        let row_embeds = freq_table.index_select(&rows, 0)?;
        let col_embeds = freq_table.index_select(&cols, 0)?;
        Tensor::stack(&[row_embeds, col_embeds], D::Minus2)?
            .reshape((rows.dim(0)?, freq_table.dim(D::Minus1)? * 2))
            .map_err(Into::into)
    }

    fn get_window_index(&self, grid_thw: &Tensor) -> Result<(Vec<u32>, Vec<usize>)> {
        let grid = grid_thw.to_vec2::<u32>()?;
        let mut window_index = Vec::new();
        let mut cu_window_seqlens = vec![0usize];
        let mut window_index_id = 0u32;
        let vit_merger_window_size = self.window_size / self.spatial_merge_size / self.patch_size;

        for g in &grid {
            let grid_t = g[0] as usize;
            let grid_h = g[1] as usize;
            let grid_w = g[2] as usize;
            let llm_grid_h = grid_h / self.spatial_merge_size;
            let llm_grid_w = grid_w / self.spatial_merge_size;
            let pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size;
            let pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size;
            let num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
            let num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

            for t in 0..grid_t {
                for win_h in 0..num_windows_h {
                    for win_w in 0..num_windows_w {
                        let mut groups_in_window = 0usize;
                        for inner_h in 0..vit_merger_window_size {
                            for inner_w in 0..vit_merger_window_size {
                                let h_idx = win_h * vit_merger_window_size + inner_h;
                                let w_idx = win_w * vit_merger_window_size + inner_w;
                                if h_idx < llm_grid_h && w_idx < llm_grid_w {
                                    let idx =
                                        t * llm_grid_h * llm_grid_w + h_idx * llm_grid_w + w_idx;
                                    window_index.push(window_index_id + idx as u32);
                                    groups_in_window += 1;
                                }
                            }
                        }
                        let next = cu_window_seqlens.last().copied().unwrap()
                            + groups_in_window * self.spatial_merge_unit;
                        cu_window_seqlens.push(next);
                    }
                }
            }
            window_index_id += (grid_t * llm_grid_h * llm_grid_w) as u32;
        }

        cu_window_seqlens.dedup();
        Ok((window_index, cu_window_seqlens))
    }

    fn build_full_cu_seqlens(&self, grid_thw: &Tensor) -> Result<Vec<usize>> {
        let grid = grid_thw.to_vec2::<u32>()?;
        let mut cu = Vec::new();
        cu.push(0usize);
        let mut acc = 0usize;
        for g in &grid {
            let t = g[0] as usize;
            let seq_len = (g[1] * g[2]) as usize;
            for _ in 0..t {
                acc += seq_len;
                cu.push(acc);
            }
        }
        Ok(cu)
    }

    pub(crate) fn forward(&self, xs: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.patch_embed.forward(&xs.to_dtype(self.dtype)?)?;
        let mut rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let (window_index, cu_window_seqlens) = self.get_window_index(grid_thw)?;
        let cu_seqlens = self.build_full_cu_seqlens(grid_thw)?;

        let seq_len = hidden_states.dim(0)?;
        let group_count = seq_len / self.spatial_merge_unit;
        let index_tensor = Tensor::from_vec(
            window_index.clone(),
            (window_index.len(),),
            hidden_states.device(),
        )?;

        hidden_states = hidden_states
            .reshape((group_count, self.spatial_merge_unit, self.hidden_size))?
            .index_select(&index_tensor, 0)?
            .reshape((seq_len, self.hidden_size))?;
        rotary_pos_emb = rotary_pos_emb
            .reshape((group_count, self.spatial_merge_unit, ()))?
            .index_select(&index_tensor, 0)?
            .reshape((seq_len, rotary_pos_emb.dim(D::Minus1)?))?;

        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(DType::F32)?;
        let sin = emb.sin()?.to_dtype(DType::F32)?;

        for (layer_num, block) in self.blocks.iter().enumerate() {
            let cu_now = if self.fullatt_block_indexes.contains(&layer_num) {
                &cu_seqlens
            } else {
                &cu_window_seqlens
            };
            hidden_states = block.forward(&hidden_states, cu_now, &cos, &sin)?;
        }

        let mut reverse_indices = vec![0u32; window_index.len()];
        for (position, &idx) in window_index.iter().enumerate() {
            reverse_indices[idx as usize] = position as u32;
        }
        let reverse_indices =
            Tensor::from_vec(reverse_indices, (group_count,), hidden_states.device())?;

        self.merger
            .forward(&hidden_states)?
            .index_select(&reverse_indices, 0)
            .map_err(Into::into)
    }

    pub(crate) fn encode_image_bytes(
        &self,
        image_bytes: &[u8],
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let (pixel_values, grid_thw) = preprocess_condition_image(image_bytes, device, dtype)?;
        self.forward(&pixel_values, &grid_thw)
    }
}

pub(crate) fn preprocess_condition_image(
    image_bytes: &[u8],
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let img = image::load_from_memory(image_bytes)?.to_rgb8();
    let ratio = img.width() as f64 / img.height() as f64;
    let mut width = (CONDITION_IMAGE_AREA as f64 * ratio).sqrt();
    let mut height = width / ratio;
    width = (width / 32.0).round() * 32.0;
    height = (height / 32.0).round() * 32.0;
    let stage1 = image::imageops::resize(
        &img,
        width as u32,
        height as u32,
        image::imageops::FilterType::CatmullRom,
    );

    let (final_h, final_w) = smart_resize(
        stage1.height() as usize,
        stage1.width() as usize,
        PATCH_SIZE * SPATIAL_MERGE_SIZE,
        MIN_PIXELS,
        MAX_PIXELS,
    )?;
    let stage2 = image::imageops::resize(
        &stage1,
        final_w as u32,
        final_h as u32,
        image::imageops::FilterType::CatmullRom,
    );

    let grid_h = final_h / PATCH_SIZE;
    let grid_w = final_w / PATCH_SIZE;
    let patch_dim = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;
    let token_count = grid_h * grid_w;
    let mut flat = Vec::with_capacity(token_count * patch_dim);

    for h_block in 0..(grid_h / SPATIAL_MERGE_SIZE) {
        for w_block in 0..(grid_w / SPATIAL_MERGE_SIZE) {
            for merge_h in 0..SPATIAL_MERGE_SIZE {
                for merge_w in 0..SPATIAL_MERGE_SIZE {
                    for channel in 0..3 {
                        for _ in 0..TEMPORAL_PATCH_SIZE {
                            for patch_y in 0..PATCH_SIZE {
                                for patch_x in 0..PATCH_SIZE {
                                    let y = (h_block * SPATIAL_MERGE_SIZE + merge_h) * PATCH_SIZE
                                        + patch_y;
                                    let x = (w_block * SPATIAL_MERGE_SIZE + merge_w) * PATCH_SIZE
                                        + patch_x;
                                    let pixel = stage2.get_pixel(x as u32, y as u32);
                                    let value = pixel[channel] as f32 / 255.0;
                                    flat.push((value - CLIP_MEAN[channel]) / CLIP_STD[channel]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let pixel_values = Tensor::from_vec(flat, (token_count, patch_dim), device)?.to_dtype(dtype)?;
    let grid_thw = Tensor::new(&[[1u32, grid_h as u32, grid_w as u32]], device)?;
    Ok((pixel_values, grid_thw))
}

fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if max(height, width) as f64 / min(height, width) as f64 > 200.0 {
        bail!(
            "absolute aspect ratio must be smaller than 200, got {}",
            max(height, width) as f64 / min(height, width) as f64
        );
    }
    let mut h_bar = ((height as f64 / factor as f64).round() as usize) * factor;
    let mut w_bar = ((width as f64 / factor as f64).round() as usize) * factor;
    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = factor.max(((height as f64 / beta / factor as f64).floor() as usize) * factor);
        w_bar = factor.max(((width as f64 / beta / factor as f64).floor() as usize) * factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
        w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
    }
    Ok((h_bar, w_bar))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_condition_image_square_produces_28x28_patch_grid() {
        let img = image::RgbImage::from_fn(64, 64, |_x, _y| image::Rgb([255, 0, 0]));
        let mut bytes = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(
                &mut std::io::Cursor::new(&mut bytes),
                image::ImageFormat::Png,
            )
            .unwrap();

        let (pixel_values, grid_thw) =
            preprocess_condition_image(&bytes, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(grid_thw.to_vec2::<u32>().unwrap(), vec![vec![1, 28, 28]]);
        assert_eq!(pixel_values.dims2().unwrap(), (28 * 28, 3 * 2 * 14 * 14));
    }

    #[test]
    fn smart_resize_enforces_factor_alignment() {
        let (h, w) = smart_resize(384, 384, 28, MIN_PIXELS, MAX_PIXELS).unwrap();
        assert_eq!((h, w), (392, 392));
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn smart_resize_downscales_when_over_max_pixels() {
        let (h, w) = smart_resize(4096, 4096, 28, MIN_PIXELS, MAX_PIXELS).unwrap();
        assert!(h < 4096);
        assert!(w < 4096);
        assert!(h * w <= MAX_PIXELS);
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
    }

    #[test]
    fn smart_resize_rejects_extreme_aspect_ratios() {
        let err = smart_resize(1, 500, 28, MIN_PIXELS, MAX_PIXELS).unwrap_err();
        assert!(err
            .to_string()
            .contains("aspect ratio must be smaller than 200"));
    }
}
