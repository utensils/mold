# Qwen-Image Implementation Research Findings

> Aggregated from ComfyUI, HuggingFace diffusers, official Wan2.1, InvokeAI, and A1111.
> Date: 2026-03-30

## Critical Bugs in Our Implementation

### BUG 1: Timestep passed as sigma×1000 instead of sigma (CRITICAL)

**Diffusers pipeline** (`pipeline_qwenimage.py` line 683):
```python
noise_pred = self.transformer(
    hidden_states=latents,
    timestep=timestep / 1000,  # ← DIVIDES BY 1000
    ...
)
```

The `QwenImageTransformer2DModel` receives **sigma ∈ [0, 1]**, NOT sigma × 1000.

**Our code** passes `sigma * 1000` directly — the sinusoidal embedding sees values 1000× too large, producing completely wrong time conditioning.

**ComfyUI** (original Wan model) passes raw timestep from scheduler which is already in [0, 1000] range, but uses a `nn.Parameter` scale_shift_table (not a linear layer) that was trained with these values. The Qwen-Image diffusers port changed the convention.

**Impact**: Every denoising step receives wrong noise-level information. The model can't condition on the correct sigma, so velocity predictions are wrong.

### BUG 2: Joint attention concatenation order reversed

**Diffusers** (`transformer_qwenimage.py` lines 541-543):
```python
joint_query = torch.cat([txt_query, img_query], dim=1)  # TEXT FIRST
joint_key = torch.cat([txt_key, img_key], dim=1)
joint_value = torch.cat([txt_value, img_value], dim=1)
```

**Our code** (`quantized_transformer.rs`):
```rust
let q = Tensor::cat(&[&q_img, &q_txt], 1)?;  // IMAGE FIRST
let k = Tensor::cat(&[&k_img, &k_txt], 1)?;
```

The output split also reverses:
- Diffusers: `txt = joint[:, :seq_txt]`, `img = joint[:, seq_txt:]`
- Ours: `img = attn.narrow(1, 0, img_seq_len)`, `txt = attn.narrow(1, img_seq_len, ...)`

**Impact**: The attention mask positions are wrong — padding masks apply to wrong tokens. RoPE frequencies are misaligned between image and text positions.

### BUG 3: Output layer scale/shift ordering (ALREADY FIXED)

**Diffusers** `AdaLayerNormContinuous`: `scale = chunk[0], shift = chunk[1]`
**Original Wan** `Head`: `shift = e[0], scale = e[1]` (additive `scale_shift_table`, not linear)

These are **different conventions** between the original Wan model and the diffusers Qwen-Image port. The GGUF weights come from the **diffusers format** (`city96/Qwen-Image-gguf` converts from HF diffusers), so we must use the diffusers convention: `scale = chunk[0], shift = chunk[1]`.

**Status**: Already fixed in our code.

---

## Architecture Comparison

### Original Wan (ComfyUI, Wan2.1 repo) vs Qwen-Image (diffusers)

| Feature | Original Wan | Qwen-Image (diffusers) |
|---------|-------------|----------------------|
| Block structure | Self-attn → Cross-attn → FFN | Joint-attn → FFN (dual stream) |
| Text handling | Separate cross-attention | Joint/concatenated attention |
| Modulation source | `nn.Parameter` (scale_shift_table) + temb | `nn.Linear(dim, 6*dim)` on silu(temb) |
| Output layer | `nn.Parameter(1, 2, dim)` + temb | `AdaLayerNormContinuous` (silu → linear) |
| Text encoder | T5-XXL (4096-dim) | Qwen2.5-VL (3584-dim) |
| Timestep range | [0, 1000] raw from scheduler | **sigma / 1000 → [0, 1]** |
| Attention order | Image only (self) + text (cross) | **[text, image] concatenated** |
| fp16 clipping | Only when `dtype == float16` | Only when `dtype == float16` |
| BF16 roundtrip | `.type_as(x)` after each residual | `.type_as(x)` after each residual |

### Key Insight: Qwen-Image ≠ Wan

The `QwenImageTransformer2DModel` is a **completely different architecture** from `WanTransformer3DModel`. They share some conventions (patch embedding, RoPE, flow matching) but differ fundamentally in attention mechanism, modulation, and timestep handling.

---

## Numerical Stability Patterns Across Implementations

### 1. BF16 Roundtrip (All implementations)

Every implementation casts intermediate results back to the model dtype after residual connections:

```python
# Wan2.1 (model.py line 305):
with amp.autocast(dtype=torch.float32):
    x = x + y * e[2]  # float32 for gate multiply

# Diffusers (transformer_qwenimage.py line 711):
hidden_states = hidden_states + img_gate1 * img_attn_output  # stays in model dtype

# ComfyUI (model.py line 303):
x = x + attn(norm1(x).float() * (1 + e[1]) + e[0])
```

### 2. fp16 Clipping (Diffusers only, NOT for BF16)

```python
# transformer_qwenimage.py lines 727-730:
if encoder_hidden_states.dtype == torch.float16:
    encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
if hidden_states.dtype == torch.float16:
    hidden_states = hidden_states.clip(-65504, 65504)
```

**Only applies to fp16, NOT bfloat16 or float32.** BF16 has the same exponent range as F32.

### 3. Float32 for Critical Operations

| Operation | Wan2.1 | Diffusers | ComfyUI |
|-----------|--------|-----------|---------|
| Time embedding | `sinusoidal(...).float()` | `.float()` for sinusoidal | `position.type(torch.float32)` |
| RoPE | `torch.float64` (!!) | `.float()` intermediate | Standard float |
| LayerNorm/RMSNorm | `.float()` → compute → `.type_as(x)` | float32 variance | `.float()` → `.type_as(x)` |
| Modulation (gate×output) | `amp.autocast(float32)` | In model dtype | `cast_to()` |
| Patch embedding | N/A | N/A | `x.float()` → conv → `.to(x.dtype)` |

### 4. NaN Safety

- **Wan2.1**: `fp16_clamp()` checks for inf and clamps (T5 encoder only)
- **Diffusers**: fp16 clip at block boundaries
- **ComfyUI**: No explicit NaN handling
- **A1111**: `AutocastLinear` prevents T5 from returning all zeros in fp16

### 5. Dynamic Thresholding (Wan2.1 scheduler)

```python
# Prevents latent saturation during denoising
abs_sample = sample.abs()
s = torch.quantile(abs_sample, dynamic_thresholding_ratio, dim=1)
s = torch.clamp(s, min=1, max=sample_max_value)
sample = torch.clamp(sample, -s, s) / s
```

---

## Scheduler Details

### Qwen-Image Scheduler (diffusers `FlowMatchEulerDiscreteScheduler`)

```python
# Sigma generation:
sigmas = np.linspace(1.0, 1/num_train_timesteps, num_inference_steps)

# Dynamic time shift:
shifted = exp(mu) / (exp(mu) + (1/sigma - 1))

# Stretch to terminal:
one_minus_z = 1 - sigmas
scale_factor = one_minus_z[-1] / (1 - shift_terminal)
stretched = 1 - (one_minus_z / scale_factor)

# Timesteps for model: sigmas * 1000
# But pipeline divides by 1000 before passing to transformer
```

### Original Wan Scheduler (FlowDPMSolver++ / FlowUniPC)

- Uses different solvers (not just Euler)
- Shift formula: `sigma = shift * sigma / (1 + (shift - 1) * sigma)` (linear, not exponential)
- Default shift: 5.0 for T2V, 3.0 for I2V

---

## Text Encoder Comparison

| Feature | Wan (T5-XXL) | Qwen-Image (Qwen2.5-VL) |
|---------|-------------|------------------------|
| Hidden dim | 4096 | 3584 |
| Layers | 24 | 28 |
| Output used | Final layer (after LayerNorm) | **Penultimate layer** (no final norm) |
| Max tokens | 512 | 1024 |
| Attention | Bidirectional (encoder-only T5) | Causal (decoder LM) |
| Normalization | In model's `text_embedding` MLP | In transformer's `txt_norm` (RMSNorm) |

### Qwen2.5-VL Prompt Template

```
<|im_start|>system
Describe the image by detailing the color, shape, size, texture, quantity, text,
spatial relationships of the objects and background:<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

---

## Action Items for mold

### Must Fix (Critical)

1. **Divide timestep by 1000** before passing to `QwenImageTransformer2DModel`
   - Change: `current_timestep() / 1000.0` in pipeline
   - Or change `current_timestep()` to return just `sigma` (not `sigma * 1000`)

2. **Reverse joint attention concatenation order** to `[text, image]`
   - In `quantized_transformer.rs` and `transformer.rs`: swap cat order for Q, K, V
   - Fix output split: text = `narrow(1, 0, txt_seq_len)`, image = `narrow(1, txt_seq_len, ...)`

### Should Fix (Important)

3. **Remove ±65504 clamp** — only needed for fp16, not BF16/F32. Our quantized path uses F32.

4. **Add BF16 roundtrip inside blocks** (already done) — matches `.type_as()` behavior.

5. **Verify RoPE frequency assignment** — with reversed concatenation order, text RoPE positions may need adjustment.

### Nice to Have

6. **CFG norm rescaling** — diffusers preserves conditional norm after CFG combination
7. **Dynamic thresholding** in scheduler — prevents latent saturation
8. **Float64 for RoPE** — Wan2.1 uses float64 for RoPE computation (we use F32)

---

## Source File Reference

| Implementation | Key Files |
|---------------|-----------|
| **Diffusers** | `tmp/diffusers/src/diffusers/models/transformers/transformer_qwenimage.py` |
| | `tmp/diffusers/src/diffusers/models/normalization.py` (AdaLayerNormContinuous) |
| | `tmp/diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py` |
| | `tmp/diffusers/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py` |
| **ComfyUI** | `tmp/ComfyUI/comfy/ldm/wan/model.py` (transformer, blocks, head) |
| | `tmp/ComfyUI/comfy/text_encoders/qwen_image.py` (tokenizer, template) |
| | `tmp/ComfyUI/comfy/ldm/wan/vae.py` (VAE decoder) |
| **Wan2.1** | `tmp/Wan2.1/wan/modules/model.py` (original transformer) |
| | `tmp/Wan2.1/wan/modules/t5.py` (T5 text encoder) |
| | `tmp/Wan2.1/wan/text2video.py` (inference pipeline) |
| | `tmp/Wan2.1/wan/utils/fm_solvers.py` (scheduler) |
| **InvokeAI** | `tmp/InvokeAI/invokeai/app/invocations/z_image_denoise.py` (Z-Image reference) |
| **A1111** | `tmp/stable-diffusion-webui/modules/models/sd3/other_impls.py` (AutocastLinear) |
