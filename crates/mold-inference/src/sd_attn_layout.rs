//! The cross-attention (`attn2`) layout of a Stable Diffusion UNet.
//!
//! Shared by every adapter that injects a second key/value stream through the
//! candle fork's [`CrossAttentionHook`] — PuLID for SDXL, and IP-Adapter for
//! SD1.5 and SDXL. The layout is a fact about the UNet, not about any one
//! adapter, which is why it lives beside neither.
//!
//! Two orders have to be reconciled and the whole module exists to do it once:
//!
//! * **diffusers** registers `down_blocks`, then `up_blocks`, then
//!   `mid_block`, and within each transformer block registers `attn1` before
//!   `attn2`. A checkpoint keyed by processor position therefore counts BOTH
//!   attentions, in that order.
//! * **candle's forward** runs `down_blocks`, then `mid_block`, then
//!   `up_blocks`, and the hook cursor counts only `attn2`.
//!
//! Everything is derived from a [`UNet2DConditionModelConfig`] rather than
//! transcribed, so SD1.5's 16 modules and SDXL's 70 come out of one piece of
//! arithmetic and a config change cannot leave a stale table behind. The two
//! geometries are pinned against upstream's own enumeration by fixtures in
//! `testdata/pulid_sdxl/`.

use std::collections::BTreeMap;

use candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModelConfig;

/// One hooked cross-attention module: where it sits in the forward pass, and
/// which checkpoint index carries its weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnLayerSite {
    /// Position among `attn2` modules in the order candle's UNet forward
    /// visits them — `down_blocks -> mid_block -> up_blocks` — which is
    /// exactly the `index` the [`CrossAttentionHook`] receives.
    pub hook_index: usize,
    /// Position in diffusers' `unet.attn_processors`, which interleaves
    /// `attn1` and `attn2` and walks `down_blocks -> up_blocks -> mid_block`.
    /// This is the `<i>` in `id_adapter_attn_layers.<i>`.
    pub processor_index: usize,
    /// The module's own channel width; `id_to_k` / `id_to_v` are
    /// `[hidden_size, ID_TOKEN_DIM]`.
    pub hidden_size: usize,
    /// Attention heads this module splits `hidden_size` across.
    pub heads: usize,
}

impl AttnLayerSite {
    /// Width of one head. `IDAttnProcessor2_0` derives it the same way
    /// (`attention_processor.py:338`, `inner_dim // attn.heads`).
    pub fn dim_head(&self) -> usize {
        self.hidden_size / self.heads
    }

    /// Position among `attn2` modules in DIFFUSERS registration order — the
    /// `<i>` in IP-Adapter's `ip_adapter.<i>.to_k_ip.weight`.
    ///
    /// PuLID and IP-Adapter key their per-layer weights off two DIFFERENT
    /// index spaces, and the difference is exactly a factor of two. PuLID's
    /// `id_adapter_attn_layers.<i>` counts `unet.attn_processors` positions,
    /// which interleave `attn1` and `attn2`; IP-Adapter's checkpoint only ever
    /// has cross-attention entries, so its `<i>` counts the filtered list.
    ///
    /// Every transformer block registers `attn1` at an even position and
    /// `attn2` at the odd one after it, so [`Self::processor_index`] is always
    /// `2 * ip_index + 1`. Derived rather than stored so the two can never
    /// disagree; pinned against upstream's own `attn2_ordinal` column in
    /// `testdata/pulid_sdxl/attn_layer_map*.json`.
    pub fn ip_index(&self) -> usize {
        (self.processor_index - 1) / 2
    }
}

/// A contiguous run of `attn2` modules that share a width and a head count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Region {
    Down(usize),
    Mid,
    Up(usize),
}

#[derive(Debug, Clone, Copy)]
struct RegionShape {
    region: Region,
    /// `attn2` modules this region contributes.
    count: usize,
    hidden_size: usize,
    heads: usize,
}

/// The `attn2` layout of one UNet, in hook order, with each module's
/// checkpoint index.
///
/// Derived from the UNet config rather than transcribed, so the SD1.5 geometry
/// (16 modules) and the SDXL geometry (70) come out of the same arithmetic and
/// a config change cannot leave a hard-coded table behind. The two orders are
/// built separately and joined on `(region, local index)`:
///
/// * diffusers registers `down_blocks`, then `up_blocks`, then `mid_block`,
///   because that is `UNet2DConditionModel.__init__`'s attribute order and
///   `attn_processors` walks `named_children` — and within each transformer
///   block it registers `attn1` before `attn2`.
/// * candle's `forward` runs `down_blocks`, then `mid_block`, then `up_blocks`
///   (`unet_2d.rs`'s "3. down / 4. mid / 5. up"), which is the order the hook
///   cursor counts in.
pub fn plan_attn_layers(config: &UNet2DConditionModelConfig) -> Vec<AttnLayerSite> {
    let n_blocks = config.blocks.len();
    let mut downs = Vec::new();
    let mut ups = Vec::new();
    for index in 0..n_blocks {
        let block = config.blocks[index];
        if let Some(transformer_layers) = block.use_cross_attn {
            downs.push(RegionShape {
                region: Region::Down(index),
                count: config.layers_per_block * transformer_layers,
                hidden_size: block.out_channels,
                heads: block.attention_head_dim,
            });
        }
        // `up_blocks[i]` is built from `blocks[n - 1 - i]` and carries one more
        // resnet layer than its down-block mirror (`unet_2d.rs`'s
        // `num_layers: config.layers_per_block + 1`).
        let mirrored = config.blocks[n_blocks - 1 - index];
        if let Some(transformer_layers) = mirrored.use_cross_attn {
            ups.push(RegionShape {
                region: Region::Up(index),
                count: (config.layers_per_block + 1) * transformer_layers,
                hidden_size: mirrored.out_channels,
                heads: mirrored.attention_head_dim,
            });
        }
    }
    // The mid block is always cross-attentional and always takes the last
    // block's width, head count, and transformer depth (`unet_2d.rs`'s
    // `mid_transformer_layers_per_block`, mirroring diffusers' own
    // `unet_2d_condition.py:462`).
    let mid = config.blocks.last().map(|block| RegionShape {
        region: Region::Mid,
        count: block.use_cross_attn.unwrap_or(1),
        hidden_size: block.out_channels,
        heads: block.attention_head_dim,
    });

    // Diffusers registration order assigns the checkpoint indices.
    let mut processor_of: BTreeMap<(Region, usize), usize> = BTreeMap::new();
    let mut next_processor = 0usize;
    let diffusers_order = downs
        .iter()
        .chain(ups.iter())
        .chain(mid.iter())
        .copied()
        .collect::<Vec<_>>();
    for shape in &diffusers_order {
        for local in 0..shape.count {
            // Every transformer block registers `attn1` then `attn2`.
            next_processor += 1;
            processor_of.insert((shape.region, local), next_processor);
            next_processor += 1;
        }
    }

    // Candle's forward order assigns the hook indices.
    let candle_order = downs
        .iter()
        .chain(mid.iter())
        .chain(ups.iter())
        .copied()
        .collect::<Vec<_>>();
    let mut sites = Vec::with_capacity(next_processor / 2);
    for shape in &candle_order {
        for local in 0..shape.count {
            let processor_index = processor_of[&(shape.region, local)];
            sites.push(AttnLayerSite {
                hook_index: sites.len(),
                processor_index,
                hidden_size: shape.hidden_size,
                heads: shape.heads,
            });
        }
    }
    sites
}
