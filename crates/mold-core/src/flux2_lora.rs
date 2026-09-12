//! Flux.2 LoRA / transformer geometry agreement.
//!
//! A Flux.2 adapter is trained against ONE tier's transformer width, and the
//! three published widths are distinct: 3072 for FLUX.2 [klein] 4B, 4096 for
//! FLUX.2 [klein] 9B, and 6144 for FLUX.2 [dev]. Nothing in an adapter's file
//! name says which, so `flux2-klein-delight-lora.safetensors` — a 4096-wide
//! adapter trained for the 9B — reads as a plain Klein adapter and was handed
//! to the 3072-wide 4B tier, where it loaded cleanly (`32 layers, rank 32`),
//! built 32 patches on 16 tensors with nothing skipped, and then died inside
//! the merge with `shape mismatch in add, lhs: [3072, 3072], rhs: [4096,
//! 4096]` — after the whole GGUF checkpoint had been read and dequantised.
//!
//! This module is the ONE place that answers "may this adapter meet this
//! transformer", and it answers from the adapter's own safetensors HEADER, so
//! the question is settled before a transformer byte is read. The engine asks
//! it at adapter load and the server asks it at admission, so the CLI, the
//! HTTP API and the durable queue refuse identically and the durable path
//! never reaches the merge.
//!
//! # Why a foreign tier's hidden size is an unambiguous signal
//!
//! A LoRA tensor pair is `A: [rank, in_features]` and `B: [out_features,
//! rank]`, so every non-rank dimension is a layer width of the transformer it
//! was trained on. Those widths are not all `hidden_size` — a fused-QKV
//! adapter carries `3h`, an MLP adapter carries `h * mlp_ratio`, a single
//! block's `linear1` carries `3h + 2 * mlp` — but for `mlp_ratio = 3.0`,
//! which all three tiers share, NO legitimate width on one tier equals
//! another tier's `hidden_size`:
//!
//! | tier | hidden | legitimate widths |
//! |---|---|---|
//! | [klein] 4B | 3072 | 128, 3072, 7680, 9216, 12288, 27648 |
//! | [klein] 9B | 4096 | 128, 4096, 12288, 16384, 36864 |
//! | [dev] | 6144 | 128, 6144, 15360, 18432, 24576, 55296 |
//!
//! `the_tier_widths_never_collide_with_a_foreign_hidden_size` pins that
//! against the same arithmetic. So the test is narrow and cannot fire on an
//! adapter that would have merged: a width equal to a DIFFERENT tier's hidden
//! size can only have come from that tier.
//!
//! The refusal is deliberately not "every width must be explicable", because
//! that would have to enumerate every module a trainer may target
//! (modulation, embedders, the final layer) and would refuse a legitimate
//! adapter the day one targets a module the list forgot. Anything this gate
//! lets through that still cannot merge is refused by name at the merge
//! itself, which no longer skips a mismatched tensor silently.

use std::collections::BTreeMap;
use std::path::Path;

/// The Flux.2 transformer widths mold knows, paired with the tier each names.
///
/// `Flux2Config::{klein, klein_9b, dev}().hidden_size` in `mold-inference`'s
/// `flux2::transformer` are these three numbers; `mold-core` cannot depend on
/// that crate, so `the_core_tier_table_is_the_engines_own` over there pins
/// the two together rather than letting them drift.
pub const FLUX2_TIER_WIDTHS: [(usize, &str); 3] = [
    (3072, "FLUX.2 [klein] 4B (flux2-klein:*)"),
    (4096, "FLUX.2 [klein] 9B (flux2-klein-9b:*)"),
    (6144, "FLUX.2 [dev] (flux2-dev:*)"),
];

/// The tier a Flux.2 transformer hidden size names, or `None` for a width
/// that is not one of the three published ones.
pub fn flux2_tier_for_hidden_size(hidden_size: usize) -> Option<&'static str> {
    FLUX2_TIER_WIDTHS
        .iter()
        .find(|(width, _)| *width == hidden_size)
        .map(|(_, tier)| *tier)
}

/// Every non-rank dimension declared by a LoRA safetensors header, with the
/// number of tensors carrying it.
///
/// A LoRA pair is `A: [rank, in]` / `B: [out, rank]`, so on a rank-2 tensor
/// the LARGER dimension is the layer width and the smaller is the LoRA rank.
/// A tensor of any other rank (an alpha scalar, a bias) carries no width and
/// is ignored, and so is a square one — there the width and the rank cannot
/// be told apart, and guessing is how a rank-4096 adapter would be read as a
/// tier.
pub fn lora_declared_widths(shapes: &BTreeMap<String, Vec<usize>>) -> BTreeMap<usize, usize> {
    let mut widths: BTreeMap<usize, usize> = BTreeMap::new();
    for dims in shapes.values() {
        if dims.len() != 2 {
            continue;
        }
        let width = dims[0].max(dims[1]);
        let rank = dims[0].min(dims[1]);
        if width == rank {
            continue;
        }
        *widths.entry(width).or_default() += 1;
    }
    widths
}

/// Refuse an adapter whose declared widths place it on a DIFFERENT Flux.2
/// tier than the transformer it is about to meet.
///
/// `adapter_label` is what the user typed, or the file's name — it is quoted
/// back so a multi-adapter stack says which one is wrong. Returns `None` when
/// the adapter carries no foreign tier width, which includes every adapter
/// trained for `transformer_hidden_size` and every adapter whose widths this
/// gate cannot place.
pub fn flux2_lora_tier_mismatch(
    adapter_label: &str,
    shapes: &BTreeMap<String, Vec<usize>>,
    transformer_hidden_size: usize,
) -> Option<String> {
    let declared = lora_declared_widths(shapes);
    let foreign: Vec<String> = declared
        .iter()
        .filter(|(width, _)| **width != transformer_hidden_size)
        .filter_map(|(width, count)| {
            flux2_tier_for_hidden_size(*width)
                .map(|tier| format!("{count} tensor(s) {width} wide, which is {tier}"))
        })
        .collect();
    if foreign.is_empty() {
        return None;
    }

    let matching = declared.get(&transformer_hidden_size).copied().unwrap_or(0);
    let this_tier =
        flux2_tier_for_hidden_size(transformer_hidden_size).unwrap_or("this FLUX.2 transformer");

    let mut message = format!(
        "LoRA adapter '{adapter_label}' was trained for a different FLUX.2 tier: it declares {}, \
         but this transformer is {transformer_hidden_size} wide — {this_tier}.",
        foreign.join(" and ")
    );
    if matching > 0 {
        // A partial match is still a refusal. Merging the agreeing tensors
        // and dropping the rest is a silently half-applied adapter, which
        // renders a plausible image nobody asked for.
        message.push_str(&format!(
            " {matching} of its tensors do match this width; a partly applied adapter is refused \
             rather than merged, because the result is an image no adapter produced."
        ));
    }
    message.push_str(
        " Use an adapter trained for this tier, or render with the tier this adapter fits.",
    );
    Some(message)
}

/// The same decision, reading the adapter's safetensors header off disk.
///
/// Header-only — the 8-byte length prefix plus the JSON, never a tensor byte
/// — so it is as cheap on a coordinator admitting a request as it is in the
/// engine. A file that cannot be opened or parsed yields `None`: a missing or
/// corrupt adapter is already refused, by name, on the paths that own that
/// question, and this gate must not turn a not-yet-downloaded adapter into a
/// tier complaint.
pub fn flux2_lora_tier_mismatch_for_file(
    path: &Path,
    adapter_label: &str,
    transformer_hidden_size: usize,
) -> Option<String> {
    let header = crate::safetensors_probe::read_safetensors_header(path).ok()?;
    flux2_lora_tier_mismatch(
        adapter_label,
        &header.tensor_shapes,
        transformer_hidden_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The header shapes of a rank-`rank` adapter targeting `sites` layers of
    /// width `width` — the `A: [rank, width]` / `B: [width, rank]` pair PEFT
    /// writes.
    fn adapter(width: usize, rank: usize, sites: usize) -> BTreeMap<String, Vec<usize>> {
        let mut shapes = BTreeMap::new();
        for i in 0..sites {
            shapes.insert(
                format!("transformer.transformer_blocks.{i}.attn.to_q.lora_A.weight"),
                vec![rank, width],
            );
            shapes.insert(
                format!("transformer.transformer_blocks.{i}.attn.to_q.lora_B.weight"),
                vec![width, rank],
            );
        }
        shapes
    }

    /// Write a real synthetic `.safetensors` whose header declares `shapes`,
    /// so the on-disk door is exercised rather than assumed. Every tensor
    /// aliases the same zero blob — the probe never reads payload.
    fn write_header_fixture(path: &Path, shapes: &BTreeMap<String, Vec<usize>>) {
        use std::io::Write;
        let mut header = serde_json::Map::new();
        for (name, dims) in shapes {
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": dims,
                    "data_offsets": [0, 4],
                }),
            );
        }
        let json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut file = std::fs::File::create(path).expect("create fixture");
        file.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&json).unwrap();
        file.write_all(&[0u8; 4]).unwrap();
    }

    fn fixture_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "mold-flux2-lora-{name}-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    /// The defect: `flux2-klein-delight-lora.safetensors` is 4096 wide and was
    /// handed to the 3072-wide 4B tier. Its real inventory is 64 tensors —
    /// 32 sites of `[32, 4096]` / `[4096, 32]`, rank 32.
    #[test]
    fn a_9b_adapter_is_refused_on_the_4b_tier() {
        let refusal = flux2_lora_tier_mismatch(
            "flux2-klein-delight-lora.safetensors",
            &adapter(4096, 32, 32),
            3072,
        )
        .expect("a 4096-wide adapter must be refused on a 3072-wide transformer");

        assert!(
            refusal.contains("flux2-klein-delight-lora.safetensors"),
            "must name the adapter: {refusal}"
        );
        assert!(refusal.contains("4096"), "must name its width: {refusal}");
        assert!(
            refusal.contains("3072"),
            "must name the tier's width: {refusal}"
        );
        assert!(
            refusal.contains("[klein] 9B"),
            "must name the tier it fits: {refusal}"
        );
        assert!(
            refusal.contains("[klein] 4B"),
            "must name the tier it was handed to: {refusal}"
        );
    }

    /// Every wrong pairing of the three published widths refuses, and every
    /// right one is silent — driven off real header fixtures so the on-disk
    /// door is covered at 3072, 4096 and 6144.
    #[test]
    fn every_tier_accepts_only_its_own_width() {
        for (adapter_width, _) in FLUX2_TIER_WIDTHS {
            let path = fixture_path(&format!("w{adapter_width}"));
            write_header_fixture(&path, &adapter(adapter_width, 16, 4));

            for (transformer_width, _) in FLUX2_TIER_WIDTHS {
                let verdict = flux2_lora_tier_mismatch_for_file(
                    &path,
                    "fixture.safetensors",
                    transformer_width,
                );
                if adapter_width == transformer_width {
                    assert!(
                        verdict.is_none(),
                        "{adapter_width} on {transformer_width} must load: {verdict:?}"
                    );
                } else {
                    let refusal = verdict.unwrap_or_else(|| {
                        panic!("{adapter_width} on {transformer_width} must refuse")
                    });
                    assert!(refusal.contains(&adapter_width.to_string()), "{refusal}");
                    assert!(
                        refusal.contains(&transformer_width.to_string()),
                        "{refusal}"
                    );
                }
            }
            let _ = std::fs::remove_file(&path);
        }
    }

    /// A partly matching adapter is a refusal, not a partial merge — and the
    /// message says so, because "it loaded" and "it applied" are different
    /// claims.
    #[test]
    fn a_partly_matching_adapter_is_refused_not_half_merged() {
        let mut shapes = adapter(3072, 32, 6);
        shapes.extend(adapter(4096, 32, 2).into_iter().map(|(key, dims)| {
            (
                key.replace("transformer_blocks", "single_transformer_blocks"),
                dims,
            )
        }));

        let refusal = flux2_lora_tier_mismatch("mixed.safetensors", &shapes, 3072)
            .expect("any foreign-tier tensor must refuse the whole adapter");
        assert!(
            refusal.contains("do match this width"),
            "must say some tensors matched: {refusal}"
        );
        assert!(
            refusal.contains("partly applied"),
            "must say why a partial merge is not the answer: {refusal}"
        );
    }

    /// Fused and MLP widths are legitimate on their own tier and must not be
    /// read as a foreign tier. A 4B fused-QKV adapter carries 9216 and a 4B
    /// `linear1` adapter carries 27648; neither is another tier's hidden size.
    #[test]
    fn fused_and_mlp_widths_on_the_right_tier_are_accepted() {
        let mut shapes = BTreeMap::new();
        // 4B: hidden 3072, mlp_ratio 3.0 -> mlp 9216.
        for (name, width) in [
            ("double_blocks.0.img_attn.qkv", 9216),  // 3h
            ("double_blocks.0.img_mlp.0", 9216),     // mlp
            ("double_blocks.0.img_mlp.2", 9216),     // mlp (in)
            ("single_blocks.0.linear1", 27648),      // 3h + 2*mlp
            ("single_blocks.0.linear2", 12288),      // h + mlp
            ("txt_in", 7680),                        // context_in_dim
            ("double_blocks.0.img_attn.proj", 3072), // h
        ] {
            shapes.insert(format!("{name}.lora_A.weight"), vec![16, width]);
            shapes.insert(format!("{name}.lora_B.weight"), vec![width, 16]);
        }
        assert_eq!(
            flux2_lora_tier_mismatch("fused.safetensors", &shapes, 3072),
            None,
            "no legitimate 4B width may read as another tier"
        );
    }

    /// The premise the whole gate rests on, checked against the arithmetic
    /// rather than asserted: for `mlp_ratio = 3.0`, no width a tier can
    /// legitimately carry equals a DIFFERENT tier's hidden size.
    #[test]
    fn the_tier_widths_never_collide_with_a_foreign_hidden_size() {
        // (hidden, context_in_dim) from `Flux2Config::{klein, klein_9b, dev}`.
        for (hidden, context_in_dim) in [(3072usize, 7680usize), (4096, 12288), (6144, 15360)] {
            let mlp = hidden * 3; // mlp_ratio 3.0 on all three tiers
            let legitimate = [
                128, // in_channels
                hidden,
                3 * hidden,
                mlp,
                hidden + mlp,
                3 * hidden + 2 * mlp,
                context_in_dim,
            ];
            for width in legitimate {
                if width == hidden {
                    continue;
                }
                assert_eq!(
                    flux2_tier_for_hidden_size(width),
                    None,
                    "width {width} is legitimate at hidden {hidden} but reads as a tier"
                );
            }
        }
    }

    /// Ranks are never mistaken for widths, and a tensor that declares no
    /// width at all is ignored rather than guessed at.
    #[test]
    fn ranks_alphas_and_square_tensors_declare_no_width() {
        let widths = lora_declared_widths(&adapter(4096, 32, 2));
        assert_eq!(
            widths,
            BTreeMap::from([(4096, 4)]),
            "rank 32 is not a width"
        );

        let odd = BTreeMap::from([
            ("alpha".to_string(), vec![]),
            ("bias".to_string(), vec![4096]),
            ("square".to_string(), vec![4096, 4096]),
            ("rank3".to_string(), vec![2, 32, 4096]),
        ]);
        assert!(
            lora_declared_widths(&odd).is_empty(),
            "only a rank-2 non-square tensor declares a width"
        );
        assert_eq!(
            flux2_lora_tier_mismatch("odd.safetensors", &odd, 3072),
            None
        );
    }

    /// An unreadable adapter is not a tier complaint — the paths that own
    /// "is this file there" answer that, by name.
    #[test]
    fn an_unreadable_adapter_yields_no_verdict() {
        let missing = fixture_path("absent");
        let _ = std::fs::remove_file(&missing);
        assert_eq!(
            flux2_lora_tier_mismatch_for_file(&missing, "absent.safetensors", 3072),
            None
        );
    }
}
