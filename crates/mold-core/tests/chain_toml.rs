//! Round-trip `ChainScript ↔ TOML` ↔ normalise integration tests.
//! Lives in `tests/` (not in-crate) so it exercises the public API only.

use mold_core::chain::{ChainRequest, ChainScript, ChainScriptChain, ChainStage, TransitionMode};
use mold_core::chain_toml::{read_script, write_script};
use mold_core::types::OutputFormat;

fn script_under_test() -> ChainScript {
    ChainScript {
        schema: "mold.chain.v1".into(),
        chain: ChainScriptChain {
            model: "ltx-2-19b-distilled:fp8".into(),
            width: 1216,
            height: 704,
            fps: 24,
            seed: Some(42),
            steps: 8,
            guidance: 3.0,
            strength: 1.0,
            motion_tail_frames: 25,
            output_format: OutputFormat::Mp4,
        },
        stages: vec![
            ChainStage {
                prompt: "a cat walks into the autumn forest".into(),
                frames: 97,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Smooth,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            },
            ChainStage {
                prompt: "the forest opens to a clearing".into(),
                frames: 49,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Smooth,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            },
            ChainStage {
                prompt: "a spaceship lands".into(),
                frames: 97,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Cut,
                fade_frames: None,
                model: None,
                loras: vec![],
                references: vec![],
            },
            ChainStage {
                prompt: "the cat looks up in wonder".into(),
                frames: 97,
                source_image: None,
                negative_prompt: None,
                seed_offset: None,
                transition: TransitionMode::Fade,
                fade_frames: Some(12),
                model: None,
                loras: vec![],
                references: vec![],
            },
        ],
    }
}

#[test]
fn write_then_read_is_identity() {
    let script = script_under_test();
    let toml_out = write_script(&script).unwrap();
    let back = read_script(&toml_out).unwrap();
    assert_eq!(back.schema, script.schema);
    assert_eq!(back.chain.model, script.chain.model);
    assert_eq!(back.chain.seed, script.chain.seed);
    assert_eq!(back.stages.len(), script.stages.len());
    for (a, b) in back.stages.iter().zip(script.stages.iter()) {
        assert_eq!(a.prompt, b.prompt);
        assert_eq!(a.frames, b.frames);
        assert_eq!(a.transition, b.transition);
        assert_eq!(a.fade_frames, b.fade_frames);
    }
}

#[test]
fn normalised_request_survives_round_trip() {
    // Build a ChainRequest → normalise → project to ChainScript → TOML → back → compare.
    let req = ChainRequest {
        model: "ltx-2-19b-distilled:fp8".into(),
        stages: script_under_test().stages,
        motion_tail_frames: 25,
        width: 1216,
        height: 704,
        fps: 24,
        seed: Some(42),
        steps: 8,
        guidance: 3.0,
        strength: 1.0,
        output_format: OutputFormat::Mp4,
        placement: None,
        prompt: None,
        total_frames: None,
        clip_frames: None,
        source_image: None,
    };
    let normalised = req.normalise().unwrap();
    let script = ChainScript::from(&normalised);
    let toml_out = write_script(&script).unwrap();
    let back = read_script(&toml_out).unwrap();
    assert_eq!(back.stages.len(), normalised.stages.len());
}
