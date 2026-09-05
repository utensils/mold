import { describe, expect, it } from "vitest";
import {
  buildChainRequest,
  chainScriptToClips,
  clipsToChainScript,
  newSequenceClip,
  stageInvalidation,
  type SequenceClipForm,
  type SequenceSharedParams,
} from "./sequenceForm";
import type { ChainScript } from "./api/chainTypes";

const PNG_1170_BY_2532 = "iVBORw0KGgoAAAANSUhEUgAABJIAAAnk";

function shared(
  extra: Partial<SequenceSharedParams> = {},
): SequenceSharedParams {
  return {
    model: "ltx-2-19b-distilled:fp8",
    family: "ltx2",
    width: 1216,
    height: 704,
    fps: 24,
    steps: 8,
    guidance: 3,
    strength: 1,
    seed: "",
    ...extra,
  };
}

function clip(extra: Partial<SequenceClipForm> = {}): SequenceClipForm {
  return { ...newSequenceClip(97), prompt: "a kingfisher waits", ...extra };
}

describe("buildChainRequest", () => {
  it("reads the LIVE shared params at submit time", () => {
    // Regression for the web divergence bug: sequence submissions ignored
    // inspector changes because the composer kept its own copies. There is
    // no second copy here — mutate shared params after the clips exist and
    // the request must carry the new values.
    const params = shared();
    const clips = [clip(), clip({ prompt: "it lifts off" })];
    params.width = 704;
    params.height = 1216;
    params.steps = 30;
    params.guidance = 5;

    const req = buildChainRequest(params, clips, {
      motionTailFrames: 17,
      enableAudio: false,
    });
    expect(req.width).toBe(704);
    expect(req.height).toBe(1216);
    expect(req.steps).toBe(30);
    expect(req.guidance).toBe(5);
    expect(req.model).toBe("ltx-2-19b-distilled:fp8");
    expect(req.motion_tail_frames).toBe(17);
    expect(req.stages).toHaveLength(2);
  });

  it("coerces clip 1's transition to smooth and gates fade_frames", () => {
    const clips = [
      clip({ transition: "fade", fadeFrames: 12 }),
      clip({ prompt: "b", transition: "cut", fadeFrames: 12 }),
      clip({ prompt: "c", transition: "fade", fadeFrames: 8 }),
    ];
    const req = buildChainRequest(shared(), clips, {
      motionTailFrames: 17,
      enableAudio: false,
    });
    expect(req.stages[0]?.transition).toBe("smooth");
    expect(req.stages[0]?.fade_frames).toBeUndefined();
    expect(req.stages[1]?.transition).toBe("cut");
    expect(req.stages[1]?.fade_frames).toBeUndefined();
    expect(req.stages[2]?.transition).toBe("fade");
    expect(req.stages[2]?.fade_frames).toBe(8);
  });

  it("sends a numeric seed only when fixed, and audio explicitly either way", () => {
    // `enable_audio` is ALWAYS on the wire. An omitted field means "resolve
    // the recipe's default" at the server door, and that default is on for an
    // audio family — so omitting it when the author turned sound off would
    // hand the audio straight back.
    const random = buildChainRequest(shared({ seed: "" }), [clip(), clip()], {
      motionTailFrames: 17,
      enableAudio: false,
    });
    expect(random.seed).toBeUndefined();
    expect(random.enable_audio).toBe(false);

    const fixed = buildChainRequest(shared({ seed: "42" }), [clip(), clip()], {
      motionTailFrames: 17,
      enableAudio: true,
    });
    expect(fixed.seed).toBe(42);
    expect(fixed.enable_audio).toBe(true);
  });

  it("keeps an author's silence explicit in the exported script", () => {
    // `clipsToChainScript` is the amend body and the TOML export. A null
    // there reads as "resolve the default" on the way back in, which for an
    // audio family means sound — so a false has to be written down.
    const off = clipsToChainScript(shared({}), [clip(), clip()], {
      motionTailFrames: 17,
      enableAudio: false,
    });
    const on = clipsToChainScript(shared({}), [clip(), clip()], {
      motionTailFrames: 17,
      enableAudio: true,
    });
    expect(off.chain.enable_audio).toBe(false);
    expect(on.chain.enable_audio).toBe(true);
  });

  it("carries per-clip negative prompts and the opening source image", () => {
    const clips = [
      clip({
        negativePrompt: "no rain",
      }),
      clip({ prompt: "b" }),
    ];
    const req = buildChainRequest(shared(), clips, {
      motionTailFrames: 17,
      enableAudio: false,
      openingImage: { filename: "open.png", base64: "QUJD" },
    });
    expect(req.stages[0]?.negative_prompt).toBe("no rain");
    expect(req.stages[0]?.source_image).toBe("QUJD");
    expect(req.stages[1]?.source_image).toBeUndefined();
  });

  it("serializes a distinct camera motion LoRA on each clip", () => {
    const req = buildChainRequest(
      shared(),
      [
        clip({ cameraControl: "dolly-in" }),
        clip({
          prompt: "b",
          cameraControl: "/models/camera/custom.safetensors",
        }),
      ],
      { motionTailFrames: 17, enableAudio: false },
    );
    expect(req.stages[0]?.loras).toEqual([
      {
        path: "camera-control:dolly-in",
        scale: 1,
        name: "Dolly in",
      },
    ]);
    expect(req.stages[1]?.loras).toEqual([
      {
        path: "/models/camera/custom.safetensors",
        scale: 1,
        name: "Camera motion",
      },
    ]);
  });
});

describe("script round-trip", () => {
  it("keeps request and script stage projections equivalent", () => {
    const clips = [
      clip({ negativePrompt: "no rain", cameraControl: "dolly-in" }),
      clip({
        prompt: "landing",
        transition: "fade",
        fadeFrames: 12,
        cameraControl: "/models/camera/custom.safetensors",
        sourceImage: { filename: "landing.png", base64: "REVG" },
      }),
    ];
    const options = {
      motionTailFrames: 17,
      enableAudio: true,
      openingImage: { filename: "opening.png", base64: "QUJD" },
    };

    const requestStages = buildChainRequest(shared(), clips, options).stages;
    const scriptStages = clipsToChainScript(shared(), clips, options).stages;

    expect(scriptStages).toEqual(
      requestStages.map(({ source_image, ...stage }) => ({
        ...stage,
        ...(source_image ? { source_image_b64: source_image } : {}),
      })),
    );
  });

  it("converts a job's effective script into editable clips and back", () => {
    const script: ChainScript = {
      schema: "mold.chain.v1",
      chain: {
        model: "ltx-2-19b-distilled:fp8",
        motion_tail_frames: 17,
        width: 1216,
        height: 704,
        fps: 24,
        steps: 8,
        guidance: 3,
        strength: 0.6,
        enable_audio: true,
      },
      stages: [
        {
          prompt: "opening",
          frames: 97,
          source_image_b64: "QUJD",
          loras: [
            {
              path: "camera-control:dolly-left",
              scale: 1,
              name: "Dolly left",
            },
          ],
        },
        { prompt: "landing", frames: 33, transition: "fade", fade_frames: 8 },
      ],
    };

    const loaded = chainScriptToClips(script);
    expect(loaded.clips).toHaveLength(2);
    expect(loaded.clips[0]?.prompt).toBe("opening");
    expect(loaded.clips[0]?.frames).toBe(97);
    expect(loaded.clips[1]?.transition).toBe("fade");
    expect(loaded.clips[0]?.cameraControl).toBe("dolly-left");
    expect(loaded.clips[1]?.fadeFrames).toBe(8);
    expect(loaded.openingImage?.base64).toBe("QUJD");
    expect(loaded.enableAudio).toBe(true);
    expect(loaded.shared.model).toBe("ltx-2-19b-distilled:fp8");
    expect(loaded.shared.width).toBe(1216);
    expect(loaded.shared.strength).toBe(0.6);

    const back = clipsToChainScript(shared(), loaded.clips, {
      motionTailFrames: 17,
      enableAudio: true,
      openingImage: loaded.openingImage,
    });
    expect(back.schema).toBe("mold.chain.v1");
    expect(back.stages).toHaveLength(2);
    expect(back.stages[1]?.transition).toBe("fade");
    expect(back.stages[1]?.fade_frames).toBe(8);
    expect(back.stages[0]?.source_image_b64).toBe("QUJD");
    expect(back.stages[0]?.loras?.[0]?.path).toBe("camera-control:dolly-left");
    expect(back.chain.strength).toBe(1);
  });

  it("recovers source dimensions from script media without adding wire fields", () => {
    const script: ChainScript = {
      chain: { model: "ltx-2-19b-distilled:fp8" },
      stages: [
        {
          prompt: "opening",
          frames: 97,
          source_image_b64: PNG_1170_BY_2532,
        },
        { prompt: "landing", frames: 97 },
      ],
    };

    const loaded = chainScriptToClips(script);
    expect(loaded.openingImage).toMatchObject({
      width: 1170,
      height: 2532,
      base64: PNG_1170_BY_2532,
    });
    const back = clipsToChainScript(shared(), loaded.clips, {
      motionTailFrames: 17,
      enableAudio: false,
      openingImage: loaded.openingImage,
    });
    expect(back.stages[0]).not.toHaveProperty("source_image_width");
    expect(back.stages[0]).not.toHaveProperty("source_image_height");
    expect(back.stages[0]?.source_image_b64).toBe(PNG_1170_BY_2532);
  });

  it("refuses to silently drop an opening image whose payload is still restoring", () => {
    expect(() =>
      buildChainRequest(shared(), [clip(), clip()], {
        motionTailFrames: 17,
        enableAudio: false,
        openingImage: { filename: "open.png", base64: null },
      }),
    ).toThrow("Reattach opening image open.png");
  });
});

describe("stageInvalidation", () => {
  const baseline = [
    clip({ prompt: "one", frames: 97, transition: "smooth" }),
    clip({ prompt: "two", frames: 97, transition: "smooth" }),
    clip({ prompt: "three", frames: 97, transition: "cut" }),
  ];

  function plan(
    current: SequenceClipForm[],
    opts: { chainLevelDirty?: boolean; completedStages?: number } = {},
  ) {
    return stageInvalidation(baseline, current, {
      chainLevelDirty: opts.chainLevelDirty ?? false,
      completedStages: opts.completedStages ?? 3,
    });
  }

  it("keeps everything cached when nothing changed", () => {
    const p = plan(baseline.map((c) => ({ ...c })));
    expect(p.firstDirtyStage).toBeNull();
    expect(p.perClip).toEqual(["cached", "cached", "cached"]);
  });

  it("edits to clip k re-render k..end (smooth carry cascades)", () => {
    const current = baseline.map((c) => ({ ...c }));
    const middle = current[1];
    if (middle) middle.prompt = "two, revised";
    const p = plan(current);
    expect(p.firstDirtyStage).toBe(1);
    expect(p.perClip).toEqual(["cached", "rerender", "rerender"]);
  });

  it("treats a per-clip camera motion change as a pixel change", () => {
    const current = baseline.map((c) => ({ ...c }));
    current[1]!.cameraControl = "jib-up";
    const p = plan(current);
    expect(p.firstDirtyStage).toBe(1);
    expect(p.perClip).toEqual(["cached", "rerender", "rerender"]);
  });

  it("chain-level changes re-render everything", () => {
    const p = plan(
      baseline.map((c) => ({ ...c })),
      { chainLevelDirty: true },
    );
    expect(p.firstDirtyStage).toBe(0);
    expect(p.perClip).toEqual(["rerender", "rerender", "rerender"]);
  });

  it("cut↔fade and fade-length edits are finalize-only (no re-render)", () => {
    // Under raw segments, cut vs fade only changes how finalize blends the
    // seam — the rendered pixels are identical. Smooth is different: it
    // carries motion INTO the clip, so it participates in render identity.
    const current = baseline.map((c) => ({ ...c }));
    const last = current[2];
    if (last) {
      last.transition = "fade";
      last.fadeFrames = 12;
    }
    const p = plan(current);
    expect(p.firstDirtyStage).toBeNull();
    expect(p.perClip).toEqual(["cached", "cached", "cached"]);
  });

  it("smooth↔other changes the carry and re-renders from that clip", () => {
    const current = baseline.map((c) => ({ ...c }));
    const middle = current[1];
    if (middle) middle.transition = "cut";
    const p = plan(current);
    expect(p.firstDirtyStage).toBe(1);
  });

  it("appended clips render as new; the old prefix stays cached", () => {
    const current = [
      ...baseline.map((c) => ({ ...c })),
      clip({ prompt: "four" }),
    ];
    const p = plan(current);
    expect(p.firstDirtyStage).toBe(3);
    expect(p.perClip).toEqual(["cached", "cached", "cached", "new"]);
  });

  it("removing trailing clips is boundary-only (zero renders)", () => {
    const current = baseline.slice(0, 2).map((c) => ({ ...c }));
    const p = plan(current);
    expect(p.firstDirtyStage).toBeNull();
    expect(p.perClip).toEqual(["cached", "cached"]);
  });

  it("never marks unrendered stages as cached", () => {
    const p = plan(
      baseline.map((c) => ({ ...c })),
      { completedStages: 1 },
    );
    expect(p.perClip).toEqual(["cached", "rerender", "rerender"]);
  });
});
