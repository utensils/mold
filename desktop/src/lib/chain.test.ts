import { describe, expect, it } from "vitest";
import {
  chainToToml,
  cycleTransition,
  durationSeconds,
  estimatedTotalFrames,
  frames8n1Error,
  isLtx2FrameCount,
  snapFrames,
} from "./chain";
import {
  chainFormToRequest,
  chainFormToScript,
  newChainForm,
  newStage,
  tomlToChainForm,
} from "./chainForm";
import type { ChainStage, TransitionMode } from "./api/types";

describe("frame-count validation", () => {
  it("starts new guided sequences at the Plato-verified comfortable duration", () => {
    expect(newChainForm().stages.map((stage) => stage.frames)).toEqual([25, 25]);
  });

  it("accepts only 8n+1 counts", () => {
    for (const ok of [1, 9, 17, 25, 33, 97]) expect(isLtx2FrameCount(ok)).toBe(true);
    for (const bad of [0, 2, 8, 50, 96, 98, 100]) expect(isLtx2FrameCount(bad)).toBe(false);
  });

  it("returns the design-copy error for invalid counts", () => {
    expect(frames8n1Error(97)).toBeNull();
    expect(frames8n1Error(50)).toBe("Frames must be 8n+1 — try 97.");
  });

  it("snaps to the nearest valid count", () => {
    expect(snapFrames(50)).toBe(49);
    expect(snapFrames(52)).toBe(49);
    expect(snapFrames(54)).toBe(57);
    expect(snapFrames(0)).toBe(1);
    expect(snapFrames(97)).toBe(97);
  });
});

describe("estimatedTotalFrames", () => {
  const stages = (specs: [TransitionMode, number, number?][]): ChainStage[] =>
    specs.map(([transition, frames, fade]) => ({
      prompt: "x",
      frames,
      transition,
      ...(fade != null ? { fade_frames: fade } : {}),
    }));

  it("drops the motion tail on smooth continuations (matches chain.rs)", () => {
    // 97 + (97-25) + (97-25) = 241
    expect(
      estimatedTotalFrames(
        stages([
          ["smooth", 97],
          ["smooth", 97],
          ["smooth", 97],
        ]),
        25,
      ),
    ).toBe(241);
  });

  it("keeps the whole clip after a cut", () => {
    // 97 + 97 (cut) + (97-25) (smooth) = 266
    expect(
      estimatedTotalFrames(
        stages([
          ["smooth", 97],
          ["cut", 97],
          ["smooth", 97],
        ]),
        25,
      ),
    ).toBe(266);
  });

  it("nets minus fade_frames on a fade", () => {
    // 97 + 97 + (97-8) = 283
    expect(
      estimatedTotalFrames(
        stages([
          ["smooth", 97],
          ["cut", 97],
          ["fade", 97, 8],
        ]),
        25,
      ),
    ).toBe(283);
  });

  it("computes duration at fps", () => {
    expect(durationSeconds(240, 24)).toBe(10);
    expect(durationSeconds(240, 0)).toBe(0);
  });
});

describe("cycleTransition", () => {
  it("cycles smooth → cut → fade → smooth", () => {
    expect(cycleTransition("smooth")).toBe("cut");
    expect(cycleTransition("cut")).toBe("fade");
    expect(cycleTransition("fade")).toBe("smooth");
  });
});

describe("newChainForm", () => {
  it("starts a guided sequence with two blank clips", () => {
    const form = newChainForm();
    expect(form.stages).toHaveLength(2);
    expect(form.stages.map((stage) => stage.prompt)).toEqual(["", ""]);
  });
});

describe("chainToToml", () => {
  it("emits mold.chain.v1 with a [chain] table and one [[stage]] per stage", () => {
    const form = newChainForm();
    form.model = "ltx-2-19b-distilled:fp8";
    form.seed = "42";
    form.enableAudio = true;
    form.stages = [
      newStage("dawn over the sea"),
      { ...newStage("a storm rolls in"), transition: "fade", fadeFrames: 12 },
    ];
    const toml = chainToToml(chainFormToScript(form));

    expect(toml).toContain('schema = "mold.chain.v1"');
    expect(toml).toContain("[chain]");
    expect(toml).toContain('model = "ltx-2-19b-distilled:fp8"');
    expect(toml).toContain("seed = 42");
    expect(toml).toContain("enable_audio = true");
    expect(toml).toContain('output_format = "mp4"');
    // Two stages, first coerced to smooth.
    expect(toml.match(/\[\[stage\]\]/g)).toHaveLength(2);
    expect(toml).toContain('prompt = "dawn over the sea"');
    expect(toml).toContain('transition = "smooth"');
    expect(toml).toContain('transition = "fade"');
    expect(toml).toContain("fade_frames = 12");
  });

  it("omits seed and enable_audio when unset", () => {
    const form = newChainForm();
    form.model = "ltx2:fp8";
    const toml = chainToToml(chainFormToScript(form));
    expect(toml).not.toContain("seed =");
    expect(toml).not.toContain("enable_audio");
  });

  it("escapes quotes and newlines in prompts", () => {
    const form = newChainForm();
    form.model = "ltx2:fp8";
    form.stages = [newStage('a "quoted"\nprompt')];
    const toml = chainToToml(chainFormToScript(form));
    expect(toml).toContain('prompt = "a \\"quoted\\"\\nprompt"');
  });
});

describe("tomlToChainForm", () => {
  it("round-trips a form through chainToToml back to itself", () => {
    const form = newChainForm();
    form.model = "ltx-2-19b-distilled:fp8";
    form.width = 1216;
    form.height = 704;
    form.fps = 24;
    form.seed = "42";
    form.steps = 8;
    form.motionTailFrames = 17;
    form.enableAudio = true;
    form.stages = [
      { ...newStage('dawn over the "sea"'), frames: 97 },
      { ...newStage("a storm rolls in"), transition: "fade", fadeFrames: 12, frames: 49 },
    ];

    const parsed = tomlToChainForm(chainToToml(chainFormToScript(form)));
    expect(parsed).toEqual(form);
  });

  it("accepts a document with no schema key and coerces stage 0 to smooth", () => {
    const toml = [
      "[chain]",
      'model = "ltx2:fp8"',
      "width = 704",
      "height = 416",
      "fps = 24",
      "steps = 8",
      "guidance = 3.0",
      "strength = 1.0",
      "motion_tail_frames = 17",
      'output_format = "mp4"',
      "",
      "[[stage]]",
      'prompt = "a bird"',
      "frames = 33",
      'transition = "cut"',
    ].join("\n");
    const form = tomlToChainForm(toml);
    expect(form.model).toBe("ltx2:fp8");
    expect(form.stages).toHaveLength(1);
    expect(form.stages[0]!.transition).toBe("smooth");
    expect(form.seed).toBe("");
    expect(form.enableAudio).toBe(false);
  });

  it("rejects a foreign schema", () => {
    expect(() => tomlToChainForm('schema = "mold.chain.v2"\n[chain]\nmodel = "x"')).toThrow(
      /mold\.chain\.v1/,
    );
  });

  it("throws on malformed TOML", () => {
    expect(() => tomlToChainForm("this is not = = toml")).toThrow();
  });
});

// Base64 for a 1x1 PNG-ish payload — content doesn't matter, only plumbing.
const IMG_A = "aGVyby1zdGlsbA==";
const IMG_B = "c3RhZ2UtdHdv";

describe("chain source images", () => {
  it("projects per-stage sourceImage onto the wire stages", () => {
    const form = newChainForm();
    form.model = "ltx2:fp8";
    form.stages = [newStage("dawn"), { ...newStage("storm"), sourceImage: IMG_B }];
    const req = chainFormToRequest(form);
    expect(req.stages[0]!.source_image).toBeUndefined();
    expect(req.stages[1]!.source_image).toBe(IMG_B);
  });

  it("projects the chain-level startImage onto stage 0 only", () => {
    const form = newChainForm();
    form.model = "ltx2:fp8";
    form.startImage = IMG_A;
    form.stages = [newStage("dawn"), newStage("storm")];
    const req = chainFormToRequest(form);
    expect(req.stages[0]!.source_image).toBe(IMG_A);
    expect(req.stages[1]!.source_image).toBeUndefined();
    // Canonical form: the server ignores top-level source_image, so the
    // request must carry the start image on stages[0], not the chain.
    expect("source_image" in req).toBe(false);
  });

  it("prefers stage 0's own image over the chain-level startImage", () => {
    const form = newChainForm();
    form.model = "ltx2:fp8";
    form.startImage = IMG_A;
    form.stages = [{ ...newStage("dawn"), sourceImage: IMG_B }];
    expect(chainFormToRequest(form).stages[0]!.source_image).toBe(IMG_B);
    expect(chainFormToScript(form).stage[0]!.source_image).toBe(IMG_B);
  });

  it("omits source_image entirely when nothing is attached", () => {
    const form = newChainForm();
    form.model = "ltx2:fp8";
    const req = chainFormToRequest(form);
    expect("source_image" in req.stages[0]!).toBe(false);
    expect(chainToToml(chainFormToScript(form))).not.toContain("source_image");
  });

  it("keeps images aligned to their stages across reorder and removal", () => {
    const form = newChainForm();
    form.model = "ltx2:fp8";
    form.stages = [
      newStage("one"),
      { ...newStage("two"), sourceImage: IMG_B },
      { ...newStage("three"), sourceImage: IMG_A },
    ];
    // Move stage 2 ("two") to the front, then drop the middle stage.
    const [moved] = form.stages.splice(1, 1);
    form.stages.splice(0, 0, moved!);
    form.stages.splice(1, 1); // removes "one"
    const req = chainFormToRequest(form);
    expect(req.stages.map((s) => [s.prompt, s.source_image ?? null])).toEqual([
      ["two", IMG_B],
      ["three", IMG_A],
    ]);
  });

  it("emits source_image_b64 per stage in TOML export (start image folded into stage 0)", () => {
    const form = newChainForm();
    form.model = "ltx2:fp8";
    form.startImage = IMG_A;
    form.stages = [newStage("dawn"), { ...newStage("storm"), sourceImage: IMG_B }];
    const toml = chainToToml(chainFormToScript(form));
    expect(toml).toContain(`source_image_b64 = "${IMG_A}"`);
    expect(toml).toContain(`source_image_b64 = "${IMG_B}"`);
    expect(toml).not.toContain("bytes omitted");
  });

  it("parses source_image_b64 back into the stage form", () => {
    const toml = [
      "[chain]",
      'model = "ltx2:fp8"',
      "width = 704",
      "height = 416",
      "fps = 24",
      "steps = 8",
      "guidance = 3.0",
      "strength = 1.0",
      "motion_tail_frames = 17",
      'output_format = "mp4"',
      "",
      "[[stage]]",
      'prompt = "a bird"',
      "frames = 33",
      `source_image_b64 = "${IMG_A}"`,
    ].join("\n");
    const form = tomlToChainForm(toml);
    expect(form.stages[0]!.sourceImage).toBe(IMG_A);
    expect(form.startImage).toBeNull();
  });

  it("accepts the canonical source_image key on import", () => {
    const toml = [
      "[chain]",
      'model = "ltx2:fp8"',
      "width = 704",
      "height = 416",
      "fps = 24",
      "steps = 8",
      "guidance = 3.0",
      "strength = 1.0",
      "motion_tail_frames = 17",
      'output_format = "mp4"',
      "",
      "[[stage]]",
      'prompt = "a bird"',
      "frames = 33",
      `source_image = "${IMG_A}"`,
    ].join("\n");
    expect(tomlToChainForm(toml).stages[0]!.sourceImage).toBe(IMG_A);
  });

  it("rejects source_image_path with a clear error", () => {
    const toml = [
      "[chain]",
      'model = "ltx2:fp8"',
      "width = 704",
      "height = 416",
      "fps = 24",
      "steps = 8",
      "guidance = 3.0",
      "strength = 1.0",
      "motion_tail_frames = 17",
      'output_format = "mp4"',
      "",
      "[[stage]]",
      'prompt = "a bird"',
      "frames = 33",
      'source_image_path = "./hero.png"',
    ].join("\n");
    expect(() => tomlToChainForm(toml)).toThrow(/source_image_b64/);
  });

  it("rejects a stage that sets more than one image field (matches chain_toml.rs)", () => {
    const toml = [
      "[chain]",
      'model = "ltx2:fp8"',
      "width = 704",
      "height = 416",
      "fps = 24",
      "steps = 8",
      "guidance = 3.0",
      "strength = 1.0",
      "motion_tail_frames = 17",
      'output_format = "mp4"',
      "",
      "[[stage]]",
      'prompt = "a bird"',
      "frames = 33",
      `source_image = "${IMG_A}"`,
      `source_image_b64 = "${IMG_B}"`,
    ].join("\n");
    expect(() => tomlToChainForm(toml)).toThrow(/at most one/);
  });

  it("rejects a non-string source_image_b64 with a stage-specific error", () => {
    const toml = [
      "[chain]",
      'model = "ltx2:fp8"',
      "",
      "[[stage]]",
      'prompt = "a bird"',
      "frames = 33",
      "source_image_b64 = 42",
    ].join("\n");
    expect(() => tomlToChainForm(toml)).toThrow(/Stage 1: source_image_b64 must be a string/);
  });

  it("rejects a non-string canonical source_image with a stage-specific error", () => {
    const toml = [
      "[chain]",
      'model = "ltx2:fp8"',
      "",
      "[[stage]]",
      'prompt = "a bird"',
      "frames = 33",
      "source_image = [1, 2, 3]",
    ].join("\n");
    expect(() => tomlToChainForm(toml)).toThrow(/Stage 1: source_image must be a string/);
  });

  it("round-trips per-stage images through TOML exactly", () => {
    const form = newChainForm();
    form.model = "ltx-2-19b-distilled:fp8";
    form.stages = [
      { ...newStage("dawn"), sourceImage: IMG_A },
      { ...newStage("storm"), transition: "cut", sourceImage: IMG_B },
    ];
    const parsed = tomlToChainForm(chainToToml(chainFormToScript(form)));
    expect(parsed).toEqual(form);
  });

  it("round-trips a chain-level start image as a wire-equivalent form", () => {
    const form = newChainForm();
    form.model = "ltx2:fp8";
    form.startImage = IMG_A;
    const parsed = tomlToChainForm(chainToToml(chainFormToScript(form)));
    // The start image comes back attached to stage 0 (TOML has no chain-level
    // image), so the forms differ — but the requests they build must match.
    expect(parsed.startImage).toBeNull();
    expect(parsed.stages[0]!.sourceImage).toBe(IMG_A);
    expect(chainFormToRequest(parsed)).toEqual(chainFormToRequest(form));
  });
});
