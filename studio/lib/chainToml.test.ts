import { describe, expect, it } from "vitest";
import {
  CHAIN_SCHEMA,
  parseChainScript,
  serializeChainScript,
} from "./chainToml";
import type { ChainScript } from "./api/chainTypes";

function script(): ChainScript {
  return {
    schema: CHAIN_SCHEMA,
    chain: {
      model: "ltx-2-19b-distilled:fp8",
      motion_tail_frames: 17,
      width: 1216,
      height: 704,
      fps: 24,
      seed: "42",
      steps: 8,
      guidance: 3,
      enable_audio: true,
    },
    stages: [
      { prompt: "opening shot", frames: 97, transition: "smooth" },
      {
        prompt: "the reveal",
        frames: 33,
        transition: "fade",
        fade_frames: 8,
        negative_prompt: "blurry",
        source_image_b64: "QUJD",
      },
    ],
  };
}

describe("serializeChainScript", () => {
  it("emits the mold.chain.v1 header, [chain] table, and [[stage]] tables", () => {
    const toml = serializeChainScript(script());
    expect(CHAIN_SCHEMA).toBe("mold.chain.v1");
    expect(toml).toContain('schema = "mold.chain.v1"');
    expect(toml).toContain("[chain]");
    expect(toml).toContain("[[stage]]");
    expect(toml).toContain('model = "ltx-2-19b-distilled:fp8"');
    expect(toml).toContain("frames = 97");
    expect(toml).toContain("motion_tail_frames = 17");
  });

  it("always emits the Rust-required strength and output_format fields", () => {
    // mold-core's ChainScriptChain has no serde defaults for these — a TOML
    // missing them is rejected by `mold run --script`.
    const toml = serializeChainScript(script());
    expect(toml).toContain("strength = 1.0");
    expect(toml).toContain('output_format = "mp4"');
  });

  it("keeps whole-number f64 fields as TOML floats", () => {
    // smol-toml's stringify collapses 1.0 to the integer 1, which Rust's f64
    // fields reject — the hand-rolled writer is why this holds.
    const toml = serializeChainScript(script());
    expect(toml).toContain("guidance = 3.0");
    expect(toml).not.toMatch(/guidance = 3(\n|$)/);
    expect(parseChainScript(toml).chain.guidance).toBe(3);
  });

  it("passes explicit strength and output_format through instead of the defaults", () => {
    const s = script() as ChainScript & {
      chain: { strength?: number; output_format?: string };
    };
    s.chain.strength = 0.75;
    s.chain.output_format = "webm";
    const toml = serializeChainScript(s);
    expect(toml).toContain("strength = 0.75");
    expect(toml).toContain('output_format = "webm"');
    expect(parseChainScript(toml).chain.strength).toBe(0.75);
  });

  it("emits the seed as a bare TOML integer, u64-precise", () => {
    const s = script();
    s.chain.seed = "18446744073709551615";
    const toml = serializeChainScript(s);
    expect(toml).toContain("seed = 18446744073709551615");
    // Rust deserializes seed as u64 — a quoted string would be rejected.
    expect(toml).not.toContain('seed = "');
    expect(serializeChainScript(script())).toMatch(/seed = 42(\n|$)/);
  });

  it("emits per-stage seed_offset as a bare integer, u64-precise", () => {
    const s = script();
    s.stages[1]!.seed_offset = "18446744073709551614";
    const toml = serializeChainScript(s);
    expect(toml).toContain("seed_offset = 18446744073709551614");
    expect(parseChainScript(toml).stages[1]!.seed_offset).toBe(
      "18446744073709551614",
    );
  });

  it("omits null and undefined fields", () => {
    const s = script();
    s.chain.seed = null;
    s.chain.enable_audio = null;
    const toml = serializeChainScript(s);
    expect(toml).not.toContain("seed");
    expect(toml).not.toContain("enable_audio");
    expect(toml).not.toContain("negative_prompt =\n");
  });

  it("emits fade_frames only on fade seams and inline images as source_image_b64", () => {
    const s = script();
    s.stages[0]!.source_image_b64 = "aGVsbG8=";
    s.stages[0]!.fade_frames = 12;
    const toml = serializeChainScript(s);
    expect(toml).toContain('source_image_b64 = "aGVsbG8="');
    expect(toml).toContain("fade_frames = 8");
    expect(toml).not.toContain("fade_frames = 12");
  });

  it("escapes quotes, backslashes, and newlines in prompts", () => {
    const s = script();
    s.stages[0]!.prompt = 'a "quoted" prompt\nwith C:\\paths';
    const toml = serializeChainScript(s);
    expect(toml).toContain(
      'prompt = "a \\"quoted\\" prompt\\nwith C:\\\\paths"',
    );
    expect(parseChainScript(toml).stages[0]!.prompt).toBe(
      'a "quoted" prompt\nwith C:\\paths',
    );
  });
});

describe("parseChainScript", () => {
  it("round-trips serializeChainScript output", () => {
    const parsed = parseChainScript(serializeChainScript(script()));
    expect(parsed.schema).toBe(CHAIN_SCHEMA);
    expect(parsed.chain.model).toBe("ltx-2-19b-distilled:fp8");
    expect(parsed.chain.motion_tail_frames).toBe(17);
    expect(parsed.chain.width).toBe(1216);
    expect(parsed.chain.height).toBe(704);
    expect(parsed.chain.fps).toBe(24);
    expect(parsed.chain.steps).toBe(8);
    expect(parsed.chain.seed).toBe("42");
    expect(parsed.chain.enable_audio).toBe(true);
    expect(parsed.stages).toHaveLength(2);
    expect(parsed.stages[0]!.prompt).toBe("opening shot");
    expect(parsed.stages[1]).toMatchObject({
      prompt: "the reveal",
      frames: 33,
      transition: "fade",
      fade_frames: 8,
      negative_prompt: "blurry",
      source_image_b64: "QUJD",
    });
  });

  it("accepts a script without a schema key (the server injects it too)", () => {
    const parsed = parseChainScript(
      [
        "[chain]",
        'model = "ltx-video"',
        "",
        "[[stage]]",
        'prompt = "a"',
        "frames = 25",
        "",
        "[[stage]]",
        'prompt = "b"',
        "frames = 25",
      ].join("\n"),
    );
    expect(parsed.schema).toBe(CHAIN_SCHEMA);
    expect(parsed.chain.model).toBe("ltx-video");
    expect(parsed.stages).toHaveLength(2);
    expect(parsed.stages[0]!.frames).toBe(25);
  });

  it("accepts the natural plural [[stages]] key from hand-written documents", () => {
    const parsed = parseChainScript(
      [
        "[chain]",
        'model = "m"',
        "",
        "[[stages]]",
        'prompt = "a"',
        "frames = 25",
      ].join("\n"),
    );
    expect(parsed.stages).toHaveLength(1);
    expect(parsed.stages[0]!.prompt).toBe("a");
  });

  it("preserves u64 seeds beyond Number.MAX_SAFE_INTEGER as decimal strings", () => {
    const parsed = parseChainScript(
      [
        "[chain]",
        'model = "m"',
        "seed = 18446744073709551615",
        "",
        "[[stage]]",
        'prompt = "a"',
      ].join("\n"),
    );
    expect(parsed.chain.seed).toBe("18446744073709551615");
  });

  it("accepts the canonical source_image key and folds it into source_image_b64", () => {
    const parsed = parseChainScript(
      [
        "[chain]",
        'model = "m"',
        "",
        "[[stage]]",
        'prompt = "a"',
        'source_image = "aGk="',
      ].join("\n"),
    );
    expect(parsed.stages[0]!.source_image_b64).toBe("aGk=");
  });

  it("rejects source_image_path with a friendly message (no script folder here)", () => {
    expect(() =>
      parseChainScript(
        [
          "[chain]",
          'model = "m"',
          "",
          "[[stage]]",
          'prompt = "a"',
          'source_image_path = "./hero.png"',
        ].join("\n"),
      ),
    ).toThrow(/source_image_path/);
  });

  it("rejects a stage that sets more than one image field", () => {
    expect(() =>
      parseChainScript(
        [
          "[chain]",
          'model = "m"',
          "",
          "[[stage]]",
          'prompt = "a"',
          'source_image = "aGk="',
          'source_image_b64 = "aGk="',
        ].join("\n"),
      ),
    ).toThrow(/at most one/);
  });

  it("rejects malformed TOML, foreign schemas, and missing tables with friendly messages", () => {
    expect(() => parseChainScript("not = [toml")).toThrow(
      /Couldn't parse the TOML/,
    );
    expect(() => parseChainScript('schema = "mold.chain.v2"')).toThrow(
      /mold\.chain\.v1/,
    );
    expect(() =>
      parseChainScript('schema = "mold.chain.v2"\n[chain]\nmodel = "m"'),
    ).toThrow(/mold\.chain\.v1/);
    expect(() => parseChainScript('schema = "mold.chain.v1"')).toThrow(
      /\[chain\]/,
    );
    expect(() =>
      parseChainScript('schema = "mold.chain.v1"\n[chain]\nmodel = "m"'),
    ).toThrow(/\[\[stage\]\]/);
    expect(() => parseChainScript('[chain]\nmodel = "m"')).toThrow(
      /\[\[stage\]\]/,
    );
  });

  it("coerces stage 0's transition to smooth and defaults unknown transitions", () => {
    const parsed = parseChainScript(
      [
        "[chain]",
        'model = "m"',
        "",
        "[[stage]]",
        'prompt = "a"',
        'transition = "fade"',
        "",
        "[[stage]]",
        'prompt = "b"',
        'transition = "sparkle"',
      ].join("\n"),
    );
    expect(parsed.stages[0]!.transition).toBe("smooth");
    expect(parsed.stages[1]!.transition).toBe("smooth");
  });
});
