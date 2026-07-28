import { describe, expect, it } from "vitest";
import { parseChainScript, serializeChainScript } from "./chainToml";
import type { ChainScript } from "./api/chainTypes";

function script(): ChainScript {
  return {
    schema: "mold.chain.v1",
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
        frames: 97,
        transition: "fade",
        fade_frames: 8,
        negative_prompt: "blurry",
      },
    ],
  };
}

describe("serializeChainScript", () => {
  it("emits the mold.chain.v1 header, [chain] table, and [[stage]] tables", () => {
    const toml = serializeChainScript(script());
    expect(toml).toContain('schema = "mold.chain.v1"');
    expect(toml).toContain("[chain]");
    expect(toml).toMatch(/\[\[stage\]\]/);
    expect(toml).toContain('model = "ltx-2-19b-distilled:fp8"');
    expect(toml).toContain("frames = 97");
  });

  it("always emits the Rust-required strength and output_format fields", () => {
    // mold-core's ChainScriptChain has no serde defaults for these — a TOML
    // missing them is rejected by `mold run --script`.
    const toml = serializeChainScript(script());
    expect(toml).toContain("strength = 1.0");
    expect(toml).toContain('output_format = "mp4"');
  });

  it("emits the seed as a bare TOML integer, u64-precise", () => {
    const s = script();
    s.chain.seed = "18446744073709551615";
    const toml = serializeChainScript(s);
    expect(toml).toContain("seed = 18446744073709551615");
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
    const toml = serializeChainScript(s);
    expect(toml).toContain('source_image_b64 = "aGVsbG8="');
    expect(toml).toContain("fade_frames = 8");
  });
});

describe("parseChainScript", () => {
  it("round-trips serializeChainScript output", () => {
    const parsed = parseChainScript(serializeChainScript(script()));
    expect(parsed.schema).toBe("mold.chain.v1");
    expect(parsed.chain.model).toBe("ltx-2-19b-distilled:fp8");
    expect(parsed.chain.seed).toBe("42");
    expect(parsed.chain.enable_audio).toBe(true);
    expect(parsed.stages).toHaveLength(2);
    expect(parsed.stages[1]).toMatchObject({
      prompt: "the reveal",
      frames: 97,
      transition: "fade",
      fade_frames: 8,
      negative_prompt: "blurry",
    });
  });

  it("accepts a script without a schema key (the server injects it too)", () => {
    const parsed = parseChainScript(
      ['[chain]', 'model = "ltx-video"', "", "[[stage]]", 'prompt = "a"', "frames = 25"].join("\n"),
    );
    expect(parsed.chain.model).toBe("ltx-video");
    expect(parsed.stages[0]!.frames).toBe(25);
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
      ["[chain]", 'model = "m"', "", "[[stage]]", 'prompt = "a"', 'source_image = "aGk="'].join(
        "\n",
      ),
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
    expect(() => parseChainScript("not = [toml")).toThrow(/Couldn't parse the TOML/);
    expect(() => parseChainScript('schema = "mold.chain.v2"\n[chain]\nmodel = "m"')).toThrow(
      /mold\.chain\.v1/,
    );
    expect(() => parseChainScript('schema = "mold.chain.v1"')).toThrow(/\[chain\]/);
    expect(() => parseChainScript('[chain]\nmodel = "m"')).toThrow(/\[\[stage\]\]/);
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
