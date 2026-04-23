import { describe, expect, it } from "vitest";
import { readChainScript, writeChainScript } from "./chainToml";
import type { ChainScriptToml } from "./chainToml";

describe("chainToml", () => {
  const minimal: ChainScriptToml = {
    schema: "mold.chain.v1",
    chain: {
      model: "ltx-2-19b-distilled:fp8",
      width: 1216,
      height: 704,
      fps: 24,
      steps: 8,
      guidance: 3.0,
      strength: 1.0,
      motion_tail_frames: 25,
      output_format: "mp4",
    },
    stage: [
      { prompt: "first", frames: 97 },
      { prompt: "second", frames: 49, transition: "cut" },
    ],
  };

  it("round-trips a minimal script", () => {
    const toml = writeChainScript(minimal);
    const back = readChainScript(toml);
    expect(back.schema).toBe("mold.chain.v1");
    expect(back.stage.length).toBe(2);
    expect(back.stage[0].prompt).toBe("first");
    expect(back.stage[0].frames).toBe(97);
    expect(back.stage[1].transition).toBe("cut");
    expect(back.chain.model).toBe("ltx-2-19b-distilled:fp8");
    expect(back.chain.guidance).toBe(3.0);
  });

  it("preserves optional stage fields", () => {
    const script: ChainScriptToml = {
      ...minimal,
      stage: [
        {
          prompt: "scene one",
          frames: 97,
          negative_prompt: "blurry",
          seed_offset: 5,
        },
        {
          prompt: "scene two",
          frames: 49,
          transition: "fade",
          fade_frames: 12,
        },
      ],
    };
    const back = readChainScript(writeChainScript(script));
    expect(back.stage[0].negative_prompt).toBe("blurry");
    expect(back.stage[0].seed_offset).toBe(5);
    expect(back.stage[1].fade_frames).toBe(12);
  });

  it("rejects unknown schema", () => {
    const bad = `schema = "mold.chain.v99"\n[chain]\nmodel = "x"\nwidth=1\nheight=1\nfps=1\nsteps=1\nguidance=1.0\nstrength=1.0\nmotion_tail_frames=0\noutput_format="mp4"\n[[stage]]\nprompt="x"\nframes=1`;
    expect(() => readChainScript(bad)).toThrow(/v99/);
  });

  it("rejects missing [chain]", () => {
    const bad = `schema = "mold.chain.v1"\n[[stage]]\nprompt="x"\nframes=1`;
    expect(() => readChainScript(bad)).toThrow(/missing \[chain\]/);
  });

  it("rejects missing [[stage]]", () => {
    const bad = `schema = "mold.chain.v1"\n[chain]\nmodel = "x"\nwidth=1\nheight=1\nfps=1\nsteps=1\nguidance=1.0\nstrength=1.0\nmotion_tail_frames=0\noutput_format="mp4"`;
    expect(() => readChainScript(bad)).toThrow(/missing \[\[stage\]\]/);
  });

  it("defaults schema to v1 when missing", () => {
    const noSchema = `[chain]\nmodel = "x"\nwidth=1\nheight=1\nfps=1\nsteps=1\nguidance=1.0\nstrength=1.0\nmotion_tail_frames=0\noutput_format="mp4"\n[[stage]]\nprompt="x"\nframes=1`;
    const result = readChainScript(noSchema);
    expect(result.schema).toBe("mold.chain.v1");
  });
});
