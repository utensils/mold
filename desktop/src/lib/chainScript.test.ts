import { describe, expect, it } from "vitest";
import { countLeadingCompletedStages, normalizeServerChainScript } from "./chainScript";
import type { ChainJobStageDetail } from "@studio/lib/api/chainTypes";

describe("normalizeServerChainScript", () => {
  it("normalizes the server wire (stage key, source_image, numeric seed)", () => {
    // Today's server serializes ChainScript with serde rename `stage` and the
    // canonical `source_image` field; the studio contract reads `stages` +
    // `source_image_b64`. Both shapes must load.
    const script = normalizeServerChainScript({
      schema: "mold.chain.v1",
      chain: {
        model: "ltx-video",
        width: 1216,
        height: 704,
        fps: 24,
        seed: 42,
        steps: 8,
        guidance: 3,
        strength: 1,
        motion_tail_frames: 0,
        output_format: "mp4",
      },
      stage: [
        { prompt: "opening", frames: 25, transition: "smooth", source_image: "aGk=" },
        { prompt: "next", frames: 25, transition: "fade", fade_frames: 8, negative_prompt: "x" },
      ],
    });
    expect(script?.chain).toMatchObject({ model: "ltx-video", width: 1216, seed: "42" });
    expect(script?.stages).toHaveLength(2);
    expect(script?.stages[0]).toMatchObject({ prompt: "opening", source_image_b64: "aGk=" });
    expect(script?.stages[1]).toMatchObject({ transition: "fade", fade_frames: 8 });
  });

  it("passes through the studio shape (stages key, source_image_b64) unchanged", () => {
    const script = normalizeServerChainScript({
      chain: { model: "m", seed: "18446744073709551615" },
      stages: [{ prompt: "a", frames: 97, source_image_b64: "aGk=" }],
    });
    expect(script?.chain.seed).toBe("18446744073709551615");
    expect(script?.stages[0]?.source_image_b64).toBe("aGk=");
  });

  it("returns null for a missing or model-less script", () => {
    expect(normalizeServerChainScript(null)).toBeNull();
    expect(normalizeServerChainScript({ chain: {}, stage: [] })).toBeNull();
  });
});

describe("countLeadingCompletedStages", () => {
  const stage = (idx: number, state: ChainJobStageDetail["state"]): ChainJobStageDetail => ({
    idx,
    state,
  });

  it("counts only the completed prefix, ordered by idx", () => {
    expect(
      countLeadingCompletedStages([
        stage(1, "completed"),
        stage(0, "completed"),
        stage(2, "failed"),
        stage(3, "completed"),
      ]),
    ).toBe(2);
    expect(countLeadingCompletedStages([stage(0, "pending")])).toBe(0);
    expect(countLeadingCompletedStages([])).toBe(0);
  });
});
