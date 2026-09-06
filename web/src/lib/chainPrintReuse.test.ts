import { describe, expect, it } from "vitest";
import { oneShotPromptForPrint } from "./chainPrintReuse";
import type { OutputMetadata } from "../types";

function metadata(patch: Partial<OutputMetadata> = {}): OutputMetadata {
  return {
    prompt: "a lighthouse at dusk",
    model: "flux-dev:fp16",
    seed: 4242,
    steps: 20,
    guidance: 3.5,
    width: 1024,
    height: 768,
    version: "test",
    ...patch,
  } as OutputMetadata;
}

describe("oneShotPromptForPrint", () => {
  it("keeps an ordinary print's own prompt", () => {
    expect(oneShotPromptForPrint(metadata())).toBe("a lighthouse at dusk");
  });

  it("restores the FIRST clip's prompt, never the newline-joined blob", () => {
    const stitched = metadata({
      prompt: "the harbour at dawn\nthe harbour at noon\nthe harbour at dusk",
      chain: {
        stage_count: 3,
        motion_tail_frames: 8,
        stages: [
          { prompt: "the harbour at dawn", frames: 97, transition: "smooth" },
          { prompt: "the harbour at noon", frames: 97, transition: "smooth" },
          { prompt: "the harbour at dusk", frames: 97, transition: "smooth" },
        ],
      },
    });

    expect(oneShotPromptForPrint(stitched)).toBe("the harbour at dawn");
    expect(oneShotPromptForPrint(stitched)).not.toContain("\n");
  });

  it("falls back to the print's prompt when the provenance carries no usable clip", () => {
    expect(
      oneShotPromptForPrint(
        metadata({
          chain: { stage_count: 0, motion_tail_frames: 8, stages: [] },
        }),
      ),
    ).toBe("a lighthouse at dusk");
    expect(
      oneShotPromptForPrint(
        metadata({
          chain: {
            stage_count: 1,
            motion_tail_frames: 8,
            stages: [{ prompt: "   ", frames: 97, transition: "smooth" }],
          },
        }),
      ),
    ).toBe("a lighthouse at dusk");
  });
});
