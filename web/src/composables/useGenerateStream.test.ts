import { describe, expect, it } from "vitest";
import {
  isPrebuiltChainRequest,
  resolveChainRequest,
} from "./useGenerateStream";
import type { ChainRequestWire, GenerateRequestWire } from "../types";
import type { ChainRoutingDecision } from "../lib/chainRouting";

function chainDecision(
  overrides: Partial<Extract<ChainRoutingDecision, { kind: "chain" }>> = {},
): Extract<ChainRoutingDecision, { kind: "chain" }> {
  return {
    kind: "chain",
    clipFrames: 97,
    motionTail: 4,
    stageCount: 3,
    ...overrides,
  };
}

function singleGen(
  overrides: Partial<GenerateRequestWire> = {},
): GenerateRequestWire {
  return {
    prompt: "a cat walking through autumn leaves",
    model: "ltx-2-19b-distilled:fp8",
    width: 1216,
    height: 704,
    steps: 8,
    guidance: 3.0,
    strength: 1.0,
    fps: 24,
    output_format: "mp4",
    frames: 241,
    ...overrides,
  };
}

function scriptChain(
  overrides: Partial<ChainRequestWire> = {},
): ChainRequestWire {
  return {
    model: "ltx-2-19b-distilled:fp8",
    stages: [
      { prompt: "cat in a garden", frames: 97, transition: "smooth" },
      { prompt: "cat on a rooftop", frames: 97, transition: "cut" },
      { prompt: "cat on the moon", frames: 97, transition: "fade" },
    ],
    motion_tail_frames: 4,
    width: 1216,
    height: 704,
    fps: 24,
    steps: 8,
    guidance: 3.0,
    strength: 1.0,
    output_format: "mp4",
    ...overrides,
  };
}

describe("isPrebuiltChainRequest", () => {
  it("returns true for a ChainRequestWire with populated stages", () => {
    expect(isPrebuiltChainRequest(scriptChain())).toBe(true);
  });

  it("returns false for a single-clip GenerateRequestWire", () => {
    expect(isPrebuiltChainRequest(singleGen())).toBe(false);
  });

  it("returns false when `stages` is an empty array", () => {
    // An empty stages[] is ambiguous — treat it as 'not pre-built' so the
    // auto-expand helper takes over. (Server would 422 either way, but this
    // keeps the router predictable.)
    const req = scriptChain({ stages: [] });
    expect(isPrebuiltChainRequest(req)).toBe(false);
  });
});

describe("resolveChainRequest", () => {
  it("passes a script payload through verbatim (regression: prior code nuked stages)", () => {
    // Repro for the HTTP 422
    //   "chain request needs either stages[] or prompt + total_frames"
    // that fired whenever Script mode submitted a ChainRequestWire: submit()
    // used to unconditionally re-project through buildChainRequest, which
    // reads GenerateRequestWire fields that don't exist on a script payload
    // and dropped `stages` entirely. The outgoing body ended up with no
    // stages and no auto-expand form → server 422.
    const req = scriptChain();
    const resolved = resolveChainRequest(req, chainDecision());
    expect(resolved).toBe(req);
    expect(resolved.stages).toHaveLength(3);
    expect(resolved.stages?.[0]?.prompt).toBe("cat in a garden");
    expect(resolved.stages?.[1]?.transition).toBe("cut");
    expect(resolved.stages?.[2]?.transition).toBe("fade");
    // Script mode must not smuggle in auto-expand fields either; those are
    // mutually exclusive with stages[] and the server's normalise() would
    // prefer stages[] regardless, but having them unset keeps the wire body
    // unambiguous.
    expect(resolved.prompt).toBeUndefined();
    expect(resolved.total_frames).toBeUndefined();
  });

  it("projects a single-prompt request into the auto-expand form", () => {
    const req = singleGen({ prompt: "a single prompt", frames: 241 });
    const resolved = resolveChainRequest(req, chainDecision());
    expect(resolved.stages).toBeUndefined();
    expect(resolved.prompt).toBe("a single prompt");
    expect(resolved.total_frames).toBe(241);
    expect(resolved.clip_frames).toBe(97);
    expect(resolved.motion_tail_frames).toBe(4);
  });

  it("falls back to auto-expand when stages[] is empty", () => {
    // Defensive: a caller shouldn't send empty stages, but if they do we
    // don't want to pass that through (the server would 422 on empty
    // stages). The resolver treats this as 'not pre-built' and re-projects.
    const req = scriptChain({ stages: [] });
    const resolved = resolveChainRequest(
      req as unknown as GenerateRequestWire,
      chainDecision(),
    );
    // buildChainRequest reads `prompt`/`frames` off the input — those are
    // absent on an empty-stages chain request, so they come through as
    // undefined. That's the expected downstream failure mode (422 from the
    // server), not a silent success. The assertion here only verifies that
    // we took the non-passthrough branch.
    expect(resolved.stages).toBeUndefined();
  });
});
