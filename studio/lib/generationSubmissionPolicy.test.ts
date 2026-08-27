import { describe, expect, it } from "vitest";
import {
  generationHostSubmissionPolicy,
  truthfulGenerationPhase,
  type GenerationSubmissionHost,
} from "./generationSubmissionPolicy";

function canonicalHost(
  hostId: string,
  overrides: Partial<GenerationSubmissionHost> = {},
): GenerationSubmissionHost {
  return {
    hostId,
    queue: { heterogeneous_batch_max_outputs: 64 },
    durableMedia: {
      protocol_version: 2,
      encrypted_at_rest: true,
      generate_request_media: true,
      identity: true,
      h3_references: true,
      private_h3: true,
    },
    ...overrides,
  };
}

describe("generation submission policy", () => {
  it("sends every pinned model family directly to canonical durable admission", () => {
    for (const request of [
      { model: "flux-dev", prompt: "still" },
      { model: "ltx-2-19b", source_image: "frame" },
      { model: "minimax-h3-fl2va", source_image: "frame" },
      { model: "minimax-h3-ref2va", references: [] },
    ]) {
      expect(
        generationHostSubmissionPolicy(
          { kind: "pinned", hostId: "hal" },
          canonicalHost("hal"),
          request,
        ),
      ).toEqual({
        routing: "none",
        admission: "canonical_durable",
        refusal: null,
      });
    }
  });

  it("fans automatic routing through cache-only probes", () => {
    for (const target of [{ kind: "auto" }, { kind: "capable" }] as const) {
      expect(
        generationHostSubmissionPolicy(target, canonicalHost("hal"), {
          model: "wan22",
          prompt: "clip",
        }),
      ).toMatchObject({
        routing: "telemetry_only",
        admission: "canonical_durable",
      });
    }
  });

  it("refuses a request trait by name instead of routing it elsewhere", () => {
    const cases: Array<[object, Partial<GenerationSubmissionHost>, string]> = [
      [
        { model: "flux-dev", hdr_exr_dir: "/private/hdr" },
        {},
        "an HDR EXR output directory cannot be queued",
      ],
      [
        { model: "flux-dev", source_image: "frame", lora: "local.safetensors" },
        {},
        "a LoRA cannot be combined with source media in a queued print",
      ],
      [
        {
          model: "flux-dev",
          source_image: "frame",
          loras: [{ path: "local.safetensors" }],
        },
        {},
        "a LoRA cannot be combined with source media in a queued print",
      ],
      [
        { model: "minimax-h3-ref2va", references: [{ image: "frame" }] },
        {
          durableMedia: {
            ...canonicalHost("hal").durableMedia!,
            h3_references: false,
          },
        },
        "this machine cannot store reference media durably",
      ],
      [
        { model: "flux-dev", id_image: "face" },
        {
          durableMedia: {
            ...canonicalHost("hal").durableMedia!,
            identity: false,
          },
        },
        "this machine cannot store identity photos durably",
      ],
      [
        { family: "minimax-h3", model: "hf:opaque", prompt: "still" },
        {
          durableMedia: {
            ...canonicalHost("hal").durableMedia!,
            private_h3: false,
          },
        },
        "this machine cannot store MiniMax H3 request media durably",
      ],
      [
        { model: "flux-dev", source_image: "frame" },
        { durableMedia: null },
        "this machine cannot store request media durably",
      ],
      [
        { model: "flux-dev", prompt: "still" },
        { queue: null },
        "this machine does not advertise the durable generation queue",
      ],
    ];
    for (const [request, overrides, refusal] of cases) {
      expect(
        generationHostSubmissionPolicy(
          { kind: "pinned", hostId: "hal" },
          canonicalHost("hal", overrides),
          request,
        ),
      ).toEqual({ routing: "none", admission: "refused", refusal });
    }
  });

  it("keeps sequences on the chain-job route with its placement preview", () => {
    expect(
      generationHostSubmissionPolicy(
        { kind: "pinned", hostId: "hal" },
        canonicalHost("hal"),
        { model: "ltx-2", stages: [] },
        "sequence",
      ),
    ).toMatchObject({
      routing: "placement_preview",
      admission: "refused",
    });
  });

  it("presents the authoritative durable child lifecycle", () => {
    expect(truthfulGenerationPhase({ state: "accepted" })).toBe("accepted");
    expect(truthfulGenerationPhase({ state: "held" })).toBe("held");
    expect(truthfulGenerationPhase({ state: "cancelling" })).toBe("cancelling");
    expect(truthfulGenerationPhase({ phase: "running" })).toBe("running");
    expect(truthfulGenerationPhase({ phase: "complete" })).toBe("terminal");
  });
});
