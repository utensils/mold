import { describe, expect, it } from "vitest";
import {
  planGenerationSubmission,
  truthfulGenerationPhase,
  type GenerationSubmissionHost,
} from "./generationSubmissionPolicy";

const legacyQueue = {
  heterogeneous_batch: true,
  heterogeneous_batch_max_outputs: 64,
  durable_batch_outcomes: true,
};

function canonicalHost(
  hostId: string,
  overrides: Partial<GenerationSubmissionHost> = {},
): GenerationSubmissionHost {
  return {
    hostId,
    queue: { ...legacyQueue, admission_protocol_version: 2 },
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
        planGenerationSubmission({
          target: { kind: "pinned", hostId: "hal" },
          hosts: [canonicalHost("hal")],
          request,
        }).hosts[0],
      ).toMatchObject({
        compatibility: "canonical_v2",
        routing: "none",
        admission: "canonical_durable",
      });
    }
  });

  it("fans automatic routing through cache-only probes on v2 hosts", () => {
    const plan = planGenerationSubmission({
      target: { kind: "auto" },
      hosts: [canonicalHost("hal"), canonicalHost("studio")],
      request: { model: "wan22", prompt: "clip" },
    });
    expect(plan.hosts.map(({ hostId, routing }) => [hostId, routing])).toEqual([
      ["hal", "telemetry_only"],
      ["studio", "telemetry_only"],
    ]);
  });

  it("keeps request traits outside protocol v2 on the attached transport", () => {
    const cases = [
      { model: "flux-dev", hdr_exr_dir: "/private/hdr" },
      { model: "flux-dev", source_image: "frame", lora: "local.safetensors" },
      {
        model: "flux-dev",
        source_image: "frame",
        loras: [{ path: "local.safetensors" }],
      },
      { model: "minimax-h3-ref2va", references: [{ image: "frame" }] },
    ];
    for (const request of cases) {
      const host = canonicalHost("hal", {
        durableMedia: {
          ...canonicalHost("hal").durableMedia!,
          h3_references: false,
        },
      });
      expect(
        planGenerationSubmission({
          target: { kind: "pinned", hostId: "hal" },
          hosts: [host],
          request,
        }).hosts[0],
      ).toMatchObject({
        compatibility: "legacy",
        admission: "legacy_attached",
      });
    }
  });

  it("keeps old and incompletely capable hosts on explicit legacy paths", () => {
    const plan = planGenerationSubmission({
      target: { kind: "auto" },
      hosts: [
        { hostId: "old", queue: legacyQueue },
        canonicalHost("no-media-v2", {
          durableMedia: {
            protocol_version: 1,
            encrypted_at_rest: true,
            generate_request_media: true,
            identity: true,
            h3_references: false,
            private_h3: false,
          },
        }),
      ],
      request: { model: "opaque", source_image: "frame" },
    });
    expect(plan.hosts).toMatchObject([
      {
        hostId: "old",
        compatibility: "legacy",
        routing: "legacy_placement",
        admission: "legacy_attached",
      },
      {
        hostId: "no-media-v2",
        compatibility: "legacy",
        routing: "legacy_placement",
        admission: "legacy_durable",
      },
    ]);
  });

  it("keeps an opaque H3 family attached on a legacy host", () => {
    expect(
      planGenerationSubmission({
        target: { kind: "pinned", hostId: "old" },
        hosts: [
          {
            hostId: "old",
            queue: legacyQueue,
            durableMedia: {
              protocol_version: 1,
              encrypted_at_rest: true,
              generate_request_media: true,
              identity: true,
              h3_references: false,
              private_h3: false,
            },
          },
        ],
        request: {
          family: "minimax-h3",
          model: "hf:opaque-h3-checkpoint",
          source_image: "frame",
        },
      }).hosts[0],
    ).toMatchObject({
      compatibility: "legacy",
      routing: "legacy_placement",
      admission: "legacy_attached",
    });
  });

  it("filters a pinned plan to its frozen host", () => {
    const plan = planGenerationSubmission({
      target: { kind: "pinned", hostId: "studio" },
      hosts: [canonicalHost("hal"), canonicalHost("studio")],
      request: { model: "qwen-image" },
    });
    expect(plan.hosts.map((host) => host.hostId)).toEqual(["studio"]);
  });

  it("keeps sequences on their existing typed admission contract", () => {
    expect(
      planGenerationSubmission({
        target: { kind: "pinned", hostId: "hal" },
        hosts: [canonicalHost("hal")],
        request: { model: "ltx-2", stages: [] },
        outputKind: "sequence",
      }).hosts[0],
    ).toMatchObject({
      compatibility: "legacy",
      routing: "legacy_placement",
    });
  });

  it("presents the authoritative durable child lifecycle", () => {
    expect(truthfulGenerationPhase({ state: "accepted" })).toBe("accepted");
    expect(truthfulGenerationPhase({ state: "held" })).toBe("held");
    expect(truthfulGenerationPhase({ phase: "running" })).toBe("running");
    expect(truthfulGenerationPhase({ phase: "complete" })).toBe("terminal");
  });
});
