import { describe, expect, it } from "vitest";
import { createGenerationBatchTracker } from "@studio/lib/generationLifecycle";
import {
  DURABLE_GENERATION_STORAGE_KEY,
  durableChildSummary,
  loadDurableGenerationRecovery,
  parseEventAuthority,
  parseEventResync,
  generationRefusalReason,
  saveDurableGenerationRecovery,
} from "./durableGeneration";
import type { GenerateRequest } from "./api/types";

const request = (overrides: Partial<GenerateRequest> = {}): GenerateRequest => ({
  prompt: "private prompt is never persisted by the authority record",
  model: "flux2-klein",
  width: 512,
  height: 512,
  steps: 4,
  ...overrides,
});

describe("desktop durable generation recovery", () => {
  const queue = { heterogeneous_batch_max_outputs: 64 };
  const durableMedia = {
    protocol_version: 2,
    encrypted_at_rest: true,
    generate_request_media: true,
    identity: true,
    h3_references: false,
    private_h3: false,
  };

  it("admits a media-free print and media the host actually covers", () => {
    expect(generationRefusalReason(request(), queue, undefined)).toBeNull();
    expect(
      generationRefusalReason(request({ source_image: "bytes" }), queue, durableMedia),
    ).toBeNull();
  });

  it("names the refusal instead of routing the print to a second path", () => {
    const cases: Array<{
      request: GenerateRequest;
      queue?: typeof queue;
      media?: typeof durableMedia;
      family?: string;
      reason: string;
    }> = [
      {
        request: request(),
        queue: { heterogeneous_batch_max_outputs: 0 },
        reason: "this machine does not advertise the durable generation queue",
      },
      {
        request: request({ source_image: "bytes" }),
        reason: "this machine cannot store request media durably",
      },
      {
        request: request({ id_image: "face" }),
        media: { ...durableMedia, identity: false },
        reason: "this machine cannot store identity photos durably",
      },
      {
        request: request({ source_image: "bytes", loras: [] }),
        media: durableMedia,
        reason: "a LoRA cannot be combined with source media in a queued print",
      },
      {
        request: request({ model: "minimax-h3-ref2va", references: [] }),
        media: { ...durableMedia, private_h3: true, h3_references: false },
        reason: "this machine cannot store reference media durably",
      },
      {
        request: request({ model: "hf:opaque-h3-checkpoint", source_image: "bytes" }),
        media: durableMedia,
        family: "minimax-h3",
        reason: "this machine cannot store MiniMax H3 request media durably",
      },
      {
        request: { ...request(), hdr_exr_dir: "/hdr" } as GenerateRequest,
        media: durableMedia,
        reason: "an HDR EXR output directory cannot be queued",
      },
    ];
    for (const candidate of cases) {
      expect(
        generationRefusalReason(
          candidate.request,
          candidate.queue ?? queue,
          candidate.media,
          candidate.family,
        ),
      ).toBe(candidate.reason);
    }
  });

  it("admits H3 through the machine's own private durable contract", () => {
    expect(
      generationRefusalReason(
        request({ model: "hf:opaque-h3-checkpoint", source_image: "bytes" }),
        queue,
        { ...durableMedia, private_h3: true },
        "minimax-h3",
      ),
    ).toBeNull();
  });

  it("persists recovery authority without API secrets, prompts, or media", () => {
    let written = "";
    saveDurableGenerationRecovery(
      [
        {
          tracker: createGenerationBatchTracker({
            hostId: "render-host",
            expectedInstanceId: "instance-1",
            clientBatchId: "client-1",
            submittedAtMs: 10,
          }),
          hostLabel: "Render host",
          hostKind: "remote",
          mirrorRemoteOutput: true,
          children: [durableChildSummary(request({ seed: 7 }), 1, 42)],
          cancelRequestedChildIndexes: [],
          effectReceipts: [],
        },
      ],
      { setItem: (key, value) => void (written = `${key}:${value}`) },
    );

    expect(written.startsWith(`${DURABLE_GENERATION_STORAGE_KEY}:`)).toBe(true);
    expect(written).not.toContain("private prompt");
    expect(written).not.toContain("api-key");
    expect(written).not.toContain("source_image");
    const payload = written.slice(written.indexOf(":") + 1);
    expect(
      loadDurableGenerationRecovery({ getItem: () => payload })[0]?.tracker.clientBatchId,
    ).toBe("client-1");
  });

  it.each(["QuotaExceededError", "SecurityError"])(
    "degrades recovery without throwing when storage raises %s",
    (name) => {
      const error = Object.assign(new Error("storage unavailable"), { name });
      expect(
        saveDurableGenerationRecovery([], {
          setItem: () => {
            throw error;
          },
        }),
      ).toBe(false);
    },
  );

  it("parses the authority and explicit event-gap contract fail-closed", () => {
    expect(parseEventAuthority('{"instance_id":"instance-1"}')).toEqual({
      instanceId: "instance-1",
    });
    expect(parseEventResync('{"instance_id":"instance-1","missed_events":3}')).toEqual({
      instanceId: "instance-1",
      missedEvents: 3,
    });
    expect(parseEventAuthority("{}")).toBeNull();
    expect(parseEventResync('{"instance_id":"instance-1","missed_events":-1}')).toBeNull();
  });
});
