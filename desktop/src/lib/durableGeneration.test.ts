import { describe, expect, it } from "vitest";
import { createGenerationBatchTracker } from "@studio/lib/generationLifecycle";
import {
  DURABLE_GENERATION_STORAGE_KEY,
  durableChildSummary,
  loadDurableGenerationRecovery,
  parseEventAuthority,
  parseEventResync,
  requestIsEligibleForDurableGeneration,
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
  const queue = { heterogeneous_batch: true, durable_batch_outcomes: true };
  const durableMedia = {
    protocol_version: 1,
    encrypted_at_rest: true,
    generate_request_media: true,
    identity: true,
    h3_references: false,
    private_h3: false,
  };

  it("accepts media only through the exact encrypted host capability", () => {
    expect(requestIsEligibleForDurableGeneration(request(), queue, undefined)).toBe(true);
    expect(
      requestIsEligibleForDurableGeneration(
        request({ source_image: "bytes" }),
        queue,
        durableMedia,
      ),
    ).toBe(true);
    expect(
      requestIsEligibleForDurableGeneration(request({ id_image: "face" }), queue, {
        ...durableMedia,
        identity: false,
      }),
    ).toBe(false);
    expect(
      requestIsEligibleForDurableGeneration(
        request({ source_image: "bytes", loras: [] }),
        queue,
        durableMedia,
      ),
    ).toBe(false);
    expect(
      requestIsEligibleForDurableGeneration(
        request({ model: "minimax-h3-ref2va", references: [] }),
        queue,
        durableMedia,
      ),
    ).toBe(false);
    expect(
      requestIsEligibleForDurableGeneration(
        request({ model: "hf:opaque-h3-checkpoint", source_image: "bytes" }),
        queue,
        durableMedia,
        "minimax-h3",
      ),
    ).toBe(false);
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
