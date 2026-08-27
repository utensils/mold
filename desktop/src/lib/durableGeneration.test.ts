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

  it("admits every request trait the durable protocol carries", () => {
    // The server takes media, LoRAs, HDR directories, identity photos, and
    // H3's ordered references on this route. A client-side per-trait fence
    // could only refuse work the server would have accepted.
    expect(generationRefusalReason(queue, durableMedia)).toBeNull();
  });

  it("names a machine that does not speak the contract", () => {
    expect(generationRefusalReason({ heterogeneous_batch_max_outputs: 0 }, durableMedia)).toBe(
      "this machine does not advertise the durable generation queue",
    );
    // durable_media is the server's per-request refusal, not a client fence:
    // a host whose media store is degraded still admits every media-free print.
    expect(generationRefusalReason(queue, null)).toBeNull();
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
