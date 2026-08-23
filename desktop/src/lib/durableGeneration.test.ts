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
  it("accepts ordinary singleton/batch children but retains media and H3 on legacy transport", () => {
    expect(requestIsEligibleForDurableGeneration(request())).toBe(true);
    expect(requestIsEligibleForDurableGeneration(request({ source_image: "bytes" }))).toBe(false);
    expect(requestIsEligibleForDurableGeneration(request({ id_image: "face" }))).toBe(false);
    expect(
      requestIsEligibleForDurableGeneration(
        request({ model: "minimax-h3-ref2va", references: [] }),
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
