import { afterEach, describe, expect, it, vi } from "vitest";
import {
  parseQueueListing,
  reduceQueuePlanEvent,
  setQueueDevicePin,
  type QueueListing,
} from "./queuePlan";

afterEach(() => vi.unstubAllGlobals());

describe("queue plan contract", () => {
  it("preserves legacy queue responses without a plan", () => {
    const listing = parseQueueListing({
      entries: [
        {
          id: "j1",
          model: "flux-dev:q8",
          state: "queued",
          started_at_unix_ms: 1,
          position: 0,
        },
      ],
    });
    expect(listing.entries[0]?.id).toBe("j1");
    expect(listing.plan).toBeNull();
  });

  it("reactively replaces only newer plan versions", () => {
    const current: QueueListing = {
      entries: [],
      plan: {
        plan_version: 4,
        state_version: 7,
        optimizer_state: "optimized",
        dirty_since_unix_ms: null,
        next_replan_at_unix_ms: null,
        work_items: [],
      },
    };
    expect(
      reduceQueuePlanEvent(current, {
        type: "queue_plan_changed",
        plan: { ...current.plan!, plan_version: 3 },
      }).plan?.plan_version,
    ).toBe(4);
    expect(
      reduceQueuePlanEvent(current, {
        type: "queue_plan_changed",
        plan: { ...current.plan!, plan_version: 5 },
      }).plan?.plan_version,
    ).toBe(5);
  });

  it("clears a stable pin on the explicit authenticated target", async () => {
    let captured: [RequestInfo | URL, RequestInit | undefined] | null = null;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        captured = [input, init];
        return Response.json({
          id: "job/1",
          model: "flux-dev:q8",
          state: "queued",
          started_at_unix_ms: 1,
          position: 0,
        });
      }),
    );
    await setQueueDevicePin(
      { baseUrl: "https://gpu.example", apiKey: "secret" },
      "job/1",
      null,
    );
    const [url, init] = captured!;
    expect(url).toBe("https://gpu.example/api/queue/job%2F1");
    expect((init?.headers as Headers).get("x-api-key")).toBe("secret");
    expect(JSON.parse(String(init?.body))).toEqual({
      hard_pinned_device_id: null,
    });
  });
});
