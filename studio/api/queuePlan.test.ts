import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cancelQueueJob,
  parseQueueListing,
  predictedCompletionUnixMs,
  reduceQueuePlanEvent,
  setQueueDevicePin,
  type QueuePlan,
  type QueueListing,
} from "./queuePlan";

afterEach(() => vi.unstubAllGlobals());

describe("queue plan contract", () => {
  const plan = (finish: number | null): QueuePlan => ({
    plan_version: 1,
    state_version: 1,
    optimizer_state: "optimized",
    dirty_since_unix_ms: null,
    next_replan_at_unix_ms: null,
    work_items: [
      {
        work_id: "job-1",
        parent_id: "job-1",
        work_kind: "generation",
        priority_class: "user",
        queue_rank: 0,
        bypass_count: 0,
        estimated_finish_unix_ms: finish,
        estimate_confidence: "low",
      },
    ],
  });

  it("keeps a plan with no finite finish estimate unknown", () => {
    expect(predictedCompletionUnixMs(plan(null), 50_000)).toBeNull();
    expect(
      predictedCompletionUnixMs({ ...plan(null), work_items: [] }, 50_000),
    ).toBeNull();
    expect(predictedCompletionUnixMs(plan(Number.NaN), 50_000)).toBeNull();
  });

  it("clamps a known finish to now without inventing an unknown estimate", () => {
    expect(predictedCompletionUnixMs(plan(40_000), 50_000)).toBe(50_000);
    expect(predictedCompletionUnixMs(plan(60_000), 50_000)).toBe(60_000);
  });

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

  it("preserves typed host lanes without treating them as device identities", () => {
    const listing = parseQueueListing({
      entries: [],
      plan: {
        plan_version: 1,
        state_version: 1,
        optimizer_state: "optimized",
        dirty_since_unix_ms: null,
        next_replan_at_unix_ms: null,
        work_items: [
          {
            work_id: "cpu-work",
            parent_id: "parent",
            work_kind: "prompt_expansion",
            priority_class: "user",
            queue_rank: 0,
            bypass_count: 0,
            planned_device_id: null,
            planned_lane_kind: "host_utility",
            lane_order: 0,
            estimate_confidence: "low",
          },
        ],
      },
    });

    expect(listing.plan?.work_items[0]).toMatchObject({
      planned_device_id: null,
      planned_lane_kind: "host_utility",
    });
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

  it("cancels a queued job on the explicit authenticated target", async () => {
    let captured: [RequestInfo | URL, RequestInit | undefined] | null = null;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        captured = [input, init];
        return new Response(null, { status: 204 });
      }),
    );

    await cancelQueueJob(
      { baseUrl: "https://gpu.example", apiKey: "secret" },
      "job/1",
    );

    const [url, init] = captured!;
    expect(url).toBe("https://gpu.example/api/queue/job%2F1");
    expect(init?.method).toBe("DELETE");
    expect((init?.headers as Headers).get("x-api-key")).toBe("secret");
  });
});
