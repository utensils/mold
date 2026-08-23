import { afterEach, describe, expect, it, vi } from "vitest";
import { IncompatibleHostError } from "./client";
import {
  cancelQueueJob,
  listQueue,
  mergeQueueEntries,
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
    expect(listing.page).toBeUndefined();
    expect(listing.live_only_entries).toBeUndefined();
  });

  it("parses an additive page and live-only entries", () => {
    const listing = parseQueueListing({
      entries: [
        {
          id: "durable-1",
          model: "flux-dev:q8",
          state: "queued",
          started_at_unix_ms: 1,
          position: 4,
        },
      ],
      live_only_entries: [
        {
          id: "live-1",
          model: "minimax-h3:nvfp4",
          state: "running",
          started_at_unix_ms: 2,
          position: 0,
        },
      ],
      page: {
        limit: 8,
        offset: 16,
        returned: 1,
        next_cursor: "opaque-cursor",
      },
    });

    expect(listing.page).toEqual({
      limit: 8,
      offset: 16,
      returned: 1,
      next_cursor: "opaque-cursor",
    });
    expect(listing.live_only_entries?.map(({ id }) => id)).toEqual(["live-1"]);
  });

  it.each([
    [null, "page.limit"],
    [{ limit: 0, offset: 0, returned: 0 }, "page.limit"],
    [{ limit: 2.5, offset: 0, returned: 0 }, "page.limit"],
    [{ limit: 2, offset: -1, returned: 0 }, "page.offset"],
    [
      { limit: 2, offset: Number.POSITIVE_INFINITY, returned: 0 },
      "page.offset",
    ],
    [{ limit: 2, offset: 0, returned: 3 }, "page.returned"],
    [{ limit: 2, offset: 0, returned: 1, next_cursor: "" }, "page.next_cursor"],
  ])("rejects an invalid additive queue page %#", (page, field) => {
    const parse = () => parseQueueListing({ entries: [], page });
    expect(parse).toThrowError(IncompatibleHostError);
    expect(parse).toThrow(field);
  });

  it("rejects malformed additive live-only rows as an incompatible host", () => {
    expect(() =>
      parseQueueListing({ entries: [], live_only_entries: [{ id: "bad" }] }),
    ).toThrowError(IncompatibleHostError);
  });

  it("merges durable order with stable, cross-page live-only deduplication", () => {
    const entry = (id: string): import("./queuePlan").QueueEntry => ({
      id,
      model: "flux-dev:q8",
      state: "queued",
      started_at_unix_ms: 1,
      position: 0,
    });
    const durable = [entry("durable-2"), entry("durable-1")];
    const repeatedLiveOnlyRows = [
      entry("live-1"),
      entry("durable-1"),
      entry("live-1"),
      entry("live-2"),
    ];

    expect(
      mergeQueueEntries(durable, repeatedLiveOnlyRows).map(({ id }) => id),
    ).toEqual(["durable-2", "durable-1", "live-1", "live-2"]);
  });

  it("keeps the legacy listQueue URL and authenticated target unchanged", async () => {
    const fetchMock = vi.fn().mockResolvedValue(Response.json({ entries: [] }));
    vi.stubGlobal("fetch", fetchMock);

    await listQueue({ baseUrl: "https://gpu.example", apiKey: "secret" });

    expect(fetchMock).toHaveBeenCalledWith(
      "https://gpu.example/api/queue",
      expect.objectContaining({
        headers: expect.any(Headers),
      }),
    );
    const headers = fetchMock.mock.calls[0]?.[1]?.headers as Headers;
    expect(headers.get("x-api-key")).toBe("secret");
  });

  it("encodes a caller-supplied cursor and rejects invalid limits before fetch", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      Response.json({
        entries: [],
        page: { limit: 7, offset: 0, returned: 0 },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await listQueue(
      { baseUrl: "https://gpu.example", apiKey: "secret" },
      { limit: 7, cursor: "opaque/+ token=" },
    );
    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "https://gpu.example/api/queue?limit=7&cursor=opaque%2F%2B+token%3D",
    );

    for (const limit of [0, -1, 1.5, Number.POSITIVE_INFINITY]) {
      await expect(
        listQueue(
          { baseUrl: "https://gpu.example", apiKey: "secret" },
          { limit },
        ),
      ).rejects.toThrow("positive integer");
    }
    expect(fetchMock).toHaveBeenCalledTimes(1);
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
