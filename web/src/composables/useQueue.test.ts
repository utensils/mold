import { afterEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import { useQueue } from "./useQueue";

vi.mock("../api", () => ({
  fetchQueue: vi.fn(),
}));

import { fetchQueue } from "../api";

afterEach(() => {
  vi.useRealTimers();
  vi.mocked(fetchQueue).mockReset();
});

describe("useQueue", () => {
  it("loads /api/queue immediately and exposes entries as reusable state", async () => {
    vi.mocked(fetchQueue).mockResolvedValue({
      entries: [
        {
          id: "srv-1",
          model: "flux-dev:fp16",
          state: "running",
          started_at_unix_ms: 0,
          position: 0,
          gpu: 0,
        },
      ],
    });

    const q = useQueue({ pollMs: 60_000 });
    await nextTick();
    await Promise.resolve();

    expect(fetchQueue).toHaveBeenCalledOnce();
    expect(q.entries.value).toHaveLength(1);
    expect(q.entries.value[0].gpu).toBe(0);
    expect(q.error.value).toBeNull();
    q.stop();
  });

  it("polls until stopped and keeps the last successful listing on errors", async () => {
    vi.useFakeTimers();
    vi.mocked(fetchQueue)
      .mockResolvedValueOnce({
        entries: [
          {
            id: "a",
            model: "m",
            state: "queued",
            started_at_unix_ms: 0,
            position: 0,
          },
        ],
      })
      .mockRejectedValue(new Error("offline"));

    const q = useQueue({ pollMs: 1_000 });
    await vi.runOnlyPendingTimersAsync();
    await Promise.resolve();
    expect(q.entries.value.map((e) => e.id)).toEqual(["a"]);

    await vi.advanceTimersByTimeAsync(1_001);
    expect(q.entries.value.map((e) => e.id)).toEqual(["a"]);
    expect(q.error.value?.message).toContain("offline");

    const callsBeforeStop = vi.mocked(fetchQueue).mock.calls.length;
    q.stop();
    await vi.advanceTimersByTimeAsync(10_000);
    expect(fetchQueue).toHaveBeenCalledTimes(callsBeforeStop);
  });
});
