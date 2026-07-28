import { describe, expect, it, vi } from "vitest";
import type { ChainJobDetail, ChainJobEvent } from "@studio/lib/api/chainTypes";
import { watchChainJob, type SequenceWatchVisibility } from "./sequenceWatch";

const target = { baseUrl: "http://studio.local:7680", apiKey: "secret" };

function detail(overrides: Partial<ChainJobDetail> = {}): ChainJobDetail {
  return {
    id: "job-1",
    state: "running",
    model: "ltx2:q8",
    stage_count: 2,
    current_stage: 0,
    created_at_unix_ms: 1_000,
    updated_at_unix_ms: 1_000,
    error: null,
    stages: [
      { idx: 0, state: "running" },
      { idx: 1, state: "pending" },
    ],
    ...overrides,
  };
}

interface StreamCall {
  path: string;
  target: unknown;
  onEvent: (event: string, data: string) => void;
  onOpen?: (() => void) | undefined;
  onClose?: ((error: Error | null) => void) | undefined;
  signal: AbortSignal;
}

function harness() {
  const streams: StreamCall[] = [];
  const stream = vi.fn(
    (
      path: string,
      options: {
        target?: unknown;
        onEvent: (event: string, data: string) => void;
        onOpen?: () => void;
        onClose?: (error: Error | null) => void;
        signal: AbortSignal;
      },
    ) => {
      streams.push({
        path,
        target: options.target,
        onEvent: options.onEvent,
        onOpen: options.onOpen,
        onClose: options.onClose,
        signal: options.signal,
      });
      return Promise.resolve();
    },
  );
  const fetchDetail = vi.fn(() => Promise.resolve(detail()));
  const listeners: Array<() => void> = [];
  let hidden = false;
  const visibility: SequenceWatchVisibility = {
    get hidden() {
      return hidden;
    },
    addEventListener: (_type, listener) => listeners.push(listener),
    removeEventListener: (_type, listener) => {
      const idx = listeners.indexOf(listener);
      if (idx >= 0) listeners.splice(idx, 1);
    },
  };
  return {
    streams,
    stream,
    fetchDetail,
    visibility,
    setHidden(value: boolean) {
      hidden = value;
    },
    fireVisibility() {
      for (const listener of [...listeners]) listener();
    },
    get listenerCount() {
      return listeners.length;
    },
  };
}

function emit(call: StreamCall, event: ChainJobEvent): void {
  call.onEvent("message", JSON.stringify(event));
}

describe("watchChainJob", () => {
  it("streams the frozen route and reduces snapshot + delta frames", async () => {
    const h = harness();
    const updates: Array<number | null> = [];
    const handle = watchChainJob({
      target,
      jobId: "job 1",
      onUpdate: (live) => updates.push(live.activeStage),
      onError: () => {},
      stream: h.stream,
      fetchDetail: h.fetchDetail,
      visibility: h.visibility,
    });
    await Promise.resolve();
    await Promise.resolve();

    // The job id is percent-encoded and the frozen target is passed verbatim.
    expect(h.streams).toHaveLength(1);
    expect(h.streams[0]?.path).toBe("/api/chain-jobs/job%201/events");
    expect(h.streams[0]?.target).toEqual(target);
    expect(h.fetchDetail).toHaveBeenCalledWith(target, "job 1");

    emit(h.streams[0]!, { type: "stage_start", stage_idx: 1 });
    emit(h.streams[0]!, { type: "denoise_step", stage_idx: 1, step: 4, total: 8 });
    expect(handle.live.activeStage).toBe(1);
    expect(handle.live.progress[1]).toEqual({ step: 4, total: 8 });
    expect(updates.at(-1)).toBe(1);
    handle.stop();
  });

  it("falls back to a 5s poll when the stream fails, and stops polling when it reopens", async () => {
    vi.useFakeTimers();
    try {
      const h = harness();
      const handle = watchChainJob({
        target,
        jobId: "job-1",
        onUpdate: () => {},
        onError: () => {},
        pollIntervalMs: 5_000,
        stream: h.stream,
        fetchDetail: h.fetchDetail,
        visibility: h.visibility,
      });
      await vi.advanceTimersByTimeAsync(0);
      expect(handle.polling).toBe(false);
      const initialFetches = h.fetchDetail.mock.calls.length;

      // A dead stream re-syncs immediately over HTTP, then keeps polling.
      h.streams[0]?.onClose?.(new Error("network down"));
      expect(handle.polling).toBe(true);
      expect(h.fetchDetail.mock.calls.length).toBe(initialFetches + 1);
      expect(h.streams).toHaveLength(1);

      // Each poll tick also re-attempts the stream so a healed network wins.
      await vi.advanceTimersByTimeAsync(5_000);
      expect(h.fetchDetail.mock.calls.length).toBe(initialFetches + 2);
      expect(h.streams).toHaveLength(2);

      // A later stream attempt that opens cleanly retires the poll.
      h.streams.at(-1)?.onOpen?.();
      expect(handle.polling).toBe(false);
      await vi.advanceTimersByTimeAsync(5_000);
      expect(h.fetchDetail.mock.calls.length).toBe(initialFetches + 2);
      handle.stop();
    } finally {
      vi.useRealTimers();
    }
  });

  it("re-fetches a snapshot when iOS wakes the webview", async () => {
    const h = harness();
    const handle = watchChainJob({
      target,
      jobId: "job-1",
      onUpdate: () => {},
      onError: () => {},
      stream: h.stream,
      fetchDetail: h.fetchDetail,
      visibility: h.visibility,
    });
    await Promise.resolve();
    const before = h.fetchDetail.mock.calls.length;

    h.setHidden(true);
    h.fireVisibility();
    expect(h.fetchDetail.mock.calls.length).toBe(before);

    h.setHidden(false);
    h.fireVisibility();
    expect(h.fetchDetail.mock.calls.length).toBe(before + 1);
    handle.stop();
  });

  it("aborts the stream and drops its visibility listener on stop", async () => {
    const h = harness();
    const handle = watchChainJob({
      target,
      jobId: "job-1",
      onUpdate: () => {},
      onError: () => {},
      stream: h.stream,
      fetchDetail: h.fetchDetail,
      visibility: h.visibility,
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(h.listenerCount).toBe(1);
    expect(handle.live.activeStage).toBe(0);

    handle.stop();

    expect(h.streams[0]?.signal.aborted).toBe(true);
    expect(h.listenerCount).toBe(0);

    // Frames that land after stop must not mutate the retired view.
    emit(h.streams[0]!, { type: "stage_start", stage_idx: 1 });
    expect(handle.live.activeStage).toBe(0);
  });

  it("surfaces snapshot failures without tearing the watch down", async () => {
    const h = harness();
    h.fetchDetail.mockRejectedValue(new Error("gone"));
    const errors: unknown[] = [];
    const handle = watchChainJob({
      target,
      jobId: "job-1",
      onUpdate: () => {},
      onError: (error) => errors.push(error),
      stream: h.stream,
      fetchDetail: h.fetchDetail,
      visibility: h.visibility,
    });
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(errors).toHaveLength(1);
    expect(h.streams).toHaveLength(1);
    handle.stop();
  });
});
