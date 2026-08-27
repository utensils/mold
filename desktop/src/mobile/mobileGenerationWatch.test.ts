import { describe, expect, it, vi } from "vitest";
import { watchMobileGenerationHost } from "./mobileGenerationWatch";

function harness() {
  let callbacks:
    | {
        onOpen?: () => void;
        onEvent: (event: string, data: string) => void;
        onClose?: (error: Error | null) => void;
        signal: AbortSignal;
      }
    | undefined;
  const reasons: string[] = [];
  const scopes: Array<string[] | undefined> = [];
  const gaps: Array<{ instanceId: string; missedEvents?: number }> = [];
  const watch = watchMobileGenerationHost({
    target: { baseUrl: "https://host", apiKey: "native-secret" },
    expectedInstanceId: "instance-1",
    onReconcile: (reason, jobIds) => {
      reasons.push(reason);
      scopes.push(jobIds ? [...jobIds].sort() : undefined);
    },
    onGap: (gap) => gaps.push(gap),
    stream: vi.fn(async (_path, options) => {
      callbacks = options;
    }),
  });
  const flush = async () => await Promise.resolve();
  return { watch, reasons, scopes, gaps, callbacks: () => callbacks!, flush };
}

describe("mobile durable generation host watch", () => {
  it("uses one host stream and coalesces event hints into one bulk reconcile", async () => {
    const h = harness();
    h.callbacks().onOpen?.();
    h.callbacks().onEvent("event", JSON.stringify({ type: "job_queued", id: "a" }));
    h.callbacks().onEvent("event", JSON.stringify({ type: "job_started", id: "a" }));
    await h.flush();
    expect(h.reasons).toEqual(["open"]);

    // Neither `gallery_added` nor `job_ended` is a lifecycle authority: the
    // commit hint that follows every settlement is the only read trigger.
    h.callbacks().onEvent("event", JSON.stringify({ type: "gallery_added" }));
    h.callbacks().onEvent("event", JSON.stringify({ type: "job_ended", id: "a" }));
    await h.flush();
    expect(h.reasons).toEqual(["open"]);

    h.callbacks().onEvent("event", JSON.stringify({ type: "job_state_committed", id: "a" }));
    await h.flush();
    expect(h.reasons).toEqual(["open", "event"]);
    expect(h.scopes).toEqual([undefined, ["a"]]);

    h.callbacks().onEvent("event", JSON.stringify({ type: "job_state_committed", id: "b" }));
    h.callbacks().onEvent("event", JSON.stringify({ type: "job_state_committed", id: "c" }));
    await h.flush();
    expect(h.scopes.at(-1)).toEqual(["b", "c"]);

    h.callbacks().onEvent("event", JSON.stringify({ type: "generation_states_committed" }));
    await h.flush();
    expect(h.scopes.at(-1)).toBeUndefined();
  });

  it("reconciles event gaps, malformed frames, close, and wake", async () => {
    const h = harness();
    h.callbacks().onEvent(
      "resync_required",
      JSON.stringify({ instance_id: "instance-1", missed_events: 7 }),
    );
    await h.flush();
    expect(h.gaps).toEqual([{ instanceId: "instance-1", missedEvents: 7 }]);
    expect(h.reasons).toEqual(["event_gap"]);

    h.callbacks().onEvent("event", "not-json");
    await h.flush();
    h.callbacks().onClose?.(null);
    await h.flush();
    h.watch.wake();
    await h.flush();
    expect(h.reasons).toEqual(["event_gap", "malformed", "close", "wake"]);
  });

  it("fences an authority frame from a replacement instance", async () => {
    const h = harness();
    h.callbacks().onEvent("authority", JSON.stringify({ instance_id: "replacement" }));
    await h.flush();
    expect(h.gaps).toEqual([{ instanceId: "replacement" }]);
    expect(h.reasons).toEqual(["instance_mismatch"]);
  });

  it("aborts without cancelling server-owned generation work", () => {
    const h = harness();
    h.watch.stop();
    expect(h.callbacks().signal.aborted).toBe(true);
    // The watcher owns only GET /api/events; it has no cancellation callback.
    expect(h.reasons).toEqual([]);
  });
});
