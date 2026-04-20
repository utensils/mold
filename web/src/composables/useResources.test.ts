import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import { useResources } from "./useResources";
import type { ResourceSnapshot } from "../types";

function snap(overrides: Partial<ResourceSnapshot> = {}): ResourceSnapshot {
  return {
    hostname: "unit",
    timestamp: 1,
    gpus: [],
    system_ram: { total: 100, used: 0, used_by_mold: 0, used_by_other: 0 },
    ...overrides,
  };
}

/**
 * Minimal EventSource stub. Vitest's happy-dom doesn't ship a functional
 * EventSource; we replace it with a class that lets each test drive
 * `message` / `error` / `open` events deterministically.
 */
class MockEventSource implements Partial<EventSource> {
  static instances: MockEventSource[] = [];
  url: string;
  listeners = new Map<string, ((e: MessageEvent) => void)[]>();
  onopen: ((e: Event) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }
  addEventListener(type: string, cb: (e: MessageEvent) => void) {
    const arr = this.listeners.get(type) ?? [];
    arr.push(cb);
    this.listeners.set(type, arr);
  }
  removeEventListener() {}
  close() {
    this.closed = true;
  }
  fire(event: string, data: unknown) {
    const listeners = this.listeners.get(event) ?? [];
    const evt = new MessageEvent(event, { data: JSON.stringify(data) });
    for (const l of listeners) l(evt);
  }
  fireError() {
    if (this.onerror) this.onerror(new Event("error"));
  }
}

describe("useResources", () => {
  beforeEach(() => {
    MockEventSource.instances = [];
    vi.stubGlobal("EventSource", MockEventSource);
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("starts with null snapshot, then updates on message", async () => {
    const r = useResources();
    expect(r.snapshot.value).toBeNull();

    const es = MockEventSource.instances[0];
    es.fire("snapshot", snap({ hostname: "alpha", timestamp: 42 }));
    await nextTick();

    expect(r.snapshot.value?.hostname).toBe("alpha");
    expect(r.snapshot.value?.timestamp).toBe(42);

    r.stop();
  });

  it("reconnects after an error with exponential backoff", async () => {
    const r = useResources();
    const first = MockEventSource.instances[0];
    first.fireError();

    // Backoff: first retry fires at 1000ms.
    vi.advanceTimersByTime(1000);
    await nextTick();

    expect(MockEventSource.instances.length).toBeGreaterThanOrEqual(2);
    expect(first.closed).toBe(true);

    r.stop();
  });

  it("replaces snapshot on each message (no append)", async () => {
    const r = useResources();
    const es = MockEventSource.instances[0];
    es.fire("snapshot", snap({ timestamp: 1 }));
    await nextTick();
    es.fire("snapshot", snap({ timestamp: 2 }));
    await nextTick();

    expect(r.snapshot.value?.timestamp).toBe(2);
    r.stop();
  });

  it("gpuList exposes the gpus array or [] when snapshot is null", async () => {
    const r = useResources();
    expect(r.gpuList.value).toEqual([]);

    const es = MockEventSource.instances[0];
    es.fire(
      "snapshot",
      snap({
        gpus: [
          {
            ordinal: 0,
            name: "RTX 3090",
            backend: "cuda",
            vram_total: 24,
            vram_used: 10,
            vram_used_by_mold: 8,
            vram_used_by_other: 2,
          },
        ],
      }),
    );
    await nextTick();

    expect(r.gpuList.value.length).toBe(1);
    expect(r.gpuList.value[0].ordinal).toBe(0);
    r.stop();
  });
});
