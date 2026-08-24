import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { defineComponent, h } from "vue";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchStatus } from "../api";
import type { ServerStatus } from "../types";
import { useStatusPoll, type UseStatusPoll } from "./useStatusPoll";

vi.mock("../api", () => ({
  fetchStatus: vi.fn(),
}));

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function serverStatus(version: string): ServerStatus {
  return {
    version,
    models_loaded: [],
    busy: false,
    uptime_secs: 1,
  };
}

function setDocumentHidden(hidden: boolean): void {
  Object.defineProperty(document, "hidden", {
    configurable: true,
    value: hidden,
  });
}

function mountStatusPoll(intervalMs = 100): {
  wrapper: VueWrapper;
  poll: UseStatusPoll;
} {
  let poll!: UseStatusPoll;
  const wrapper = mount(
    defineComponent({
      setup() {
        poll = useStatusPoll(intervalMs);
        return () => h("div");
      },
    }),
  );
  return { wrapper, poll };
}

describe("useStatusPoll", () => {
  let wrapper: VueWrapper | null = null;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(fetchStatus).mockReset();
    setDocumentHidden(false);
  });

  afterEach(() => {
    wrapper?.unmount();
    wrapper = null;
    vi.useRealTimers();
    setDocumentHidden(false);
  });

  it("ignores a delayed rejection from a status request intentionally aborted while hidden", async () => {
    const first = deferred<ServerStatus>();
    const second = deferred<ServerStatus>();
    let firstSignal: AbortSignal | undefined;
    vi.mocked(fetchStatus)
      .mockImplementationOnce((signal) => {
        firstSignal = signal;
        return first.promise;
      })
      .mockImplementationOnce(() => second.promise);

    const mounted = mountStatusPoll();
    wrapper = mounted.wrapper;

    expect(fetchStatus).toHaveBeenCalledOnce();
    setDocumentHidden(true);
    document.dispatchEvent(new Event("visibilitychange"));
    expect(firstSignal?.aborted).toBe(true);

    setDocumentHidden(false);
    document.dispatchEvent(new Event("visibilitychange"));
    expect(fetchStatus).toHaveBeenCalledTimes(2);

    first.reject(new Error("Engine offline"));
    await flushPromises();

    expect(mounted.poll.error.value).toBeNull();

    second.resolve(serverStatus("current"));
    await flushPromises();
    expect(mounted.poll.status.value?.version).toBe("current");
  });

  it("keeps status requests single-flight and schedules the next poll after settlement", async () => {
    const first = deferred<ServerStatus>();
    const second = deferred<ServerStatus>();
    vi.mocked(fetchStatus)
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise);

    const mounted = mountStatusPoll();
    wrapper = mounted.wrapper;
    expect(fetchStatus).toHaveBeenCalledOnce();

    await vi.advanceTimersByTimeAsync(300);
    expect(fetchStatus).toHaveBeenCalledOnce();

    first.resolve(serverStatus("first"));
    await flushPromises();
    await vi.advanceTimersByTimeAsync(99);
    expect(fetchStatus).toHaveBeenCalledOnce();
    await vi.advanceTimersByTimeAsync(1);
    expect(fetchStatus).toHaveBeenCalledTimes(2);

    second.resolve(serverStatus("second"));
    await flushPromises();
    expect(mounted.poll.status.value?.version).toBe("second");
  });

  it("retains the last verified status and reports reconnecting after a transient failure", async () => {
    vi.mocked(fetchStatus)
      .mockResolvedValueOnce(serverStatus("last-good"))
      .mockRejectedValueOnce(new Error("status timeout"));
    const mounted = mountStatusPoll();
    wrapper = mounted.wrapper;
    await flushPromises();

    await vi.advanceTimersByTimeAsync(100);
    await flushPromises();

    expect(mounted.poll.status.value?.version).toBe("last-good");
    expect(mounted.poll.stale.value).toBe(true);
    expect(mounted.poll.error.value).toBe("status timeout");
  });

  it("aborts the active request and stops scheduling when unmounted", async () => {
    const request = deferred<ServerStatus>();
    let signal: AbortSignal | undefined;
    vi.mocked(fetchStatus).mockImplementation((requestSignal) => {
      signal = requestSignal;
      return request.promise;
    });

    const mounted = mountStatusPoll();
    wrapper = mounted.wrapper;
    expect(fetchStatus).toHaveBeenCalledOnce();

    wrapper.unmount();
    wrapper = null;

    expect(signal?.aborted).toBe(true);

    request.resolve(serverStatus("late"));
    await flushPromises();
    await vi.advanceTimersByTimeAsync(300);
    expect(fetchStatus).toHaveBeenCalledOnce();
  });
});
