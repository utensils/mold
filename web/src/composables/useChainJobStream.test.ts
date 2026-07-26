import { afterEach, describe, expect, it, vi } from "vitest";
import { effectScope, nextTick, ref } from "vue";
import { useChainJobStream } from "./useChainJobStream";

const fetchEventSourceMock = vi.hoisted(() =>
  vi.fn<(...args: unknown[]) => Promise<void>>(() => new Promise(() => {})),
);

vi.mock("@microsoft/fetch-event-source", () => ({
  fetchEventSource: fetchEventSourceMock,
}));

vi.mock("../api", () => ({
  chainJobEventsUrl: (id: string) => `/api/chain-jobs/${id}/events`,
  getChainJob: vi.fn(() => new Promise(() => {})),
}));

describe("useChainJobStream", () => {
  afterEach(() => {
    fetchEventSourceMock.mockClear();
  });

  it("marks a dropped stream disconnected and delegates one retry to fetch-event-source", async () => {
    const scope = effectScope();
    const stream = scope.run(() => useChainJobStream(ref("job-1")))!;
    await nextTick();

    const options = fetchEventSourceMock.mock.calls[0]![1] as {
      onopen: (response: Response) => Promise<void>;
      onerror: () => number | void;
    };
    await options.onopen(new Response(null, { status: 200 }));
    expect(stream.connected.value).toBe(true);

    expect(options.onerror()).toBe(1_000);
    expect(stream.connected.value).toBe(false);
    expect(fetchEventSourceMock).toHaveBeenCalledTimes(1);
    scope.stop();
  });
});
