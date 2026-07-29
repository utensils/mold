import { beforeEach, describe, expect, it, vi } from "vitest";

const sseStream = vi.hoisted(() => vi.fn().mockResolvedValue(undefined));
const fetchServerCapabilities = vi.hoisted(() =>
  vi.fn().mockResolvedValue({ events: { available: true } }),
);

vi.mock("./sse", () => ({ sseStream }));
vi.mock("./serverCapabilities", () => ({ fetchServerCapabilities }));

import { subscribeToDeviceSnapshots } from "./deviceEvents";

beforeEach(() => {
  sseStream.mockClear();
  fetchServerCapabilities.mockClear();
});

describe("subscribeToDeviceSnapshots", () => {
  it("uses the explicit authenticated target and refetches on open and relevant events", async () => {
    const refresh = vi.fn();
    const target = { baseUrl: "https://gpu-host.example", apiKey: "secret" };

    subscribeToDeviceSnapshots(target, new AbortController().signal, refresh);
    await vi.waitFor(() => expect(sseStream).toHaveBeenCalledTimes(1));

    const [, options] = sseStream.mock.calls[0] as [
      string,
      {
        target: typeof target;
        onOpen: () => void;
        onEvent: (event: string, data: string) => void;
      },
    ];
    expect(sseStream).toHaveBeenCalledWith(
      "/api/events",
      expect.objectContaining({
        target,
        retry: true,
        terminalHttpStatuses: [401, 403, 404],
      }),
    );

    options.onOpen();
    options.onEvent("message", JSON.stringify({ type: "job_queued" }));
    options.onEvent("message", JSON.stringify({ type: "device_state_changed", state: {} }));
    options.onEvent("message", JSON.stringify({ type: "queue_plan_changed", plan: {} }));

    expect(refresh).toHaveBeenCalledTimes(3);
  });

  it("ignores malformed and unrelated frames", async () => {
    const refresh = vi.fn();
    subscribeToDeviceSnapshots(
      { baseUrl: "http://host", apiKey: null },
      new AbortController().signal,
      refresh,
    );
    await vi.waitFor(() => expect(sseStream).toHaveBeenCalledTimes(1));
    const [, options] = sseStream.mock.calls[0] as [
      string,
      { onEvent: (event: string, data: string) => void },
    ];

    options.onEvent("message", "not-json");
    options.onEvent("message", JSON.stringify({}));
    options.onEvent("message", JSON.stringify({ type: "gallery_added" }));

    expect(refresh).not.toHaveBeenCalled();
  });

  it("does not connect when the capability probe says events are unavailable", async () => {
    fetchServerCapabilities.mockResolvedValueOnce({ events: { available: false } });
    subscribeToDeviceSnapshots(
      { baseUrl: "http://old-host", apiKey: null },
      new AbortController().signal,
      vi.fn(),
    );

    await Promise.resolve();

    expect(sseStream).not.toHaveBeenCalled();
  });
});
