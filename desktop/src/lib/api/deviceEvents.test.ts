import { beforeEach, describe, expect, it, vi } from "vitest";

const sseStream = vi.hoisted(() => vi.fn().mockResolvedValue(undefined));

vi.mock("./sse", () => ({ sseStream }));

import { subscribeToDeviceSnapshots } from "./deviceEvents";

beforeEach(() => {
  sseStream.mockClear();
});

describe("subscribeToDeviceSnapshots", () => {
  it("uses the explicit authenticated target and refetches on open and relevant events", async () => {
    const refresh = vi.fn();
    const target = { baseUrl: "https://gpu-host.example", apiKey: "secret" };

    subscribeToDeviceSnapshots(target, new AbortController().signal, refresh);

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
      expect.objectContaining({ target, retry: true }),
    );

    options.onOpen();
    options.onEvent("message", JSON.stringify({ type: "job_queued" }));
    options.onEvent("message", JSON.stringify({ type: "device_state_changed", state: {} }));
    options.onEvent("message", JSON.stringify({ type: "queue_plan_changed", plan: {} }));

    expect(refresh).toHaveBeenCalledTimes(3);
  });

  it("ignores malformed and unrelated frames", () => {
    const refresh = vi.fn();
    subscribeToDeviceSnapshots(
      { baseUrl: "http://host", apiKey: null },
      new AbortController().signal,
      refresh,
    );
    const [, options] = sseStream.mock.calls[0] as [
      string,
      { onEvent: (event: string, data: string) => void },
    ];

    options.onEvent("message", "not-json");
    options.onEvent("message", JSON.stringify({}));
    options.onEvent("message", JSON.stringify({ type: "gallery_added" }));

    expect(refresh).not.toHaveBeenCalled();
  });
});
