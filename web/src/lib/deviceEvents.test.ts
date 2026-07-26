import { beforeEach, describe, expect, it, vi } from "vitest";
import { subscribeToDeviceSnapshots } from "./deviceEvents";

const fetchEventSource = vi.hoisted(() => vi.fn().mockResolvedValue(undefined));
const apiJsonTo = vi.hoisted(() =>
  vi.fn().mockResolvedValue({ events: { available: true } }),
);

vi.mock("@microsoft/fetch-event-source", () => ({ fetchEventSource }));
vi.mock("@studio/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/client")>()),
  apiJsonTo,
}));

beforeEach(() => {
  fetchEventSource.mockClear();
  apiJsonTo.mockClear();
});

describe("subscribeToDeviceSnapshots", () => {
  it("uses the exact target and refetches on open and semantic invalidations", async () => {
    const refresh = vi.fn();
    const controller = new AbortController();
    subscribeToDeviceSnapshots(
      { baseUrl: "https://render-box", apiKey: "rotated-key" },
      controller.signal,
      refresh,
    );
    await vi.waitFor(() => expect(fetchEventSource).toHaveBeenCalledTimes(1));
    const [url, options] = fetchEventSource.mock.calls[0]!;

    expect(apiJsonTo).toHaveBeenCalledWith(
      { baseUrl: "https://render-box", apiKey: "rotated-key" },
      "/api/capabilities",
    );
    expect(url).toBe("https://render-box/api/events");
    expect(new Headers(options.headers).get("x-api-key")).toBe("rotated-key");
    await options.onopen(new Response(null, { status: 200 }));
    options.onmessage({
      data: JSON.stringify({ type: "device_state_changed" }),
    });
    options.onmessage({
      data: JSON.stringify({ type: "queue_plan_changed" }),
    });
    options.onmessage({ data: JSON.stringify({ type: "gallery_added" }) });

    expect(refresh).toHaveBeenCalledTimes(3);
  });

  it("does not connect when the host does not advertise event support", async () => {
    apiJsonTo.mockResolvedValueOnce({ events: { available: false } });

    subscribeToDeviceSnapshots(
      { baseUrl: "https://old-host", apiKey: null },
      new AbortController().signal,
      vi.fn(),
    );
    await Promise.resolve();

    expect(fetchEventSource).not.toHaveBeenCalled();
  });

  it.each([401, 403, 404])(
    "terminates instead of retrying HTTP %i",
    async (status) => {
      subscribeToDeviceSnapshots(
        { baseUrl: "https://render-box", apiKey: "bad-key" },
        new AbortController().signal,
        vi.fn(),
      );
      await vi.waitFor(() => expect(fetchEventSource).toHaveBeenCalledTimes(1));
      const [, options] = fetchEventSource.mock.calls[0]!;
      const error = new Error("terminal") as Error & { status?: number };
      error.status = status;

      expect(() => options.onerror(error)).toThrow(error);
    },
  );
});
