import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick, ref } from "vue";
import {
  cancelQueueJob,
  hostCapabilities,
  hostModelDownload,
  hostDiscoveryPeers,
  hostDownloads,
  hostStatus,
  moveQueueJob,
  pauseHostQueue,
  resumeHostQueue,
  cancelAllHostQueue,
  setQueueJobLane,
  useHostPoll,
} from "./hostClient";
import type { HostEntry } from "../../lib/hostRegistry";

const originalFetch = globalThis.fetch;
let fetchMock: ReturnType<typeof vi.fn>;

function ok(data: unknown, status = 200) {
  return { ok: status < 400, status, json: async () => data };
}

const currentStatus = {
  version: "0.20.2",
  instance_id: "instance-1",
  hostname: "studio",
  queue_depth: 0,
  gpu_info: { backend: "cuda" },
};

const remote: HostEntry = {
  id: "192-168-1-20-7680",
  name: "Studio",
  url: "http://192.168.1.20:7680",
  apiKey: "sekret",
};
const keyless: HostEntry = {
  id: "origin",
  name: "this server",
  url: "http://localhost:7680",
};

beforeEach(() => {
  fetchMock = vi.fn();
  globalThis.fetch = fetchMock as unknown as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe("hostClient auth + requests", () => {
  it("attaches the x-api-key header for keyed hosts and hits the host origin", async () => {
    fetchMock.mockResolvedValueOnce(ok(currentStatus));
    await hostStatus(remote);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://192.168.1.20:7680/api/status");
    expect((init as RequestInit).headers).toMatchObject({
      "x-api-key": "sekret",
    });
  });

  it("omits the auth header for a keyless host", async () => {
    fetchMock.mockResolvedValueOnce(ok(currentStatus));
    await hostStatus(keyless);
    const init = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(
      (init.headers as Record<string, string>)["x-api-key"],
    ).toBeUndefined();
  });

  it("fetches discovery peers from the primary with its API-key header", async () => {
    fetchMock.mockResolvedValueOnce(ok([]));
    await hostDiscoveryPeers(remote);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://192.168.1.20:7680/api/discovery/peers");
    expect((init as RequestInit).headers).toMatchObject({
      "x-api-key": "sekret",
    });
  });

  it("PATCHes a lane change with target_gpu", async () => {
    fetchMock.mockResolvedValueOnce(ok({}, 200));
    await setQueueJobLane(remote, "job1", 1);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://192.168.1.20:7680/api/queue/job1");
    expect((init as RequestInit).method).toBe("PATCH");
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({
      target_gpu: 1,
    });
  });

  it("PATCHes a reorder with an absolute position", async () => {
    fetchMock.mockResolvedValueOnce(ok({}, 200));
    await moveQueueJob(remote, "job1", 0);
    const init = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(JSON.parse(init.body as string)).toEqual({ position: 0 });
  });

  it("rejects a 404 cancel because cancellation was not confirmed", async () => {
    fetchMock.mockResolvedValueOnce(ok({}, 404));
    await expect(cancelQueueJob(remote, "job1")).rejects.toThrow(
      "DELETE /api/queue/job1 failed: 404",
    );
  });

  it("routes queue-wide controls to the selected host", async () => {
    fetchMock.mockResolvedValue(ok({}, 200));
    await pauseHostQueue(remote);
    await resumeHostQueue(remote);
    await cancelAllHostQueue(remote);

    expect(
      fetchMock.mock.calls.map(([url, init]) => [url, init.method]),
    ).toEqual([
      ["http://192.168.1.20:7680/api/queue/pause", "POST"],
      ["http://192.168.1.20:7680/api/queue/resume", "POST"],
      ["http://192.168.1.20:7680/api/queue", "DELETE"],
    ]);
  });

  it("falls back to controls-off capabilities when unreachable", async () => {
    fetchMock.mockRejectedValueOnce(new Error("network down"));
    const caps = await hostCapabilities(remote);
    expect(caps.queue?.can_reorder).toBe(false);
  });

  it("normalizes a null active download field", async () => {
    fetchMock.mockResolvedValueOnce(
      ok({ active: null, queued: [], history: [] }),
    );
    const listing = await hostDownloads(remote);
    expect(listing.active).toBeNull();
    expect(listing.queued).toEqual([]);
  });

  it("starts a catalog download on the chosen host with its key header", async () => {
    fetchMock.mockResolvedValueOnce(ok({ model: "cv:1", queued: 2 }));
    const result = await hostModelDownload(remote, "cv:1/2");

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe(
      "http://192.168.1.20:7680/api/catalog/cv%3A1%2F2/download",
    );
    expect((init as RequestInit).method).toBe("POST");
    expect((init as RequestInit).headers).toMatchObject({
      "x-api-key": "sekret",
    });
    expect(result).toMatchObject({ queued: 2 });
  });

  it("sends a manifest model name to /api/downloads, which is the only route that accepts it", async () => {
    // `/api/catalog/:id/download` answers 400 "id must be `cv:` or `hf:`
    // prefixed" for a plain manifest name, and the installed shelf is full of
    // them — so the id shape, not the caller, picks the endpoint.
    fetchMock.mockResolvedValueOnce(ok({ status: "created", id: "job-1" }));
    const result = await hostModelDownload(remote, "flux-schnell:q8");

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://192.168.1.20:7680/api/downloads");
    expect((init as RequestInit).method).toBe("POST");
    expect((init as RequestInit).headers).toMatchObject({
      "x-api-key": "sekret",
      "content-type": "application/json",
    });
    expect((init as RequestInit).body).toBe(
      JSON.stringify({ model: "flux-schnell:q8" }),
    );
    expect(result).toBeNull();
  });

  it("surfaces a rejected catalog download instead of reporting success", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 404,
      text: async () => "unknown catalog entry",
      json: async () => ({}),
    });
    await expect(hostModelDownload(remote, "cv:1")).rejects.toThrow(
      /unknown catalog entry/,
    );
  });
});

describe("useHostPoll target sessions", () => {
  it("clears host A immediately when rebinding to a stalled host B", async () => {
    const target = ref<HostEntry>({ ...remote });
    fetchMock.mockImplementation(
      (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.startsWith(remote.url)) {
          if (url.endsWith("/api/status"))
            return Promise.resolve(ok(currentStatus));
          if (url.endsWith("/api/resources"))
            return Promise.resolve(ok({ hostname: "studio-a", gpus: [] }));
          if (url.endsWith("/api/devices"))
            return Promise.resolve(ok({ devices: [], plan_version: 1 }));
        }
        return new Promise((_resolve, reject) => {
          init?.signal?.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          );
        });
      },
    );

    const poll = useHostPoll(target, {
      withResources: true,
      intervalMs: 60_000,
    });
    await vi.waitFor(() => expect(poll.online.value).toBe(true));
    expect(poll.resources.value).not.toBeNull();

    target.value = {
      id: "host-b",
      name: "Stalled B",
      url: "http://stalled-b:7680",
      apiKey: "b-key",
    };
    await nextTick();

    expect(poll.status.value).toBeNull();
    expect(poll.devices.value).toBeNull();
    expect(poll.deviceState.value).toBeNull();
    expect(poll.resources.value).toBeNull();
    expect(poll.online.value).toBe(false);
    expect(poll.lastSeen.value).toBeNull();
    expect(poll.error.value).toBeNull();
    expect(poll.loading.value).toBe(true);
    poll.stop();
  });

  it("clears stale resources when the rebound host succeeds without them", async () => {
    const target = ref<HostEntry>({ ...remote });
    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith(remote.url)) {
        if (url.endsWith("/api/status"))
          return Promise.resolve(ok(currentStatus));
        if (url.endsWith("/api/resources"))
          return Promise.resolve(ok({ hostname: "studio-a", gpus: [] }));
        return Promise.resolve(ok({ devices: [], plan_version: 1 }));
      }
      if (url.endsWith("/api/status"))
        return Promise.resolve(
          ok({
            ...currentStatus,
            instance_id: "instance-b",
            hostname: "studio-b",
          }),
        );
      if (url.endsWith("/api/resources"))
        return Promise.reject(new Error("resources unsupported"));
      return Promise.resolve(ok({ devices: [], plan_version: 2 }));
    });

    const poll = useHostPoll(target, {
      withResources: true,
      intervalMs: 60_000,
    });
    await vi.waitFor(() => expect(poll.resources.value).not.toBeNull());

    target.value = {
      id: "host-b",
      name: "Studio B",
      url: "http://studio-b:7680",
    };
    await vi.waitFor(() =>
      expect(poll.status.value?.instance_id).toBe("instance-b"),
    );

    expect(poll.online.value).toBe(true);
    expect(poll.resources.value).toBeNull();
    poll.stop();
  });

  it("rebinds when a reactive host keeps its id but rotates URL and key", async () => {
    const target = ref<HostEntry>({ ...remote });
    fetchMock.mockResolvedValue(ok(currentStatus));
    const poll = useHostPoll(target, { intervalMs: 60_000 });
    await vi.waitFor(() => expect(poll.online.value).toBe(true));
    fetchMock.mockClear();

    target.value = {
      ...target.value,
      url: "http://rotated:7680",
      apiKey: "rotated-key",
    };
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled());

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://rotated:7680/api/status");
    expect((init as RequestInit).headers).toMatchObject({
      "x-api-key": "rotated-key",
    });
    poll.stop();
  });
});
