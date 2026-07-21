import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cancelQueueJob,
  hostCapabilities,
  hostDiscoveryPeers,
  hostDownloads,
  hostStatus,
  moveQueueJob,
  setQueueJobLane,
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

  it("treats a 404 cancel as already gone", async () => {
    fetchMock.mockResolvedValueOnce(ok({}, 404));
    await expect(cancelQueueJob(remote, "job1")).resolves.toBeUndefined();
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
});
