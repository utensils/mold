import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick, ref } from "vue";
import {
  cancelQueueJob,
  hostCapabilities,
  hostModelDownload,
  hostDiscoveryPeers,
  hostDownloads,
  hostQueue,
  hostStatus,
  moveQueueJob,
  pauseHostQueue,
  resumeHostQueue,
  cancelAllHostQueue,
  setQueueJobLane,
  useHostPoll,
} from "./hostClient";
import type { HostEntry } from "../../lib/hostRegistry";
import { ApiHttpError } from "../../api";

const originalFetch = globalThis.fetch;
let fetchMock: ReturnType<typeof vi.fn>;

function ok(data: unknown, status = 200) {
  return {
    ok: status < 400,
    status,
    statusText: "",
    headers: new Headers(),
    json: async () => data,
    clone() {
      return this;
    },
  };
}

function failed(status: number, body = "") {
  return {
    ok: false,
    status,
    text: async () => body,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((next, fail) => {
    resolve = next;
    reject = fail;
  });
  return { promise, resolve, reject };
}

const currentStatus = {
  version: "0.20.2",
  instance_id: "instance-1",
  hostname: "studio",
  queue_depth: 0,
  gpu_info: { backend: "cuda" },
};

const currentDevice = {
  id: "cuda:0",
  backend: "cuda",
  ordinal: 0,
  device_kind: "full_gpu",
  nvml_uuid: "GPU-0",
  physical_uuid: "GPU-0",
  mig_uuid: null,
  mig_parent_uuid: null,
  mig_profile: null,
  name: "GPU 0",
  pci_bus_id: null,
  compute_capability: "8.6",
  memory: {
    total_bytes: 24 * 1024 ** 3,
    used_bytes: 0,
    mold_used_bytes: 0,
    other_used_bytes: 0,
  },
  telemetry: {
    utilization_percent: 0,
    temperature_c: 30,
    power_w: 20,
  },
  desired_enabled: true,
  admin_state: "enabled",
  health: "healthy",
  activity: "idle",
  schedulable: true,
  unschedulable_reason: null,
  loaded_models: [],
  active_work_id: null,
  planned_work_ids: [],
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
    expect(new Headers((init as RequestInit).headers).get("x-api-key")).toBe(
      "sekret",
    );
  });

  it("omits the auth header for a keyless host", async () => {
    fetchMock.mockResolvedValueOnce(ok(currentStatus));
    await hostStatus(keyless);
    const init = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(
      (init.headers as Record<string, string>)["x-api-key"],
    ).toBeUndefined();
  });

  it("preserves an HTTP credential rejection as typed authority evidence", async () => {
    fetchMock.mockResolvedValueOnce(failed(401, "API key was rejected"));

    const error = await hostStatus(remote).catch((cause: unknown) => cause);

    expect(error).toBeInstanceOf(ApiHttpError);
    expect(error).toMatchObject({ status: 401 });
  });

  it("requests an explicit bounded queue page and preserves the legacy path", async () => {
    fetchMock
      .mockResolvedValueOnce(
        ok({
          entries: [],
          plan: null,
          live_only_entries: [],
          page: { limit: 17, offset: 0, returned: 0 },
        }),
      )
      .mockResolvedValueOnce(ok({ entries: [], plan: null }));

    await hostQueue(remote, undefined, { limit: 17 });
    await hostQueue(remote);

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "http://192.168.1.20:7680/api/queue?limit=17",
      "http://192.168.1.20:7680/api/queue",
    ]);
  });

  it("fetches discovery peers from the primary with its API-key header", async () => {
    fetchMock.mockResolvedValueOnce(ok([]));
    await hostDiscoveryPeers(remote);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://192.168.1.20:7680/api/discovery/peers");
    expect(new Headers((init as RequestInit).headers).get("x-api-key")).toBe(
      "sekret",
    );
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
  it("keeps verified status online but stale through a transient timeout", async () => {
    let failStatus = false;
    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/status")) {
        return failStatus
          ? Promise.reject(new Error("status timeout"))
          : Promise.resolve(ok(currentStatus));
      }
      return Promise.resolve(ok({ devices: [], plan_version: 1 }));
    });
    const poll = useHostPoll(remote, { intervalMs: 60_000 });
    await vi.waitFor(() => expect(poll.online.value).toBe(true));

    failStatus = true;
    await poll.refresh();

    expect(poll.status.value).toEqual(currentStatus);
    expect(poll.online.value).toBe(true);
    expect(poll.stale.value).toBe(true);
    expect(poll.error.value).toBe("status timeout");
    poll.stop();
  });

  it("retires a verified poll session when an auxiliary health endpoint rejects auth", async () => {
    let rejectResources = false;
    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/status"))
        return Promise.resolve(ok(currentStatus));
      if (url.endsWith("/api/resources"))
        return Promise.resolve(
          rejectResources
            ? failed(403, "API key was rejected")
            : ok({ hostname: "studio", gpus: [] }),
        );
      return Promise.resolve(ok({ devices: [], plan_version: 1 }));
    });
    const poll = useHostPoll(remote, {
      withResources: true,
      intervalMs: 60_000,
    });
    await vi.waitFor(() => expect(poll.online.value).toBe(true));

    rejectResources = true;
    await poll.refresh();

    expect(poll.status.value).toBeNull();
    expect(poll.online.value).toBe(false);
    expect(poll.stale.value).toBe(false);
    expect(poll.authorityRejected.value).toBe(true);
    poll.stop();
  });

  it("retains last-good devices when only the device probe blips", async () => {
    let failDevices = false;
    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/status"))
        return Promise.resolve(ok(currentStatus));
      if (failDevices) return Promise.reject(new Error("device timeout"));
      return Promise.resolve(ok({ devices: [currentDevice], plan_version: 1 }));
    });
    const poll = useHostPoll(remote, { intervalMs: 60_000 });
    await vi.waitFor(() => expect(poll.devices.value).toHaveLength(1));

    failDevices = true;
    await poll.refresh();

    expect(poll.devices.value).toHaveLength(1);
    expect(poll.online.value).toBe(true);
    poll.stop();
  });

  it("queues one follow-up instead of overlapping same-target refreshes", async () => {
    const first = deferred<ReturnType<typeof ok>>();
    const second = deferred<ReturnType<typeof ok>>();
    let statusCalls = 0;
    fetchMock.mockImplementation((input: RequestInfo | URL) => {
      if (String(input).endsWith("/api/status")) {
        statusCalls += 1;
        return statusCalls === 1 ? first.promise : second.promise;
      }
      return Promise.resolve(ok({ devices: [], plan_version: 1 }));
    });
    const poll = useHostPoll(remote, { intervalMs: 60_000 });

    const requested = poll.refresh();
    void poll.refresh();
    expect(statusCalls).toBe(1);

    first.resolve(ok(currentStatus));
    await vi.waitFor(() => expect(statusCalls).toBe(2));
    expect(statusCalls).toBe(2);

    second.resolve(ok({ ...currentStatus, queue_depth: 2 }));
    await requested;
    expect(poll.status.value?.queue_depth).toBe(2);
    poll.stop();
  });

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

describe("hostClient library organization plumbing", () => {
  it("adapts a web host entry to the studio ApiTarget shape", async () => {
    const { hostApiTarget } = await import("./hostClient");
    expect(hostApiTarget(remote)).toEqual({
      baseUrl: "http://192.168.1.20:7680",
      apiKey: "sekret",
    });
    expect(hostApiTarget(keyless)).toEqual({
      baseUrl: "http://localhost:7680",
      apiKey: null,
    });
  });

  it("lists the trash view with the host's key", async () => {
    const { hostGallery } = await import("./hostClient");
    fetchMock.mockResolvedValueOnce(ok([]));
    await hostGallery(remote, undefined, "trash");
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe("http://192.168.1.20:7680/api/gallery?view=trash");
    expect(new Headers((init as RequestInit).headers).get("x-api-key")).toBe(
      "sekret",
    );
  });

  it("keeps the bare gallery path for the default library view", async () => {
    const { hostGallery } = await import("./hostClient");
    fetchMock.mockResolvedValueOnce(ok([]));
    await hostGallery(keyless);
    expect(fetchMock.mock.calls[0]![0]).toBe(
      "http://localhost:7680/api/gallery",
    );
  });

  it("reads and writes one config key on a specific host", async () => {
    const { hostConfigValue, hostWriteConfig } = await import("./hostClient");
    fetchMock.mockResolvedValueOnce(
      ok({ key: "gallery.trash_retention_days", value: 7, source: "db" }),
    );
    await expect(
      hostConfigValue(remote, "gallery.trash_retention_days"),
    ).resolves.toBe(7);
    expect(fetchMock.mock.calls[0]![0]).toBe(
      "http://192.168.1.20:7680/api/config/gallery.trash_retention_days",
    );

    fetchMock.mockResolvedValueOnce(ok({}));
    await hostWriteConfig(remote, "gallery.trash_retention_days", 30);
    const [url, init] = fetchMock.mock.calls[1]!;
    expect(url).toBe(
      "http://192.168.1.20:7680/api/config/gallery.trash_retention_days",
    );
    expect((init as RequestInit).method).toBe("PUT");
    expect((init as RequestInit).body).toBe(JSON.stringify({ value: 30 }));
    expect((init as RequestInit).headers).toMatchObject({
      "x-api-key": "sekret",
      "content-type": "application/json",
    });
  });

  it("surfaces a failed config write instead of swallowing it as a 404", async () => {
    const { hostWriteConfig } = await import("./hostClient");
    fetchMock.mockResolvedValueOnce(ok({}, 404));
    await expect(
      hostWriteConfig(remote, "gallery.trash_retention_days", 30),
    ).rejects.toThrow(/404/);
  });
});
