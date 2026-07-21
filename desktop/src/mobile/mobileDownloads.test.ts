import { flushPromises } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { nextTick, watchEffect } from "vue";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, type ApiTarget } from "../lib/api/client";
import type { CatalogEntry, DownloadEvent, DownloadJob } from "../lib/api/types";
import type { MobileHost } from "./hosts";

const { apiFetchTo, sseStream, startCatalogDownload, streams } = vi.hoisted(() => ({
  apiFetchTo: vi.fn(),
  sseStream: vi.fn(),
  startCatalogDownload: vi.fn(),
  streams: [] as Array<{
    target: ApiTarget;
    signal: AbortSignal;
    onOpenError?: (error: Error) => void;
    onClose?: (error: Error | null) => void;
    onEvent: (event: string, data: string) => void;
  }>,
}));

vi.mock("../lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/api/client")>()),
  apiFetchTo,
}));

vi.mock("../lib/api/catalog", () => ({ startCatalogDownload }));
vi.mock("../lib/api/sse", () => ({ sseStream }));

import { useMobileDownloadsStore } from "./mobileDownloads";

const studio: MobileHost = {
  id: "studio",
  name: "Studio",
  baseUrl: "https://studio.test:7680",
  apiKey: "studio-key",
  hostname: "studio",
  version: "0.18.0",
  online: true,
};

const render: MobileHost = {
  id: "render",
  name: "Render",
  baseUrl: "https://render.test:7680",
  apiKey: "render-key",
  hostname: "render",
  version: "0.18.0",
  online: true,
};

const entry: CatalogEntry = {
  id: "hf:owner/repo",
  source: "hf",
  source_id: "owner/repo",
  name: "Friendly model",
  family: "flux",
  kind: "checkpoint",
  nsfw: false,
  installed: false,
  size_bytes: 1,
  thumbnail_url: null,
};

function snapshot(
  activeJobs: DownloadJob[] = [],
  queued: DownloadJob[] = [],
  history: DownloadJob[] = [],
): DownloadEvent {
  return { type: "snapshot", listing: { active_jobs: activeJobs, queued, history } };
}

function queuedJob(id: string, model: string, catalogId?: string): DownloadJob {
  return {
    id,
    model,
    catalog_id: catalogId ?? null,
    status: "queued",
    files_done: 0,
    files_total: 0,
    bytes_done: 0,
    bytes_total: 0,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

beforeEach(() => {
  setActivePinia(createPinia());
  streams.length = 0;
  vi.clearAllMocks();
  apiFetchTo.mockResolvedValue(new Response(null, { status: 204 }));
  startCatalogDownload.mockResolvedValue("download-1");
  sseStream.mockImplementation(
    (
      _path: string,
      options: {
        target: ApiTarget;
        signal: AbortSignal;
        onOpenError?: (error: Error) => void;
        onClose?: (error: Error | null) => void;
        onEvent: (event: string, data: string) => void;
      },
    ) => {
      streams.push(options);
      return Promise.resolve();
    },
  );
});

describe("mobile download authority", () => {
  it("shares one stream and waits for its reduced opening snapshot before posting", async () => {
    const store = useMobileDownloadsStore();
    store.registerConsumer("catalog", [studio]);
    store.registerConsumer("generate", [studio]);
    expect(streams).toHaveLength(1);

    const pull = store.startPull(entry, studio);
    await flushPromises();
    expect(store.pullStatusForHost(entry, studio.id)?.phase).toBe("connecting");
    expect(startCatalogDownload).not.toHaveBeenCalled();

    streams[0]!.onEvent("download", JSON.stringify(snapshot()));
    await flushPromises();
    expect(startCatalogDownload).toHaveBeenCalledWith(
      entry.id,
      {
        baseUrl: studio.baseUrl,
        apiKey: studio.apiKey,
      },
      false,
    );
    await pull;
  });

  it("adopts opening-snapshot jobs and suppresses duplicate pulls", async () => {
    const store = useMobileDownloadsStore();
    store.registerConsumer("catalog", [studio]);

    const first = store.startPull(entry, studio);
    const second = store.startPull(entry, studio);
    streams[0]!.onEvent(
      "download",
      JSON.stringify(snapshot([queuedJob("already-running", "canonical-name", entry.id)])),
    );
    await Promise.all([first, second]);

    expect(startCatalogDownload).not.toHaveBeenCalled();
    expect(store.pullStatusForHost(entry, studio.id)).toEqual({
      label: "Queued",
      phase: "queued",
    });
  });

  it("adopts the real bare expansion alias but keeps colon-bearing catalog ids literal", async () => {
    const store = useMobileDownloadsStore();
    const expansion = { id: "qwen3-expand", name: "qwen3-expand" };
    store.registerConsumer("generate", [studio]);

    const pull = store.startPull(expansion, studio);
    streams[0]!.onEvent(
      "download",
      JSON.stringify(snapshot([], [queuedJob("canonical", "qwen3-expand:q8")])),
    );
    await expect(pull).resolves.toEqual({ kind: "adopted" });
    expect(startCatalogDownload).not.toHaveBeenCalled();
    expect(store.pullStatusForHost(expansion, studio.id)?.phase).toBe("queued");

    setActivePinia(createPinia());
    streams.length = 0;
    const literalStore = useMobileDownloadsStore();
    literalStore.registerConsumer("catalog", [studio]);
    const literalPull = literalStore.startPull(entry, studio);
    streams[0]!.onEvent(
      "download",
      JSON.stringify(snapshot([], [queuedJob("prefixed", `${entry.name}:q8`)])),
    );
    await literalPull;
    expect(startCatalogDownload).toHaveBeenCalledWith(entry.id, expect.anything(), false);
  });

  it("adopts a valid 409 body id as the authoritative bare-alias job", async () => {
    const store = useMobileDownloadsStore();
    const expansion = { id: "qwen3-expand", name: "qwen3-expand" };
    startCatalogDownload.mockRejectedValue(
      new ApiError("already queued", 409, { id: "existing-A" }),
    );
    store.registerConsumer("generate", [studio]);
    streams[0]!.onEvent("download", JSON.stringify(snapshot()));

    await expect(store.startPull(expansion, studio)).resolves.toEqual({
      kind: "conflict",
      jobId: "existing-A",
    });
    streams[0]!.onEvent(
      "download",
      JSON.stringify({
        type: "enqueued",
        id: "competing-B",
        model: "qwen3-expand:q8",
        position: 0,
      } satisfies DownloadEvent),
    );
    expect(store.pullStatusForHost(expansion, studio.id)?.phase).toBe("starting");

    streams[0]!.onEvent(
      "download",
      JSON.stringify({
        type: "enqueued",
        id: "existing-A",
        model: "qwen3-expand:q8",
        position: 1,
      } satisfies DownloadEvent),
    );
    expect(store.pullStatusForHost(expansion, studio.id)?.phase).toBe("queued");
  });

  it("ignores matching in-flight and terminal races after the opening snapshot", async () => {
    const store = useMobileDownloadsStore();
    const expansion = { id: "qwen3-expand:q8", name: "qwen3-expand:q8" };
    const response = deferred<string | null>();
    startCatalogDownload.mockReturnValue(response.promise);
    store.registerConsumer("generate", [studio]);
    streams[0]!.onEvent("download", JSON.stringify(snapshot()));

    const pull = store.startPull(expansion, studio);
    await flushPromises();
    expect(store.pullStatusForHost(expansion, studio.id)?.phase).toBe("starting");

    streams[0]!.onEvent(
      "download",
      JSON.stringify(snapshot([], [queuedJob("competing-B", "qwen3-expand:q8")])),
    );
    expect(store.pullStatusForHost(expansion, studio.id)?.phase).toBe("starting");
    expect(store.pendingPulls.get(`${studio.id}:${expansion.id}`)?.jobId).toBeNull();

    streams[0]!.onEvent(
      "download",
      JSON.stringify({
        type: "job_done",
        id: "competing-B",
        model: "qwen3-expand:q8",
      } satisfies DownloadEvent),
    );
    expect(store.pullStatusForHost(expansion, studio.id)?.phase).toBe("starting");

    response.resolve("our-job-A");
    await expect(pull).resolves.toEqual({ kind: "started", jobId: "our-job-A" });
    expect(store.pendingPulls.get(`${studio.id}:${expansion.id}`)?.jobId).toBe("our-job-A");
  });

  it("retains returned job-id association across canonical model-name differences", async () => {
    const store = useMobileDownloadsStore();
    const start = deferred<string | null>();
    startCatalogDownload.mockReturnValue(start.promise);
    store.registerConsumer("catalog", [studio]);
    streams[0]!.onEvent("download", JSON.stringify(snapshot()));

    const pull = store.startPull(entry, studio);
    await flushPromises();
    start.resolve("canonical-job");
    await pull;
    streams[0]!.onEvent(
      "download",
      JSON.stringify({
        type: "enqueued",
        id: "canonical-job",
        model: "manifest-canonical:q8",
        position: 0,
      } satisfies DownloadEvent),
    );

    expect(store.pullStatusForHost(entry, studio.id)?.phase).toBe("queued");
    streams[0]!.onEvent(
      "download",
      JSON.stringify({
        type: "job_done",
        id: "canonical-job",
        model: "manifest-canonical:q8",
      } satisfies DownloadEvent),
    );
    expect(store.pullStatusForHost(entry, studio.id)).toBeNull();
    expect(store.downloadsByHost[studio.id]?.history[0]).toMatchObject({
      id: "canonical-job",
      status: "completed",
    });
  });

  it("derives exact-host ETA from the shared progress stream", () => {
    const store = useMobileDownloadsStore();
    const now = vi.spyOn(Date, "now");
    store.registerConsumer("generate", [studio]);
    streams[0]!.onEvent(
      "download",
      JSON.stringify(
        snapshot([
          {
            ...queuedJob("eta-job", "qwen3-expand:q8"),
            status: "active",
            bytes_done: 100,
            bytes_total: 1_000,
          },
        ]),
      ),
    );
    now.mockReturnValueOnce(1_000);
    streams[0]!.onEvent(
      "download",
      JSON.stringify({ type: "progress", id: "eta-job", files_done: 0, bytes_done: 100 }),
    );
    now.mockReturnValueOnce(2_000);
    streams[0]!.onEvent(
      "download",
      JSON.stringify({ type: "progress", id: "eta-job", files_done: 0, bytes_done: 300 }),
    );

    expect(store.etaFor(studio.id, "eta-job")).toBe(4);
    now.mockRestore();
  });

  it("prefers the returned job id over a competing same-name job", async () => {
    const store = useMobileDownloadsStore();
    startCatalogDownload.mockResolvedValue("our-job");
    store.registerConsumer("catalog", [studio]);
    streams[0]!.onEvent("download", JSON.stringify(snapshot()));
    await store.startPull(entry, studio);

    streams[0]!.onEvent(
      "download",
      JSON.stringify(
        snapshot(
          [
            {
              ...queuedJob("someone-elses-job", entry.name),
              status: "active",
              files_total: 1,
              bytes_done: 75,
              bytes_total: 100,
            },
          ],
          [queuedJob("our-job", "manifest-canonical:q8")],
        ),
      ),
    );

    expect(store.pullStatusForHost(entry, studio.id)).toEqual({
      label: "Queued",
      phase: "queued",
    });

    streams[0]!.onEvent(
      "download",
      JSON.stringify({
        type: "job_done",
        id: "someone-elses-job",
        model: entry.name,
      } satisfies DownloadEvent),
    );
    expect(store.pullStatusForHost(entry, studio.id)?.phase).toBe("queued");
  });

  it("replaces changed host targets, ignores stale frames, and isolates equal job ids", () => {
    const store = useMobileDownloadsStore();
    store.registerConsumer("catalog", [studio, render]);
    expect(streams).toHaveLength(2);

    streams[0]!.onEvent("download", JSON.stringify(snapshot([], [queuedJob("same", "studio")])));
    streams[1]!.onEvent("download", JSON.stringify(snapshot([], [queuedJob("same", "render")])));
    expect(store.downloadsByHost[studio.id]?.queued[0]?.model).toBe("studio");
    expect(store.downloadsByHost[render.id]?.queued[0]?.model).toBe("render");

    const rotated = { ...studio, apiKey: "rotated-key" };
    const stale = streams[0]!;
    store.registerConsumer("catalog", [rotated, render]);
    expect(stale.signal.aborted).toBe(true);
    expect(streams).toHaveLength(3);
    expect(streams[2]!.target.apiKey).toBe("rotated-key");

    stale.onEvent("download", JSON.stringify(snapshot([], [queuedJob("stale", "stale")])));
    expect(store.downloadsByHost[studio.id]?.queued[0]?.id).toBe("same");
  });

  it("lets the latest consumer registration replace a conflicting stale host target", () => {
    const store = useMobileDownloadsStore();
    store.registerConsumer("catalog", [studio]);
    store.registerConsumer("generate", [studio]);
    const stale = streams[0]!;

    store.registerConsumer("catalog", [{ ...studio, apiKey: "rotated-key" }]);

    expect(stale.signal.aborted).toBe(true);
    expect(streams).toHaveLength(2);
    expect(streams[1]!.target).toEqual({
      baseUrl: studio.baseUrl,
      apiKey: "rotated-key",
    });
  });

  it("routes a stale caller pull through the current winning consumer lease", async () => {
    const store = useMobileDownloadsStore();
    const expansion = { id: "qwen3-expand", name: "qwen3-expand" };
    const rotated = { ...studio, apiKey: "rotated-key" };
    startCatalogDownload.mockResolvedValue("rotated-job");

    store.registerConsumer("catalog", [studio]);
    store.registerConsumer("generate", [studio]);
    store.registerConsumer("catalog", [rotated]);
    const freshStream = streams[1]!;
    expect(streams[0]!.signal.aborted).toBe(true);

    const pull = store.startPull(expansion, studio);
    streams.at(-1)!.onEvent("download", JSON.stringify(snapshot()));
    await expect(pull).resolves.toEqual({ kind: "started", jobId: "rotated-job" });

    expect(streams).toHaveLength(2);
    expect(freshStream.signal.aborted).toBe(false);
    expect(startCatalogDownload).toHaveBeenCalledWith(
      expansion.id,
      { baseUrl: studio.baseUrl, apiKey: "rotated-key" },
      false,
    );
  });

  it("keeps a frozen Generate pull on its exact credentials despite a competing Catalog lease", async () => {
    const store = useMobileDownloadsStore();
    const expansion = { id: "qwen3-expand:q8", name: "qwen3-expand:q8" };
    const rotated = { ...studio, apiKey: "catalog-rotated-key" };
    store.registerConsumer("catalog", [rotated]);
    expect(streams[0]!.target.apiKey).toBe("catalog-rotated-key");

    const pull = store.startPullFrozen(expansion, studio, "generate-attempt-1");
    expect(streams[0]!.signal.aborted).toBe(true);
    expect(streams[1]!.target).toEqual({
      baseUrl: studio.baseUrl,
      apiKey: studio.apiKey,
    });

    store.registerConsumer("catalog", [rotated]);
    expect(streams).toHaveLength(2);
    expect(streams[1]!.signal.aborted).toBe(false);
    streams[1]!.onEvent("download", JSON.stringify(snapshot()));
    await expect(pull).resolves.toEqual({ kind: "started", jobId: "download-1" });
    expect(startCatalogDownload).toHaveBeenCalledWith(
      expansion.id,
      { baseUrl: studio.baseUrl, apiKey: studio.apiKey },
      false,
    );

    store.releaseFrozenPull("generate-attempt-1");
    expect(streams[1]!.signal.aborted).toBe(true);
    expect(streams[2]!.target.apiKey).toBe("catalog-rotated-key");
  });

  it.each([
    ["connection", "connect"],
    ["POST", "post"],
  ] as const)(
    "releases a failed frozen %s attempt back to Catalog and reacquires only for Retry",
    async (_label, failure) => {
      const store = useMobileDownloadsStore();
      const expansion = { id: "qwen3-expand:q8", name: "qwen3-expand:q8" };
      const rotated = { ...studio, apiKey: "catalog-rotated-key" };
      store.registerConsumer("catalog", [rotated]);

      if (failure === "post") startCatalogDownload.mockRejectedValue(new Error("POST denied"));
      const first = store.startPullFrozen(expansion, studio, "generate-attempt-1");
      expect(streams.at(-1)!.target.apiKey).toBe(studio.apiKey);
      if (failure === "connect") streams.at(-1)!.onOpenError?.(new Error("stream denied"));
      else streams.at(-1)!.onEvent("download", JSON.stringify(snapshot()));
      await expect(first).rejects.toThrow(failure === "connect" ? "stream denied" : "POST denied");

      expect(streams.at(-1)!.target.apiKey).toBe("catalog-rotated-key");
      startCatalogDownload.mockRejectedValue(new Error("retry denied"));
      const retry = store.startPullFrozen(expansion, studio, "generate-attempt-1");
      expect(streams.at(-1)!.target.apiKey).toBe(studio.apiKey);
      streams.at(-1)!.onEvent("download", JSON.stringify(snapshot()));
      await expect(retry).rejects.toThrow("retry denied");
      expect(streams.at(-1)!.target.apiKey).toBe("catalog-rotated-key");
    },
  );

  it.each([
    ["job_done", "completed"],
    ["job_failed", "failed"],
    ["job_cancelled", "cancelled"],
  ] as const)("releases a frozen lease after terminal %s", async (type, _status) => {
    const store = useMobileDownloadsStore();
    const expansion = { id: "qwen3-expand:q8", name: "qwen3-expand:q8" };
    const rotated = { ...studio, apiKey: "catalog-rotated-key" };
    store.registerConsumer("catalog", [rotated]);
    const pull = store.startPullFrozen(expansion, studio, "generate-terminal");
    streams.at(-1)!.onEvent("download", JSON.stringify(snapshot()));
    await expect(pull).resolves.toEqual({ kind: "started", jobId: "download-1" });
    const event: DownloadEvent =
      type === "job_done"
        ? { type, id: "download-1", model: expansion.id }
        : type === "job_failed"
          ? { type, id: "download-1", error: "disk full" }
          : { type, id: "download-1" };
    streams.at(-1)!.onEvent("download", JSON.stringify(event));

    expect(streams.at(-1)!.target.apiKey).toBe("catalog-rotated-key");
  });

  it("joins an exact compatible Catalog POST already in Starting", async () => {
    const store = useMobileDownloadsStore();
    const expansion = { id: "qwen3-expand:q8", name: "qwen3-expand:q8" };
    const response = deferred<string | null>();
    startCatalogDownload.mockReturnValue(response.promise);
    store.registerConsumer("catalog", [studio]);
    const catalogPull = store.startPull(expansion, studio);
    streams[0]!.onEvent("download", JSON.stringify(snapshot()));
    await flushPromises();
    expect(store.pendingPulls.get(`${studio.id}:${expansion.id}`)?.phase).toBe("starting");

    const generatePull = store.startPullFrozen(expansion, studio, "generate-join");
    expect(startCatalogDownload).toHaveBeenCalledTimes(1);
    response.resolve("shared-exact-job");
    await expect(catalogPull).resolves.toEqual({ kind: "started", jobId: "shared-exact-job" });
    await expect(generatePull).resolves.toEqual({ kind: "started", jobId: "shared-exact-job" });
    expect(store.pendingPulls.get(`${studio.id}:${expansion.id}`)?.jobId).toBe("shared-exact-job");

    streams[0]!.onEvent(
      "download",
      JSON.stringify({
        type: "enqueued",
        id: "shared-exact-job",
        model: expansion.id,
        position: 0,
      } satisfies DownloadEvent),
    );
    expect(store.pullStatusForHost(expansion, studio.id)?.phase).toBe("queued");
    expect(startCatalogDownload).toHaveBeenCalledTimes(1);
  });

  it("shares the exact 409 job ID when Generate joins a starting Catalog POST", async () => {
    const store = useMobileDownloadsStore();
    const expansion = { id: "qwen3-expand:q8", name: "qwen3-expand:q8" };
    const response = deferred<string | null>();
    startCatalogDownload.mockReturnValue(response.promise);
    store.registerConsumer("catalog", [studio]);
    const catalogPull = store.startPull(expansion, studio);
    streams[0]!.onEvent("download", JSON.stringify(snapshot()));
    await flushPromises();

    const generatePull = store.startPullFrozen(expansion, studio, "generate-join-409");
    response.reject(new ApiError("already queued", 409, { id: "shared-conflict-job" }));
    await expect(catalogPull).resolves.toEqual({
      kind: "conflict",
      jobId: "shared-conflict-job",
    });
    await expect(generatePull).resolves.toEqual({
      kind: "conflict",
      jobId: "shared-conflict-job",
    });
    expect(store.pendingPulls.get(`${studio.id}:${expansion.id}`)?.jobId).toBe(
      "shared-conflict-job",
    );
    expect(startCatalogDownload).toHaveBeenCalledTimes(1);
  });

  it("keeps a different-model frozen pull isolated from a starting Catalog POST", async () => {
    const store = useMobileDownloadsStore();
    const catalogEntry = { id: "qwen3-expand:q8", name: "qwen3-expand:q8" };
    const generateEntry = { id: "other-expand:q8", name: "other-expand:q8" };
    const catalogResponse = deferred<string | null>();
    const generateResponse = deferred<string | null>();
    startCatalogDownload
      .mockReturnValueOnce(catalogResponse.promise)
      .mockReturnValueOnce(generateResponse.promise);
    store.registerConsumer("catalog", [studio]);
    const catalogPull = store.startPull(catalogEntry, studio);
    streams[0]!.onEvent("download", JSON.stringify(snapshot()));
    await flushPromises();

    const generatePull = store.startPullFrozen(generateEntry, studio, "generate-isolated");
    await flushPromises();
    expect(startCatalogDownload).toHaveBeenCalledTimes(2);
    catalogResponse.resolve("catalog-job");
    generateResponse.resolve("generate-job");
    await expect(catalogPull).resolves.toEqual({ kind: "started", jobId: "catalog-job" });
    await expect(generatePull).resolves.toEqual({ kind: "started", jobId: "generate-job" });
  });

  it("keeps direct caller routing when no consumer lease exists", async () => {
    const store = useMobileDownloadsStore();
    const expansion = { id: "qwen3-expand", name: "qwen3-expand" };
    const pull = store.startPull(expansion, studio);

    expect(streams).toHaveLength(1);
    expect(streams[0]!.target).toEqual({ baseUrl: studio.baseUrl, apiKey: studio.apiKey });
    streams[0]!.onEvent("download", JSON.stringify(snapshot()));

    await expect(pull).resolves.toEqual({ kind: "started", jobId: "download-1" });
    expect(startCatalogDownload).toHaveBeenCalledWith(
      expansion.id,
      { baseUrl: studio.baseUrl, apiKey: studio.apiKey },
      false,
    );
  });

  it("preserves pending pull ownership across credential-only stream replacement", async () => {
    const store = useMobileDownloadsStore();
    const response = deferred<string | null>();
    const rotated = { ...studio, apiKey: "rotated-key" };
    startCatalogDownload.mockReturnValue(response.promise);
    store.registerConsumer("catalog", [studio]);
    streams[0]!.onEvent("download", JSON.stringify(snapshot()));

    const pull = store.startPull(entry, studio);
    await flushPromises();
    expect(store.pendingPulls.get(`${studio.id}:${entry.id}`)?.phase).toBe("starting");

    store.registerConsumer("catalog", [rotated]);
    expect(streams[0]!.signal.aborted).toBe(true);
    expect(store.pendingPulls.get(`${studio.id}:${entry.id}`)?.phase).toBe("starting");
    streams[1]!.onEvent("download", JSON.stringify(snapshot()));
    expect(store.pendingPulls.get(`${studio.id}:${entry.id}`)?.jobId).toBeNull();

    response.resolve("exact-job-A");
    await expect(pull).resolves.toEqual({ kind: "started", jobId: "exact-job-A" });
    expect(store.pendingPulls.get(`${studio.id}:${entry.id}`)?.jobId).toBe("exact-job-A");

    streams[1]!.onEvent(
      "download",
      JSON.stringify({
        type: "enqueued",
        id: "exact-job-A",
        model: "manifest-canonical:q8",
        position: 0,
      } satisfies DownloadEvent),
    );
    expect(store.pullStatusForHost(entry, studio.id)?.phase).toBe("queued");
    await expect(store.startPull(entry, studio)).resolves.toEqual({ kind: "existing" });
    expect(startCatalogDownload).toHaveBeenCalledTimes(1);
  });

  it("follows a credential replacement that happens before the opening snapshot", async () => {
    const store = useMobileDownloadsStore();
    const expansion = { id: "qwen3-expand", name: "qwen3-expand" };
    const rotated = { ...studio, apiKey: "rotated-key" };
    const errors: string[] = [];
    const listener = {
      onStreamError: (_host: MobileHost, cause: Error) => errors.push(cause.message),
    };
    store.registerConsumer("catalog", [studio], listener);

    const pull = store.startPull(expansion, studio);
    void pull.catch(() => {});
    expect(store.pendingPulls.get(`${studio.id}:${expansion.id}`)?.phase).toBe("connecting");
    store.registerConsumer("catalog", [rotated], listener);

    expect(streams[0]!.signal.aborted).toBe(true);
    expect(store.pendingPulls.get(`${studio.id}:${expansion.id}`)?.phase).toBe("connecting");
    streams[1]!.onEvent("download", JSON.stringify(snapshot()));

    await expect(pull).resolves.toEqual({ kind: "started", jobId: "download-1" });
    expect(startCatalogDownload).toHaveBeenCalledTimes(1);
    expect(startCatalogDownload).toHaveBeenCalledWith(
      expansion.id,
      { baseUrl: studio.baseUrl, apiKey: "rotated-key" },
      false,
    );
    expect(errors).toEqual([]);
  });

  it("follows compatible instance-id enrichment before the opening snapshot", async () => {
    const store = useMobileDownloadsStore();
    const expansion = { id: "qwen3-expand", name: "qwen3-expand" };
    const host: MobileHost = { ...studio, instanceId: undefined };
    store.registerConsumer("catalog", [host]);

    const pull = store.startPull(expansion, host);
    void pull.catch(() => {});
    host.instanceId = "server-A";
    store.registerConsumer("catalog", [host]);

    expect(streams[0]!.signal.aborted).toBe(true);
    expect(store.pendingPulls.get(`${host.id}:${expansion.id}`)?.phase).toBe("connecting");
    streams[1]!.onEvent("download", JSON.stringify(snapshot()));
    await expect(pull).resolves.toEqual({ kind: "started", jobId: "download-1" });
    expect(startCatalogDownload).toHaveBeenCalledTimes(1);
  });

  it("follows the current generation through unresolved A to B to new A supersession", async () => {
    const store = useMobileDownloadsStore();
    const expansion = { id: "qwen3-expand", name: "qwen3-expand" };
    const targetB = { ...studio, apiKey: "key-B" };
    const errors: string[] = [];
    const listener = {
      onStreamError: (_host: MobileHost, cause: Error) => errors.push(cause.message),
    };
    store.registerConsumer("catalog", [studio], listener);
    store.registerConsumer("generate", [studio], listener);

    const pull = store.startPull(expansion, studio);
    void pull.catch(() => {});
    store.registerConsumer("catalog", [targetB], listener);
    store.registerConsumer("generate", [studio], listener);

    expect(streams).toHaveLength(3);
    expect(streams[0]!.signal.aborted).toBe(true);
    expect(streams[1]!.signal.aborted).toBe(true);
    expect(startCatalogDownload).not.toHaveBeenCalled();
    streams[2]!.onEvent("download", JSON.stringify(snapshot()));

    await expect(pull).resolves.toEqual({ kind: "started", jobId: "download-1" });
    expect(startCatalogDownload).toHaveBeenCalledTimes(1);
    expect(startCatalogDownload).toHaveBeenCalledWith(
      expansion.id,
      { baseUrl: studio.baseUrl, apiKey: studio.apiKey },
      false,
    );
    expect(errors).toEqual([]);
  });

  it("does not POST after old A resolves until new A's snapshot adopts the pull", async () => {
    const store = useMobileDownloadsStore();
    const expansion = { id: "qwen3-expand", name: "qwen3-expand" };
    const targetB = { ...studio, apiKey: "key-B" };
    store.registerConsumer("catalog", [studio]);
    store.registerConsumer("generate", [studio]);

    const pull = store.startPull(expansion, studio);
    streams[0]!.onEvent("download", JSON.stringify(snapshot()));
    store.registerConsumer("catalog", [targetB]);
    store.registerConsumer("generate", [studio]);
    await flushPromises();
    expect(startCatalogDownload).not.toHaveBeenCalled();

    streams[2]!.onEvent(
      "download",
      JSON.stringify(snapshot([], [queuedJob("existing-A", "qwen3-expand:q8")])),
    );
    await expect(pull).resolves.toEqual({ kind: "adopted" });
    expect(startCatalogDownload).not.toHaveBeenCalled();
  });

  it("clears pending ownership when the winning host endpoint changes", async () => {
    const store = useMobileDownloadsStore();
    const response = deferred<string | null>();
    startCatalogDownload.mockReturnValue(response.promise);
    store.registerConsumer("catalog", [studio]);
    streams[0]!.onEvent("download", JSON.stringify(snapshot()));

    const pull = store.startPull(entry, studio);
    await flushPromises();
    store.registerConsumer("catalog", [
      { ...studio, baseUrl: "https://replacement.test:7680", apiKey: "replacement-key" },
    ]);

    expect(store.pendingPulls.get(`${studio.id}:${entry.id}`)).toBeUndefined();
    response.resolve("old-endpoint-job");
    await pull;
  });

  it("uses immutable identity when an endpoint mutates in place and resets failed replacement state", async () => {
    const store = useMobileDownloadsStore();
    const response = deferred<string | null>();
    const errors: string[] = [];
    const contexts: Array<{ hadSnapshot: boolean }> = [];
    const host = { ...studio, instanceId: "server-A" };
    const oldHistory = queuedJob("shared-terminal", "old-model");
    oldHistory.status = "completed";
    startCatalogDownload.mockReturnValue(response.promise);
    store.registerConsumer("catalog", [host], {
      onEvent: ({ hadSnapshot }) => contexts.push({ hadSnapshot }),
      onStreamError: (_host, cause) => errors.push(cause.message),
    });
    streams[0]!.onEvent(
      "download",
      JSON.stringify(snapshot([], [queuedJob("old-job", "old-model")], [oldHistory])),
    );

    const pull = store.startPull(entry, host);
    await flushPromises();
    host.baseUrl = "https://replacement.test:7680";
    host.apiKey = "replacement-key";
    host.instanceId = "server-B";
    store.registerConsumer("catalog", [host], {
      onEvent: ({ hadSnapshot }) => contexts.push({ hadSnapshot }),
      onStreamError: (_host, cause) => errors.push(cause.message),
    });

    expect(store.pendingPulls.get(`${host.id}:${entry.id}`)).toBeUndefined();
    expect(store.downloadsByHost[host.id]).toEqual({ activeJobs: [], queued: [], history: [] });
    streams[1]!.onOpenError?.(new Error("replacement unauthorized"));
    expect(errors).toEqual(["replacement unauthorized"]);

    store.registerConsumer("catalog", [host], {
      onEvent: ({ hadSnapshot }) => contexts.push({ hadSnapshot }),
    });
    streams[2]!.onEvent("download", JSON.stringify(snapshot([], [], [oldHistory])));
    expect(contexts.at(-1)?.hadSnapshot).toBe(false);

    response.resolve("old-server-job");
    await pull;
  });

  it("uses immutable identity when a known instance id mutates in place", async () => {
    const store = useMobileDownloadsStore();
    const response = deferred<string | null>();
    const host = { ...studio, instanceId: "server-A" };
    startCatalogDownload.mockReturnValue(response.promise);
    store.registerConsumer("catalog", [host]);
    streams[0]!.onEvent("download", JSON.stringify(snapshot()));

    const pull = store.startPull(entry, host);
    await flushPromises();
    host.instanceId = "server-B";
    store.registerConsumer("catalog", [host]);

    expect(store.pendingPulls.get(`${host.id}:${entry.id}`)).toBeUndefined();
    expect(store.downloadsByHost[host.id]).toEqual({ activeJobs: [], queued: [], history: [] });
    streams[1]!.onEvent("download", JSON.stringify(snapshot()));
    response.resolve("old-server-job");
    await pull;
  });

  it("cleans host state after an already-closed stream loses its last consumer", () => {
    const store = useMobileDownloadsStore();
    const contexts: Array<{ hadSnapshot: boolean }> = [];
    const completed = queuedJob("terminal-1", "completed-model");
    completed.status = "completed";
    store.registerConsumer("catalog", [studio], {
      onEvent: ({ hadSnapshot }) => contexts.push({ hadSnapshot }),
    });
    streams[0]!.onEvent("download", JSON.stringify(snapshot([], [], [completed])));
    streams[0]!.onClose?.(new Error("network stopped"));

    store.unregisterConsumer("catalog");
    expect(store.downloadsByHost[studio.id]).toBeUndefined();

    store.registerConsumer("catalog", [studio], {
      onEvent: ({ hadSnapshot }) => contexts.push({ hadSnapshot }),
    });
    streams[1]!.onEvent("download", JSON.stringify(snapshot([], [], [completed])));
    expect(contexts.at(-1)?.hadSnapshot).toBe(false);
  });

  it("routes stale-caller cancellation through the current lease and deduplicates callers", async () => {
    const store = useMobileDownloadsStore();
    const rotated = { ...studio, apiKey: "rotated-key" };
    const cancellation = deferred<Response>();
    const job = queuedJob("job-1", "qwen3-expand:q8");
    apiFetchTo.mockReturnValue(cancellation.promise);
    store.registerConsumer("catalog", [studio]);
    store.registerConsumer("generate", [studio]);
    store.registerConsumer("catalog", [rotated]);

    const staleCaller = store.cancel(studio, job);
    const freshCaller = store.cancel(rotated, job);

    expect(apiFetchTo).toHaveBeenCalledTimes(1);
    expect(apiFetchTo).toHaveBeenCalledWith(
      { baseUrl: studio.baseUrl, apiKey: "rotated-key" },
      `/api/downloads/${job.id}`,
      { method: "DELETE" },
    );
    cancellation.resolve(new Response(null, { status: 204 }));
    await Promise.all([staleCaller, freshCaller]);
    expect(store.isCancelling(studio, job.id)).toBe(false);
  });

  it.each(["endpoint", "known-instance"] as const)(
    "does not deduplicate cancellation across an incompatible %s replacement",
    async (change) => {
      const store = useMobileDownloadsStore();
      const oldRequest = deferred<Response>();
      const replacementRequest = deferred<Response>();
      const oldHost = { ...studio, instanceId: "server-A" };
      const replacement = {
        ...oldHost,
        baseUrl: change === "endpoint" ? "https://replacement.test:7680" : oldHost.baseUrl,
        apiKey: "replacement-key",
        instanceId: "server-B",
      };
      const job = queuedJob("same-job", "qwen3-expand:q8");
      apiFetchTo
        .mockReturnValueOnce(oldRequest.promise)
        .mockReturnValueOnce(replacementRequest.promise);
      store.registerConsumer("catalog", [oldHost]);

      const oldCancel = store.cancel(oldHost, job);
      store.registerConsumer("catalog", [replacement]);
      const replacementCancel = store.cancel(replacement, job);

      expect(apiFetchTo).toHaveBeenCalledTimes(2);
      expect(apiFetchTo.mock.calls[0]?.[0]).toEqual({
        baseUrl: oldHost.baseUrl,
        apiKey: oldHost.apiKey,
      });
      expect(apiFetchTo.mock.calls[1]?.[0]).toEqual({
        baseUrl: replacement.baseUrl,
        apiKey: replacement.apiKey,
      });
      oldRequest.resolve(new Response(null, { status: 204 }));
      replacementRequest.resolve(new Response(null, { status: 204 }));
      await Promise.all([oldCancel, replacementCancel]);
    },
  );

  it("keeps incompatible replacement cancellation busy and retry state independent", async () => {
    const store = useMobileDownloadsStore();
    const oldRequest = deferred<Response>();
    const replacementRequest = deferred<Response>();
    const retryRequest = deferred<Response>();
    const oldHost = { ...studio, instanceId: "server-A" };
    const replacement = {
      ...oldHost,
      baseUrl: "https://replacement.test:7680",
      apiKey: "replacement-key",
      instanceId: "server-B",
    };
    const job = queuedJob("same-job", "qwen3-expand:q8");
    apiFetchTo
      .mockReturnValueOnce(oldRequest.promise)
      .mockReturnValueOnce(replacementRequest.promise)
      .mockReturnValueOnce(retryRequest.promise);
    store.registerConsumer("catalog", [oldHost]);

    const oldCancel = store.cancel(oldHost, job);
    store.registerConsumer("catalog", [replacement]);
    expect(store.isCancelling(replacement, job.id)).toBe(false);

    const replacementCancel = store.cancel(replacement, job);
    expect(store.isCancelling(replacement, job.id)).toBe(true);
    replacementRequest.reject(new Error("replacement denied"));
    await expect(replacementCancel).rejects.toThrow("replacement denied");
    expect(store.isCancelling(replacement, job.id)).toBe(false);

    const retry = store.cancel(replacement, job);
    expect(apiFetchTo).toHaveBeenCalledTimes(3);
    expect(store.isCancelling(replacement, job.id)).toBe(true);
    retryRequest.resolve(new Response(null, { status: 204 }));
    await retry;
    expect(store.isCancelling(replacement, job.id)).toBe(false);

    oldRequest.resolve(new Response(null, { status: 204 }));
    await oldCancel;
  });

  it("keeps an in-flight cancellation deduplicated across compatible credential rotation", async () => {
    const store = useMobileDownloadsStore();
    const request = deferred<Response>();
    const rotated = { ...studio, apiKey: "rotated-key", instanceId: "server-A" };
    const original = { ...studio, instanceId: "server-A" };
    const job = queuedJob("same-job", "qwen3-expand:q8");
    apiFetchTo.mockReturnValue(request.promise);
    store.registerConsumer("catalog", [original]);

    const originalCancel = store.cancel(original, job);
    expect(store.isCancelling(original, job.id)).toBe(true);
    store.registerConsumer("catalog", [rotated]);
    expect(store.isCancelling(rotated, job.id)).toBe(true);
    const rotatedCancel = store.cancel(rotated, job);

    expect(apiFetchTo).toHaveBeenCalledTimes(1);
    request.resolve(new Response(null, { status: 204 }));
    await Promise.all([originalCancel, rotatedCancel]);
    expect(store.isCancelling(rotated, job.id)).toBe(false);
  });

  it("keeps direct cancellation routing when no consumer lease exists", async () => {
    const store = useMobileDownloadsStore();
    const job = queuedJob("job-direct", "qwen3-expand:q8");

    await store.cancel(studio, job);

    expect(apiFetchTo).toHaveBeenCalledWith(
      { baseUrl: studio.baseUrl, apiKey: studio.apiKey },
      `/api/downloads/${job.id}`,
      { method: "DELETE" },
    );
  });

  it("reactively reports cancellation for the current authoritative identity", async () => {
    const store = useMobileDownloadsStore();
    const request = deferred<Response>();
    const job = queuedJob("reactive-job", "qwen3-expand:q8");
    const states: boolean[] = [];
    apiFetchTo.mockReturnValue(request.promise);
    store.registerConsumer("catalog", [studio]);
    const stop = watchEffect(() => states.push(store.isCancelling(studio, job.id)));

    const cancellation = store.cancel(studio, job);
    await nextTick();
    expect(states.at(-1)).toBe(true);
    request.resolve(new Response(null, { status: 204 }));
    await cancellation;
    await nextTick();
    expect(states.at(-1)).toBe(false);
    stop();
  });

  it("deduplicates cancellation and keeps a stream until its last consumer leaves", async () => {
    const store = useMobileDownloadsStore();
    const cancellation = deferred<Response>();
    apiFetchTo.mockReturnValue(cancellation.promise);
    const job = queuedJob("job-1", "flux-dev:q8");

    store.registerConsumer("catalog", [studio]);
    store.registerConsumer("generate", [studio]);
    streams[0]!.onEvent("download", JSON.stringify(snapshot([], [job])));

    const first = store.cancel(studio, job);
    const second = store.cancel(studio, job);
    expect(store.isCancelling(studio, job.id)).toBe(true);
    expect(apiFetchTo).toHaveBeenCalledTimes(1);
    cancellation.resolve(new Response(null, { status: 204 }));
    await Promise.all([first, second]);
    expect(store.isCancelling(studio, job.id)).toBe(false);

    store.unregisterConsumer("catalog");
    expect(streams[0]!.signal.aborted).toBe(false);
    store.unregisterConsumer("generate");
    expect(streams[0]!.signal.aborted).toBe(true);
    expect(store.downloadsByHost[studio.id]).toBeUndefined();
  });
});
