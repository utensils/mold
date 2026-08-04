import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type {
  ChainJobDetail,
  ChainJobSummary,
} from "@studio/lib/api/chainTypes";
import { ApiHttpError } from "../api";
import { addHost, ORIGIN_HOST_ID, originUrl } from "../lib/hostRegistry";
import {
  TRACKED_SEQUENCES_KEY,
  __testing__,
  useChainJobs,
} from "./useChainJobs";

const listChainJobsMock = vi.hoisted(() => vi.fn());
const createChainJobMock = vi.hoisted(() => vi.fn());
const getChainJobMock = vi.hoisted(() => vi.fn());
const cancelChainJobMock = vi.hoisted(() => vi.fn());
const resumeChainJobMock = vi.hoisted(() => vi.fn());
const retakeChainJobMock = vi.hoisted(() => vi.fn());
const deleteChainJobMock = vi.hoisted(() => vi.fn());
const gcChainJobsMock = vi.hoisted(() => vi.fn());
const amendChainJobMock = vi.hoisted(() => vi.fn());
const fetchEventSourceMock = vi.hoisted(() => vi.fn());

vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../api")>()),
  listChainJobs: listChainJobsMock,
  createChainJob: createChainJobMock,
  getChainJob: getChainJobMock,
  cancelChainJob: cancelChainJobMock,
  resumeChainJob: resumeChainJobMock,
  retakeChainJob: retakeChainJobMock,
  deleteChainJob: deleteChainJobMock,
  gcChainJobs: gcChainJobsMock,
  amendChainJob: amendChainJobMock,
}));

vi.mock("@microsoft/fetch-event-source", () => ({
  fetchEventSource: fetchEventSourceMock,
}));

function summary(overrides: Partial<ChainJobSummary> = {}): ChainJobSummary {
  return {
    id: "job-1",
    state: "running",
    model: "ltx-2-19b-distilled:fp8",
    stage_count: 3,
    current_stage: 1,
    created_at_unix_ms: 100,
    updated_at_unix_ms: 100,
    ...overrides,
  };
}

function detail(overrides: Partial<ChainJobDetail> = {}): ChainJobDetail {
  return {
    ...summary(),
    stages: [
      { idx: 0, state: "completed" },
      { idx: 1, state: "running" },
      { idx: 2, state: "pending" },
    ],
    ...overrides,
  };
}

type EventSourceOpts = {
  onmessage?: (message: { event: string; data: string }) => void;
  onerror?: (err: unknown) => number | undefined;
  headers?: Record<string, string>;
  signal?: AbortSignal;
};

function lastEventSourceOpts(): EventSourceOpts {
  const call = fetchEventSourceMock.mock.calls.at(-1);
  return (call?.[1] ?? {}) as EventSourceOpts;
}

describe("useChainJobs", () => {
  beforeEach(() => {
    localStorage.clear();
    __testing__.reset();
    for (const mock of [
      listChainJobsMock,
      createChainJobMock,
      getChainJobMock,
      cancelChainJobMock,
      resumeChainJobMock,
      retakeChainJobMock,
      deleteChainJobMock,
      gcChainJobsMock,
      amendChainJobMock,
      fetchEventSourceMock,
    ]) {
      mock.mockReset();
    }
    listChainJobsMock.mockResolvedValue({ jobs: [] });
    createChainJobMock.mockResolvedValue({ job_id: "job-new" });
    getChainJobMock.mockResolvedValue(detail());
    deleteChainJobMock.mockResolvedValue(undefined);
    gcChainJobsMock.mockResolvedValue({
      swept_ephemeral_jobs: 0,
      pruned_artifact_dirs: 0,
    });
    fetchEventSourceMock.mockResolvedValue(undefined);
  });

  afterEach(() => {
    __testing__.reset();
    vi.useRealTimers();
  });

  it("migrates the legacy chain-job keys into tracked-sequences.v1", () => {
    localStorage.setItem("mold.create.chain-job", "job-legacy");
    localStorage.setItem("mold.create.chain-job-host", "plato");
    const jobs = useChainJobs();
    expect(jobs.tracked.value).toEqual([
      { hostId: "plato", jobId: "job-legacy" },
    ]);
    expect(localStorage.getItem(TRACKED_SEQUENCES_KEY)).toContain("job-legacy");
    expect(localStorage.getItem("mold.create.chain-job")).toBeNull();
    expect(localStorage.getItem("mold.create.chain-job-host")).toBeNull();
  });

  it("defaults a legacy job without a host key to the origin host", () => {
    localStorage.setItem("mold.create.chain-job", "job-legacy");
    const jobs = useChainJobs();
    expect(jobs.tracked.value).toEqual([
      { hostId: ORIGIN_HOST_ID, jobId: "job-legacy" },
    ]);
  });

  it("fetches a host's jobs newest-first with the target resolved at call time", async () => {
    const host = addHost({
      url: "http://plato:7680",
      name: "plato",
      apiKey: "k",
    });
    listChainJobsMock.mockResolvedValue({
      jobs: [
        summary({ id: "old", created_at_unix_ms: 1 }),
        summary({ id: "new", created_at_unix_ms: 2 }),
      ],
    });
    const jobs = useChainJobs();
    await jobs.fetchHost(host.id);
    expect(listChainJobsMock).toHaveBeenCalledWith({
      baseUrl: "http://plato:7680",
      apiKey: "k",
    });
    expect(jobs.state.byHost[host.id]?.jobs.map((j) => j.id)).toEqual([
      "new",
      "old",
    ]);
    expect(jobs.state.byHost[host.id]?.error).toBeNull();
  });

  it("keeps automatic long-video shim jobs out of sequence surfaces", async () => {
    const host = addHost({ url: "http://plato:7680", name: "plato" });
    listChainJobsMock.mockResolvedValue({
      jobs: [
        summary({ id: "authored", ephemeral: false }),
        summary({ id: "one-shot", ephemeral: true }),
      ],
    });
    const jobs = useChainJobs();
    await jobs.fetchHost(host.id);

    expect(jobs.state.byHost[host.id]?.jobs.map((job) => job.id)).toEqual([
      "authored",
    ]);
  });

  it("keeps the last good list and records the error when a host fetch fails", async () => {
    const host = addHost({ url: "http://plato:7680", name: "plato" });
    listChainJobsMock.mockResolvedValue({ jobs: [summary({ id: "kept" })] });
    const jobs = useChainJobs();
    await jobs.fetchHost(host.id);
    listChainJobsMock.mockRejectedValue(new Error("offline"));
    await jobs.fetchHost(host.id);
    expect(jobs.state.byHost[host.id]?.jobs.map((j) => j.id)).toEqual(["kept"]);
    expect(jobs.state.byHost[host.id]?.error).toContain("offline");
  });

  it("creates on the named host, tracks the job, and starts watching it", async () => {
    const host = addHost({ url: "http://plato:7680", name: "plato" });
    // The refresh after create lists the job the server just accepted.
    listChainJobsMock.mockResolvedValue({ jobs: [summary({ id: "job-new" })] });
    const jobs = useChainJobs();
    const id = await jobs.create(host.id, {
      model: "m",
      stages: [{ prompt: "a", frames: 25 }],
      width: 768,
      height: 512,
      steps: 8,
      guidance: 3,
    });
    expect(id).toBe("job-new");
    expect(createChainJobMock.mock.calls[0]?.[1]).toEqual({
      baseUrl: "http://plato:7680",
    });
    expect(jobs.tracked.value).toContainEqual({
      hostId: host.id,
      jobId: "job-new",
    });
    expect(jobs.state.watching).toEqual({ hostId: host.id, jobId: "job-new" });
    expect(fetchEventSourceMock).toHaveBeenCalledTimes(1);
  });

  it("reduces chain_job SSE frames into the live view", async () => {
    const jobs = useChainJobs();
    jobs.watch(ORIGIN_HOST_ID, "job-1");
    const opts = lastEventSourceOpts();
    opts.onmessage?.({
      event: "chain_job",
      data: JSON.stringify({ type: "snapshot", job: detail() }),
    });
    expect(jobs.state.live.detail?.id).toBe("job-1");
    expect(jobs.state.live.activeStage).toBe(1);
    opts.onmessage?.({
      event: "chain_job",
      data: JSON.stringify({
        type: "denoise_step",
        stage_idx: 1,
        step: 4,
        total: 8,
      }),
    });
    expect(jobs.state.live.progress[1]).toEqual({ step: 4, total: 8 });
  });

  it("falls back to a 250ms poll while the watch stream errors", async () => {
    vi.useFakeTimers();
    const jobs = useChainJobs();
    jobs.watch(ORIGIN_HOST_ID, "job-1");
    const opts = lastEventSourceOpts();
    const retry = opts.onerror?.(new Error("stream down"));
    expect(retry).toBe(1_000);
    await vi.advanceTimersByTimeAsync(250);
    expect(getChainJobMock).toHaveBeenCalledWith("job-1", {
      baseUrl: originUrl(),
    });
    expect(jobs.state.live.detail?.id).toBe("job-1");
  });

  it("polls active jobs every 10s and stops when everything settles", async () => {
    vi.useFakeTimers();
    listChainJobsMock.mockResolvedValue({ jobs: [summary()] });
    const jobs = useChainJobs();
    await jobs.fetchAll();
    expect(listChainJobsMock).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(10_000);
    expect(listChainJobsMock).toHaveBeenCalledTimes(2);
    listChainJobsMock.mockResolvedValue({
      jobs: [summary({ state: "completed" })],
    });
    await vi.advanceTimersByTimeAsync(10_000);
    expect(listChainJobsMock).toHaveBeenCalledTimes(3);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(listChainJobsMock).toHaveBeenCalledTimes(3);
  });

  it("clears only inactive jobs across every host", async () => {
    const host = addHost({ url: "http://plato:7680", name: "plato" });
    listChainJobsMock.mockImplementation(
      async (target?: { baseUrl: string }) =>
        target?.baseUrl === "http://plato:7680"
          ? {
              jobs: [
                summary({ id: "p-done", state: "completed" }),
                summary({ id: "p-running", state: "running" }),
              ],
            }
          : { jobs: [summary({ id: "o-failed", state: "failed" })] },
    );
    const jobs = useChainJobs();
    await jobs.fetchAll();
    await jobs.clearInactive();
    const deleted = deleteChainJobMock.mock.calls.map((c) => c[0]);
    expect(deleted.sort()).toEqual(["o-failed", "p-done"]);
    expect(deleted).not.toContain("p-running");
    void host;
  });

  it("rethrows an amend conflict so callers can create-as-new", async () => {
    amendChainJobMock.mockRejectedValue(
      new ApiHttpError("POST /api/chain-jobs/job-1/amend", 409, "job moved on"),
    );
    const jobs = useChainJobs();
    await expect(
      jobs.amend(ORIGIN_HOST_ID, "job-1", { stages: [] }),
    ).rejects.toMatchObject({ status: 409 });
    expect(jobs.state.watching).toBeNull();
  });

  it("watches the amended job after a successful amend", async () => {
    amendChainJobMock.mockResolvedValue({
      ...summary(),
      preserved_stages: 1,
    });
    const jobs = useChainJobs();
    await jobs.amend(ORIGIN_HOST_ID, "job-1", { stages: [] });
    expect(jobs.state.watching).toEqual({
      hostId: ORIGIN_HOST_ID,
      jobId: "job-1",
    });
  });

  it("remove deletes the job, untracks it, and stops watching it", async () => {
    const jobs = useChainJobs();
    await jobs.create(ORIGIN_HOST_ID, {
      model: "m",
      stages: [{ prompt: "a", frames: 25 }],
      width: 768,
      height: 512,
      steps: 8,
      guidance: 3,
    });
    await jobs.remove(ORIGIN_HOST_ID, "job-new");
    expect(deleteChainJobMock).toHaveBeenCalled();
    expect(jobs.tracked.value).toEqual([]);
    expect(jobs.state.watching).toBeNull();
  });

  it("runs disk cleanup on every reachable host", async () => {
    addHost({ url: "http://plato:7680", name: "plato" });
    const jobs = useChainJobs();
    await jobs.gc();
    const bases = gcChainJobsMock.mock.calls.map(
      (c) => (c[0] as { baseUrl: string }).baseUrl,
    );
    expect(bases).toContain(originUrl());
    expect(bases).toContain("http://plato:7680");
  });
});
