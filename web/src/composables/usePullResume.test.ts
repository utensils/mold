import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const hostDownloadsMock = vi.hoisted(() => vi.fn());
const toastMock = vi.hoisted(() => vi.fn());
vi.mock("../components/machines/hostClient", () => ({
  hostDownloads: hostDownloadsMock,
}));
vi.mock("../lib/toasts", () => ({ toast: toastMock }));
vi.mock("../lib/hostRegistry", () => ({
  ORIGIN_HOST_ID: "origin",
  getHost: (id: string) => ({ id, url: `http://${id}:7680`, apiKey: null }),
}));

import { usePullResume } from "./usePullResume";

const listing = (
  jobs: Array<{ id: string; model: string; status: string; error?: string }>,
) => ({ active: null, active_jobs: [], queued: [], history: jobs });

describe("usePullResume", () => {
  beforeEach(() => {
    hostDownloadsMock.mockReset();
    toastMock.mockReset();
    usePullResume().cancel();
  });
  afterEach(() => usePullResume().cancel());

  it("resumes exactly once when the watched machine's pull completes", async () => {
    const resume = vi.fn();
    hostDownloadsMock.mockResolvedValue(
      listing([{ id: "a", model: "z-image-turbo:q6", status: "downloading" }]),
    );
    const pullResume = usePullResume();
    await pullResume.arm({
      model: "z-image-turbo:q6",
      jobId: null,
      hostId: "hal9000-7680",
      hostLabel: "hal9000",
      resume,
    });
    expect(resume).not.toHaveBeenCalled();
    expect(pullResume.pending.value).not.toBeNull();

    hostDownloadsMock.mockResolvedValue(
      listing([{ id: "a", model: "z-image-turbo:q6", status: "completed" }]),
    );
    await pullResume.check();
    await pullResume.check();

    expect(resume).toHaveBeenCalledTimes(1);
    expect(pullResume.pending.value).toBeNull();
    expect(toastMock).toHaveBeenCalledWith(
      "success",
      "z-image-turbo:q6 is ready on hal9000 — generating",
    );
  });

  it("never resumes off a pull that had already finished when armed", async () => {
    const resume = vi.fn();
    hostDownloadsMock.mockResolvedValue(
      listing([{ id: "old", model: "z-image-turbo:q6", status: "completed" }]),
    );
    const pullResume = usePullResume();
    await pullResume.arm({
      model: "z-image-turbo:q6",
      jobId: null,
      hostId: "hal9000-7680",
      hostLabel: "hal9000",
      resume,
    });
    await pullResume.check();

    expect(resume).not.toHaveBeenCalled();
    expect(pullResume.pending.value).not.toBeNull();
  });

  it("reports a failed pull and does not generate", async () => {
    const resume = vi.fn();
    hostDownloadsMock.mockResolvedValue(
      listing([{ id: "a", model: "z-image-turbo:q6", status: "downloading" }]),
    );
    const pullResume = usePullResume();
    await pullResume.arm({
      model: "z-image-turbo:q6",
      jobId: null,
      hostId: "hal9000-7680",
      hostLabel: "hal9000",
      resume,
    });
    hostDownloadsMock.mockResolvedValue(
      listing([
        {
          id: "a",
          model: "z-image-turbo:q6",
          status: "failed",
          error: "disk full",
        },
      ]),
    );
    await pullResume.check();

    expect(resume).not.toHaveBeenCalled();
    expect(pullResume.pending.value).toBeNull();
    expect(toastMock).toHaveBeenCalledWith(
      "error",
      "Download of z-image-turbo:q6 failed — disk full; generation not resumed.",
    );
  });

  it("resumes a pull that finished before the watch was armed", async () => {
    const resume = vi.fn();
    const pullResume = usePullResume();
    hostDownloadsMock.mockResolvedValue(listing([]));
    // Captured BEFORE the POST — the pull then completes inside the window
    // between starting it and arming the watch.
    const baseline = await pullResume.captureBaseline("hal9000-7680");
    hostDownloadsMock.mockResolvedValue(
      listing([{ id: "a", model: "z-image-turbo:q6", status: "completed" }]),
    );
    await pullResume.arm(
      {
        model: "z-image-turbo:q6",
        jobId: null,
        hostId: "hal9000-7680",
        hostLabel: "hal9000",
        resume,
      },
      baseline,
    );

    expect(resume).toHaveBeenCalledTimes(1);
    expect(pullResume.pending.value).toBeNull();
  });

  it("keeps waiting when the machine cannot be read", async () => {
    const resume = vi.fn();
    hostDownloadsMock.mockRejectedValue(new Error("unreachable"));
    const pullResume = usePullResume();
    await pullResume.arm({
      model: "z-image-turbo:q6",
      jobId: "job-1",
      hostId: "hal9000-7680",
      hostLabel: "hal9000",
      resume,
    });
    await pullResume.check();

    expect(resume).not.toHaveBeenCalled();
    expect(pullResume.pending.value).not.toBeNull();
  });
});
