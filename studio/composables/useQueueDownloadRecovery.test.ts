import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiJsonTo } from "../api/client";
import { getGenerationBatch } from "../api/generationAdmission";
import {
  getQueueJob,
  retryQueueJobRecoveringAmbiguity,
} from "../api/queuePlan";
import { runWithLicenseConsent } from "./useLicenseAcceptance";
import {
  cancelQueueDownloadRecovery,
  queueDownloadState,
  startQueueDownloadRecovery,
} from "./useQueueDownloadRecovery";

vi.mock("../api/client", async (original) => ({
  ...(await original<object>()),
  apiJsonTo: vi.fn(),
}));
vi.mock("../api/generationAdmission", () => ({ getGenerationBatch: vi.fn() }));
vi.mock("../api/queuePlan", () => ({
  getQueueJob: vi.fn(),
  retryQueueJobRecoveringAmbiguity: vi.fn(),
}));
vi.mock("./useLicenseAcceptance", () => ({
  useLicenseAcceptance: () => ({ pending: { value: null }, cancel: vi.fn() }),
  runWithLicenseConsent: vi.fn(),
}));

let sequence = 0;
let target = { baseUrl: "http://fixture", apiKey: null };
const row = {
  id: "q",
  model: "wan",
  state: "held",
  retryable: true,
  batch_id: "b",
  client_batch_id: "c",
  started_at_unix_ms: 0,
  position: 0,
};
const json = vi.mocked(apiJsonTo);
beforeEach(() => {
  vi.resetAllMocks();
  target = { baseUrl: `http://fixture-${sequence++}`, apiKey: null };
  vi.mocked(getQueueJob).mockResolvedValue({ job: { ...row } });
  vi.mocked(getGenerationBatch).mockResolvedValue({
    kind: "found",
    batch: {
      id: "b",
      instance_id: "instance",
      client_batch_id: "c",
      children: [
        {
          job_id: "q",
          state: "held",
          retryable: true,
          error_code: "MODEL_NOT_FOUND",
        },
      ],
    },
  } as Awaited<ReturnType<typeof getGenerationBatch>>);
  vi.mocked(retryQueueJobRecoveringAmbiguity).mockResolvedValue({
    kind: "accepted",
  });
  vi.mocked(runWithLicenseConsent).mockImplementation(async (options) => ({
    kind: "ok",
    value: await options.start(),
  }));
  json.mockImplementation(async (_, path) => {
    if (path === "/api/status") return { instance_id: "instance" };
    if (path === "/api/downloads")
      return {
        id: "ticket",
        active_jobs: [],
        queued: [],
        history: [{ id: "ticket", model: "wan", status: "completed" }],
      };
    throw new Error(`Unexpected ${path}`);
  });
});

describe("queue download recovery lifecycle", () => {
  it("shows starting feedback synchronously and sends only one acquisition and retry", async () => {
    const first = startQueueDownloadRecovery(target, "instance", "q", "Plato");
    expect(queueDownloadState(target, "instance", "q")).toMatchObject({
      phase: "starting",
      busy: true,
    });
    await startQueueDownloadRecovery(target, "instance", "q", "Plato");
    await first;
    expect(
      json.mock.calls.filter(
        ([, path, options]) =>
          path === "/api/downloads" && options?.method === "POST",
      ),
    ).toHaveLength(1);
    expect(retryQueueJobRecoveringAmbiguity).toHaveBeenCalledTimes(1);
    expect(queueDownloadState(target, "instance", "q")).toMatchObject({
      phase: "complete",
      busy: false,
    });
  });
  it("keeps a failed download held", async () => {
    json.mockImplementation(async (_, path, options) =>
      path === "/api/status"
        ? { instance_id: "instance" }
        : options?.method === "POST"
          ? { id: "ticket" }
          : {
              active_jobs: [],
              queued: [],
              history: [
                {
                  id: "ticket",
                  model: "wan",
                  status: "failed",
                  error: "Disk full",
                },
              ],
            },
    );
    await startQueueDownloadRecovery(target, "instance", "q", "Plato");
    expect(queueDownloadState(target, "instance", "q")).toMatchObject({
      phase: "failed",
      message: "Disk full",
      busy: false,
    });
    expect(retryQueueJobRecoveringAmbiguity).not.toHaveBeenCalled();
  });
  it("does not acquire for a changed machine or a non-missing-model hold", async () => {
    json.mockResolvedValue({ instance_id: "replacement" });
    await startQueueDownloadRecovery(target, "instance", "q", "Plato");
    expect(runWithLicenseConsent).not.toHaveBeenCalled();
    expect(queueDownloadState(target, "instance", "q")?.phase).toBe("failed");
  });
  it("license dismissal makes the outcome visible without retrying", async () => {
    vi.mocked(runWithLicenseConsent).mockResolvedValue({ kind: "declined" });
    await startQueueDownloadRecovery(target, "instance", "q", "Plato");
    expect(queueDownloadState(target, "instance", "q")).toMatchObject({
      phase: "cancelled",
      busy: false,
    });
    expect(retryQueueJobRecoveringAmbiguity).not.toHaveBeenCalled();
  });
  it("cancellation before admission leaves no acquisition or retry", async () => {
    const pending = startQueueDownloadRecovery(
      target,
      "instance",
      "q",
      "Plato",
    );
    cancelQueueDownloadRecovery(target, "instance", "q");
    await pending;
    expect(queueDownloadState(target, "instance", "q")?.phase).toBe(
      "cancelled",
    );
    expect(runWithLicenseConsent).not.toHaveBeenCalled();
    expect(retryQueueJobRecoveringAmbiguity).not.toHaveBeenCalled();
  });
  it("waits for the exact tickets returned by legacy license acceptance", async () => {
    vi.mocked(runWithLicenseConsent).mockResolvedValue({
      kind: "accepted",
      jobIds: ["ticket"],
    });
    await startQueueDownloadRecovery(target, "instance", "q", "Plato");
    expect(
      json.mock.calls.some(
        ([, path, options]) => path === "/api/downloads" && !options?.method,
      ),
    ).toBe(true);
    expect(retryQueueJobRecoveringAmbiguity).toHaveBeenCalledTimes(1);
  });
  it("does not report a reconciled held child as a successful retry", async () => {
    vi.mocked(retryQueueJobRecoveringAmbiguity).mockResolvedValue({
      kind: "reconciled",
      batch: { children: [{ job_id: "q", state: "held" }] },
    } as Awaited<ReturnType<typeof retryQueueJobRecoveringAmbiguity>>);
    await startQueueDownloadRecovery(target, "instance", "q", "Plato");
    expect(queueDownloadState(target, "instance", "q")?.phase).toBe("failed");
    expect(retryQueueJobRecoveringAmbiguity).toHaveBeenCalledTimes(1);
  });
  it("recognizes a reconciled queued child as an accepted retry", async () => {
    vi.mocked(retryQueueJobRecoveringAmbiguity).mockResolvedValue({
      kind: "reconciled",
      batch: { children: [{ job_id: "q", state: "queued" }] },
    } as Awaited<ReturnType<typeof retryQueueJobRecoveringAmbiguity>>);
    await startQueueDownloadRecovery(target, "instance", "q", "Plato");
    expect(queueDownloadState(target, "instance", "q")?.phase).toBe("complete");
  });
  it("an uncertain retry remains visible and is not repeated", async () => {
    vi.mocked(retryQueueJobRecoveringAmbiguity).mockResolvedValue({
      kind: "uncertain",
      error: "Reconnect before retrying",
    });
    await startQueueDownloadRecovery(target, "instance", "q", "Plato");
    expect(retryQueueJobRecoveringAmbiguity).toHaveBeenCalledTimes(1);
    expect(queueDownloadState(target, "instance", "q")).toMatchObject({
      phase: "failed",
      message: "Reconnect before retrying",
      busy: false,
    });
  });
});
