import { beforeEach, describe, expect, it, vi } from "vitest";
const mocks = vi.hoisted(() => ({
  json: vi.fn(),
  fetch: vi.fn(),
  detail: vi.fn(),
  lookup: vi.fn(),
  admit: vi.fn(),
  stage: vi.fn(),
  release: vi.fn(),
}));
vi.mock("./client", async (original) => ({
  ...(await original<typeof import("./client")>()),
  apiJsonTo: mocks.json,
  apiFetchTo: mocks.fetch,
}));
vi.mock("./queuePlan", () => ({ getQueueJob: mocks.detail }));
vi.mock("./generationAdmission", async (original) => ({
  ...(await original<typeof import("./generationAdmission")>()),
  lookupGenerationBatchByClientId: mocks.lookup,
  admitGenerationBatch: mocks.admit,
}));
vi.mock("./referenceUploads", () => ({
  prepareReferenceUploadBatch: mocks.stage,
}));
import { ApiError } from "./client";
import {
  queueTransferId,
  sendHeldQueueJob,
  type QueueTransferHost,
} from "./queueTransfer";
const source: QueueTransferHost = {
  id: "hal",
  label: "HAL",
  instanceId: "source-instance",
  ready: true,
  target: { baseUrl: "http://hal", apiKey: "source-key" },
};
const destination: QueueTransferHost = {
  id: "plato",
  label: "Plato",
  instanceId: "destination-instance",
  ready: true,
  target: { baseUrl: "http://plato", apiKey: "destination-key" },
};
const request = {
  model: "minimax-h3-fl2va:comfy-pruned-int8",
  prompt: "exact words",
  seed: 123,
  source_image: "AQID",
};
let clientId: string;
const batch = () => ({
  id: "destination-batch",
  client_batch_id: clientId,
  instance_id: destination.instanceId,
  durable: true,
  children: [{ job_id: "destination-job", state: "queued" }],
});
beforeEach(async () => {
  vi.clearAllMocks();
  clientId = await queueTransferId(source, "job", destination);
  mocks.json.mockImplementation(async (target, path) =>
    path === "/api/status"
      ? {
          instance_id:
            target.baseUrl === source.target.baseUrl
              ? source.instanceId
              : destination.instanceId,
        }
      : path === "/api/capabilities"
        ? {}
        : request,
  );
  mocks.detail.mockResolvedValue({
    job: {
      id: "job",
      state: "held",
      batch_id: "original-batch",
      client_batch_id: "original-client",
    },
  });
  mocks.lookup.mockResolvedValue({ kind: "missing" });
  mocks.admit.mockImplementation(async () => batch());
  mocks.stage.mockImplementation(async (options) => ({
    requests: options.requests,
    release: mocks.release,
  }));
  mocks.fetch.mockResolvedValue({});
});
describe("held queue transfer", () => {
  it("preserves the full request and removes the source only after destination acceptance", async () => {
    const result = await sendHeldQueueJob({
      source,
      destination,
      jobId: "job",
    });
    expect(mocks.admit).toHaveBeenCalledWith(
      destination.target,
      {
        client_batch_id: clientId,
        requests: [request],
      },
      undefined,
      undefined,
      destination.instanceId,
    );
    expect(mocks.fetch).toHaveBeenCalledWith(
      source.target,
      "/api/queue/job/transfer/complete",
      expect.anything(),
    );
    expect(mocks.fetch.mock.invocationCallOrder[0]).toBeGreaterThan(
      mocks.admit.mock.invocationCallOrder[0]!,
    );
    expect(result.sourceRemoved).toBe(true);
  });
  it("keeps the original and releases unused upload leases on definite rejection", async () => {
    mocks.admit.mockRejectedValue(new ApiError("invalid", 422));
    await expect(
      sendHeldQueueJob({ source, destination, jobId: "job" }),
    ).rejects.toThrow();
    expect(mocks.release).toHaveBeenCalledOnce();
    expect(mocks.fetch).not.toHaveBeenCalled();
  });
  it("reconciles a lost acceptance response without another POST", async () => {
    mocks.admit.mockRejectedValue(new TypeError("connection lost"));
    mocks.lookup
      .mockResolvedValueOnce({ kind: "missing" })
      .mockResolvedValueOnce({ kind: "found", batch: batch() });
    expect(
      (await sendHeldQueueJob({ source, destination, jobId: "job" }))
        .sourceRemoved,
    ).toBe(true);
    expect(mocks.admit).toHaveBeenCalledOnce();
  });
  it("reuses the same destination identity across app restarts and does not export again", async () => {
    mocks.lookup.mockResolvedValue({ kind: "found", batch: batch() });
    await sendHeldQueueJob({ source, destination, jobId: "job" });
    expect(mocks.admit).not.toHaveBeenCalled();
    expect(mocks.stage).not.toHaveBeenCalled();
    expect(await queueTransferId(source, "job", destination)).toBe(clientId);
    expect(
      await queueTransferId(source, "job", {
        ...destination,
        instanceId: "replacement",
      }),
    ).not.toBe(clientId);
  });
  it("does not remove the source when acceptance is unknown", async () => {
    mocks.admit.mockRejectedValue(new TypeError("connection lost"));
    await expect(
      sendHeldQueueJob({ source, destination, jobId: "job" }),
    ).rejects.toThrow("not confirmed");
    expect(mocks.fetch).not.toHaveBeenCalled();
    expect(mocks.release).not.toHaveBeenCalled();
  });
  it("reports a source race without cancelling running work or submitting twice", async () => {
    mocks.fetch.mockRejectedValue(new ApiError("not held", 409));
    const result = await sendHeldQueueJob({
      source,
      destination,
      jobId: "job",
    });
    expect(result.sourceRemoved).toBe(false);
    expect(result.message).toContain("check HAL");
    expect(mocks.admit).toHaveBeenCalledOnce();
  });
  it("fences a changed destination before sending private media", async () => {
    mocks.json.mockResolvedValue({ instance_id: "replacement" });
    await expect(
      sendHeldQueueJob({ source, destination, jobId: "job" }),
    ).rejects.toThrow("identity changed");
    expect(mocks.stage).not.toHaveBeenCalled();
    expect(mocks.admit).not.toHaveBeenCalled();
  });
});

it("uses the cross-language transfer identity without secure-context browser APIs", async () => {
  const crypto = globalThis.crypto;
  vi.stubGlobal("crypto", undefined);
  try {
    expect(
      await queueTransferId({ ...source, instanceId: "source" }, "job", {
        ...destination,
        instanceId: "dest",
      }),
    ).toBe("1bce2d47-3909-5d66-8d96-0d2d3a6ae7b3");
  } finally {
    vi.stubGlobal("crypto", crypto);
  }
});
