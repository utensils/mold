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
  preRenderTransfer: true,
  transferIdentity: "destination-instance",
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
  vi.spyOn(crypto, "randomUUID").mockReturnValue(
    clientId as `${string}-${string}-${string}-${string}-${string}`,
  );
  mocks.json.mockImplementation(async (target, path) =>
    path === "/api/status"
      ? {
          instance_id:
            target.baseUrl === source.target.baseUrl
              ? source.instanceId
              : destination.instanceId,
        }
      : path.endsWith("/reservation")
        ? null
        : path === "/api/generation-transfers/abort"
          ? {
              transfer_id: clientId,
              destination_transfer_identity: destination.instanceId,
              abort_receipt: "11111111-1111-4111-8111-111111111111",
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

describe("reserved pre-render queue transfer", () => {
  it.each(["queued", "paused"])(
    "reserves %s before reading media or admitting",
    async (state) => {
      mocks.detail.mockResolvedValue({
        job: {
          id: "job",
          state,
          batch_id: "original-batch",
          client_batch_id: "original-client",
        },
      });
      const result = await sendHeldQueueJob({
        source: { ...source, preRenderTransfer: true },
        destination,
        jobId: "job",
      });
      expect(result.sourceRemoved).toBe(true);
      const reservation = mocks.json.mock.calls.find(
        ([, path]) => path === "/api/queue/job/transfer/reserve",
      );
      expect(reservation).toBeDefined();
      expect(JSON.parse(reservation![2].body)).toMatchObject({
        transfer_id: clientId,
        destination_transfer_identity: destination.instanceId,
      });
      expect(
        mocks.json.mock.invocationCallOrder[
          mocks.json.mock.calls.indexOf(reservation!)
        ],
      ).toBeLessThan(mocks.admit.mock.invocationCallOrder[0]!);
    },
  );

  it("does not widen waiting transfer on an older source", async () => {
    mocks.detail.mockResolvedValue({
      job: {
        id: "job",
        state: "queued",
        batch_id: "batch",
        client_batch_id: "client",
      },
    });
    await expect(
      sendHeldQueueJob({ source, destination, jobId: "job" }),
    ).rejects.toThrow();
    expect(mocks.admit).not.toHaveBeenCalled();
  });
});

describe("reserved transfer failure recovery", () => {
  it("restores the source only after a durable destination abort receipt", async () => {
    mocks.admit.mockRejectedValue(new ApiError("incompatible", 422));
    await expect(
      sendHeldQueueJob({
        source: { ...source, preRenderTransfer: true },
        destination,
        jobId: "job",
      }),
    ).rejects.toThrow("incompatible");
    expect(
      mocks.json.mock.calls.some(([, path]) => path.endsWith("/transfer/seal")),
    ).toBe(true);
    expect(
      mocks.fetch.mock.calls.some(([, path]) =>
        path.endsWith("/transfer/release"),
      ),
    ).toBe(true);
    expect(
      mocks.fetch.mock.calls.some(([, path]) => path.endsWith("/complete")),
    ).toBe(false);
  });
  it("never admits after a rejected source seal", async () => {
    const json = mocks.json.getMockImplementation()!;
    mocks.json.mockImplementation(async (target, path, options) => {
      if (path.endsWith("/transfer/seal"))
        throw new ApiError("reservation released", 409);
      return json(target, path, options);
    });
    await expect(
      sendHeldQueueJob({
        source: { ...source, preRenderTransfer: true },
        destination,
        jobId: "job",
      }),
    ).rejects.toThrow("reservation released");
    expect(mocks.admit).not.toHaveBeenCalled();
  });
  it("completes the source when a competing admit beats destination abort", async () => {
    const json = mocks.json.getMockImplementation()!;
    mocks.json.mockImplementation(async (target, path, options) =>
      path === "/api/generation-transfers/abort"
        ? {
            transfer_id: clientId,
            destination_transfer_identity: destination.instanceId,
            abort_receipt: null,
          }
        : json(target, path, options),
    );
    mocks.lookup
      .mockResolvedValueOnce({ kind: "missing" })
      .mockResolvedValue({ kind: "found", batch: batch() });
    mocks.admit.mockRejectedValue(new ApiError("other client raced", 422));
    const result = await sendHeldQueueJob({
      source: { ...source, preRenderTransfer: true },
      destination,
      jobId: "job",
    });
    expect(result.sourceRemoved).toBe(true);
    expect(
      mocks.fetch.mock.calls.some(([, path]) => path.endsWith("/release")),
    ).toBe(false);
  });
  it("does not restore after a mismatched destination abort receipt", async () => {
    const json = mocks.json.getMockImplementation()!;
    mocks.json.mockImplementation(async (target, path, options) =>
      path === "/api/generation-transfers/abort"
        ? {
            transfer_id: clientId,
            destination_transfer_identity: "wrong",
            abort_receipt: "11111111-1111-4111-8111-111111111111",
          }
        : json(target, path, options),
    );
    mocks.admit.mockRejectedValue(new ApiError("refused", 422));
    await expect(
      sendHeldQueueJob({
        source: { ...source, preRenderTransfer: true },
        destination,
        jobId: "job",
      }),
    ).rejects.toThrow("identity changed");
    expect(mocks.fetch).not.toHaveBeenCalled();
  });
  it.each(["failed", "cancelled"])(
    "restores the original after accepted destination %s",
    async (state) => {
      const accepted = batch();
      accepted.children[0]!.state = state;
      mocks.lookup.mockResolvedValue({ kind: "found", batch: accepted });
      await expect(
        sendHeldQueueJob({
          source: { ...source, preRenderTransfer: true },
          destination,
          jobId: "job",
        }),
      ).rejects.toThrow("original was restored");
      expect(
        mocks.fetch.mock.calls.some(([, path]) => path.endsWith("/release")),
      ).toBe(true);
      expect(
        mocks.fetch.mock.calls.some(([, path]) => path.endsWith("/complete")),
      ).toBe(false);
    },
  );
  it("reconciles a restarted destination only with the same durable transfer identity", async () => {
    const json = mocks.json.getMockImplementation()!;
    mocks.json.mockImplementation(async (target, path, options) =>
      path.endsWith("/reservation")
        ? {
            transfer_id: clientId,
            destination_transfer_identity: "stable-destination",
          }
        : path === "/api/status" &&
            target.baseUrl === destination.target.baseUrl
          ? { instance_id: "new-process" }
          : json(target, path, options),
    );
    const accepted = { ...batch(), instance_id: "new-process" };
    mocks.lookup.mockResolvedValue({ kind: "found", batch: accepted });
    const result = await sendHeldQueueJob({
      source: { ...source, preRenderTransfer: true },
      destination: {
        ...destination,
        instanceId: "new-process",
        transferIdentity: "stable-destination",
      },
      jobId: "job",
    });
    expect(result.sourceRemoved).toBe(true);
    expect(mocks.admit).not.toHaveBeenCalled();
    await expect(
      sendHeldQueueJob({
        source: { ...source, preRenderTransfer: true },
        destination: {
          ...destination,
          instanceId: "new-process",
          transferIdentity: "replaced-owner",
        },
        jobId: "job",
      }),
    ).rejects.toThrow("reserved for another");
  });
  it("refuses missing durable destination identity before reservation", async () => {
    await expect(
      sendHeldQueueJob({
        source: { ...source, preRenderTransfer: true },
        destination: { ...destination, transferIdentity: undefined },
        jobId: "job",
      }),
    ).rejects.toThrow("identity is unavailable");
    expect(
      mocks.json.mock.calls.some(([, path]) => path.endsWith("/reserve")),
    ).toBe(false);
    expect(mocks.admit).not.toHaveBeenCalled();
  });
  it("leaves the reservation after ambiguous destination acceptance", async () => {
    const json = mocks.json.getMockImplementation()!;
    mocks.json.mockImplementation(async (target, path, options) => {
      if (path === "/api/generation-transfers/abort")
        throw new TypeError(
          "destination unreachable; acceptance not confirmed",
        );
      return json(target, path, options);
    });
    mocks.admit.mockRejectedValue(new TypeError("lost response"));
    await expect(
      sendHeldQueueJob({
        source: { ...source, preRenderTransfer: true },
        destination,
        jobId: "job",
      }),
    ).rejects.toThrow("not confirmed");
    expect(mocks.fetch).not.toHaveBeenCalled();
  });
});

describe("reservation restart reconciliation", () => {
  it("reuses persisted transfer identity after the source restarted", async () => {
    const oldClientId = clientId;
    const json = mocks.json.getMockImplementation()!;
    mocks.json.mockImplementation(async (target, path, options) =>
      path.endsWith("/reservation")
        ? {
            transfer_id: oldClientId,
            destination_transfer_identity: destination.instanceId,
          }
        : path === "/api/status" && target.baseUrl === source.target.baseUrl
          ? { instance_id: "source-restarted" }
          : json(target, path, options),
    );
    mocks.lookup.mockImplementation(async (_target, id) =>
      id === oldClientId
        ? { kind: "found", batch: batch() }
        : { kind: "missing" },
    );
    mocks.detail.mockResolvedValue({
      job: {
        id: "job",
        state: "paused",
        batch_id: "original-batch",
        client_batch_id: "original-client",
      },
    });
    const result = await sendHeldQueueJob({
      source: {
        ...source,
        instanceId: "source-restarted",
        preRenderTransfer: true,
      },
      destination,
      jobId: "job",
    });
    expect(result.sourceRemoved).toBe(true);
    expect(mocks.admit).not.toHaveBeenCalled();
    expect(
      mocks.json.mock.calls.some(([, path]) =>
        path.endsWith("/transfer/reserve"),
      ),
    ).toBe(false);
  });
  it("refuses another destination while acceptance remains unresolved", async () => {
    const json = mocks.json.getMockImplementation()!;
    mocks.json.mockImplementation(async (target, path, options) =>
      path.endsWith("/reservation")
        ? {
            transfer_id: clientId,
            destination_transfer_identity: "other-destination",
          }
        : json(target, path, options),
    );
    await expect(
      sendHeldQueueJob({
        source: { ...source, preRenderTransfer: true },
        destination,
        jobId: "job",
      }),
    ).rejects.toThrow("reserved for another");
    expect(mocks.admit).not.toHaveBeenCalled();
    expect(mocks.fetch).not.toHaveBeenCalled();
  });
});
