import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, type ApiTarget } from "./client";
import {
  canonicalGenerationBatchLimit,
  chunkGenerationBatchRequests,
  admitGenerationBatch,
  getGenerationBatch,
  isDefiniteGenerationAdmissionRejection,
  lookupGenerationBatchByClientId,
  parseGenerationBatchStatus,
  parseGenerationBatchStatusResponse,
  reconcileGenerationBatches,
} from "./generationAdmission";

const target: ApiTarget = {
  baseUrl: "http://render-box:7680",
  apiKey: "secret",
};

function batch(overrides: Record<string, unknown> = {}) {
  return {
    id: "batch-1",
    client_batch_id: "client-1",
    instance_id: "instance-1",
    durable: true,
    children: [
      {
        index: 1,
        job_id: "job-1",
        state: "queued",
        created_at_ms: 10,
        updated_at_ms: 11,
      },
    ],
    ...overrides,
  };
}

afterEach(() => vi.unstubAllGlobals());

describe("durable generation admission API", () => {
  it("distinguishes definite rejection from ambiguous delivery failures", () => {
    for (const status of [400, 401, 404, 409, 422, 426]) {
      expect(
        isDefiniteGenerationAdmissionRejection(
          new ApiError("rejected", status),
        ),
      ).toBe(true);
    }
    for (const status of [408, 425, 429, 500, 503]) {
      expect(
        isDefiniteGenerationAdmissionRejection(
          new ApiError("uncertain", status),
        ),
      ).toBe(false);
    }
    expect(
      isDefiniteGenerationAdmissionRejection(
        new TypeError("connection closed"),
      ),
    ).toBe(false);
    expect(
      isDefiniteGenerationAdmissionRejection(new SyntaxError("invalid JSON")),
    ).toBe(false);
  });

  it("parses explicit retry authority without changing the child lifecycle", () => {
    const parsed = parseGenerationBatchStatus(
      batch({
        children: [
          {
            index: 1,
            job_id: "job-1",
            state: "held",
            retryable: true,
            created_at_ms: 10,
            updated_at_ms: 11,
          },
        ],
      }),
    );
    expect(parsed.children[0]).toMatchObject({
      state: "held",
      retryable: true,
    });
  });

  it("admits singleton batches without adding an invented client limit", async () => {
    let requestBody: unknown;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        requestBody = JSON.parse(String(init?.body));
        return Response.json(batch(), { status: 202 });
      }),
    );
    const status = await admitGenerationBatch(target, {
      client_batch_id: "client-1",
      requests: [{ prompt: "one print" }],
    });
    expect(requestBody).toEqual({
      client_batch_id: "client-1",
      requests: [{ prompt: "one print" }],
    });
    expect(status.id).toBe("batch-1");
  });

  it("recovers an ambiguous admission through the encoded client id", async () => {
    let url = "";
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        url = String(input);
        return Response.json(batch());
      }),
    );
    await expect(
      lookupGenerationBatchByClientId(target, "client/id with space"),
    ).resolves.toMatchObject({ kind: "found", batch: { id: "batch-1" } });
    expect(url).toBe(
      "http://render-box:7680/api/generation-batches/by-client/client%2Fid%20with%20space",
    );
  });

  it("keeps authoritative missing separate from transport failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json(
          { error: "not found" },
          { status: 404, statusText: "Not Found" },
        ),
      ),
    );
    await expect(
      lookupGenerationBatchByClientId(target, "missing"),
    ).resolves.toEqual({ kind: "missing" });
    await expect(getGenerationBatch(target, "missing")).resolves.toEqual({
      kind: "missing",
    });

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("network down");
      }),
    );
    await expect(
      lookupGenerationBatchByClientId(target, "unknown"),
    ).rejects.toThrow("network down");
  });

  it("posts the complete caller-selected reconciliation set verbatim", async () => {
    let captured: { url: string; body: unknown } | null = null;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        captured = {
          url: String(input),
          body: JSON.parse(String(init?.body)),
        };
        return Response.json({
          instance_id: "instance-1",
          batches: [batch()],
          missing: { client_batch_ids: ["client-2"], batch_ids: ["batch-3"] },
        });
      }),
    );
    const response = await reconcileGenerationBatches(target, {
      client_batch_ids: ["client-1", "client-2"],
      batch_ids: ["batch-3"],
    });
    expect(captured).toEqual({
      url: "http://render-box:7680/api/generation-batches/status",
      body: {
        client_batch_ids: ["client-1", "client-2"],
        batch_ids: ["batch-3"],
      },
    });
    expect(response.missing).toEqual({
      client_batch_ids: ["client-2"],
      batch_ids: ["batch-3"],
    });
  });

  it("validates authority, timestamps, results, and child uniqueness", () => {
    expect(
      parseGenerationBatchStatus(
        batch({
          children: [
            {
              index: 1,
              job_id: "job-1",
              state: "complete",
              created_at_ms: 10,
              updated_at_ms: 20,
              completed_at_ms: 20,
              result: {
                filename: "print.png",
                original_filename: "original.png",
              },
            },
          ],
        }),
      ).children[0]?.result,
    ).toEqual({ filename: "print.png", original_filename: "original.png" });
    expect(() =>
      parseGenerationBatchStatus(
        batch({
          children: [
            batch().children[0],
            { ...batch().children[0], state: "running" },
          ],
        }),
      ),
    ).toThrow("duplicates a child identity");
    expect(() =>
      parseGenerationBatchStatus({ ...batch(), durable: false }),
    ).toThrow("incompatible");
  });

  it("rejects mixed-instance and duplicate bulk responses", () => {
    expect(() =>
      parseGenerationBatchStatusResponse({
        instance_id: "instance-1",
        batches: [batch({ instance_id: "instance-2" })],
        missing: { client_batch_ids: [], batch_ids: [] },
      }),
    ).toThrow("mixes server instances");
    expect(() =>
      parseGenerationBatchStatusResponse({
        instance_id: "instance-1",
        batches: [batch(), batch()],
        missing: { client_batch_ids: [], batch_ids: [] },
      }),
    ).toThrow("duplicates a batch");
    expect(() =>
      parseGenerationBatchStatusResponse({
        instance_id: "instance-1",
        batches: [batch(), batch({ id: "batch-2" })],
        missing: { client_batch_ids: [], batch_ids: [] },
      }),
    ).toThrow("duplicates a client batch");
  });

  it("derives the canonical limit and chunks without changing request order", () => {
    expect(
      canonicalGenerationBatchLimit({ heterogeneous_batch_max_outputs: 2 }),
    ).toBe(2);
    expect(chunkGenerationBatchRequests([1, 2, 3, 4, 5], 2)).toEqual([
      [1, 2],
      [3, 4],
      [5],
    ]);
    expect(() => chunkGenerationBatchRequests([1], 0)).toThrow(
      "positive integer",
    );
  });

  it("refuses a malformed or absent batch limit rather than inventing one", () => {
    for (const queue of [
      undefined,
      null,
      {},
      { heterogeneous_batch_max_outputs: null },
      { heterogeneous_batch_max_outputs: 0 },
      { heterogeneous_batch_max_outputs: 1.5 },
      { heterogeneous_batch_max_outputs: Number.NaN },
    ]) {
      expect(canonicalGenerationBatchLimit(queue)).toBeNull();
    }
  });
});
