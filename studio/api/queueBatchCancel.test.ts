import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiFetchTo } from "./client";
import {
  getGenerationBatch,
  type GenerationBatchStatus,
} from "./generationAdmission";
import { getQueueJob } from "./queuePlan";
import {
  inspectQueueBatch,
  cancelInspectedQueueBatch,
} from "./queueBatchCancel";

vi.mock("./client", () => ({ apiFetchTo: vi.fn(), apiJsonTo: vi.fn() }));
vi.mock("./queuePlan", () => ({ getQueueJob: vi.fn() }));
vi.mock("./generationAdmission", () => ({ getGenerationBatch: vi.fn() }));
const target = { baseUrl: "http://fixture", apiKey: null };
const batch: GenerationBatchStatus = {
  id: "batch/1",
  client_batch_id: "client-1",
  instance_id: "instance-1",
  durable: true,
  children: [
    {
      job_id: "job-1",
      state: "running",
      index: 1,
      created_at_ms: 1,
      updated_at_ms: 2,
    },
    {
      job_id: "job-2",
      state: "held",
      index: 2,
      created_at_ms: 1,
      updated_at_ms: 2,
    },
  ],
};

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(getQueueJob).mockResolvedValue({
    job: {
      id: "job-1",
      model: "video",
      state: "running",
      position: 0,
      started_at_unix_ms: 0,
      batch_id: batch.id,
      client_batch_id: batch.client_batch_id,
    },
  });
  vi.mocked(getGenerationBatch).mockResolvedValue({
    kind: "found",
    batch: batch,
  });
  vi.mocked(apiFetchTo).mockResolvedValue(new Response("{}"));
});

describe("queue batch cancellation", () => {
  it("cancels the selected video's entire durable batch using its exact route", async () => {
    const selection = await inspectQueueBatch(target, "instance-1", "job-1");
    expect(selection?.liveCount).toBe(2);
    await cancelInspectedQueueBatch(target, selection!);
    expect(apiFetchTo).toHaveBeenCalledWith(
      target,
      "/api/generation-batches/batch%2F1",
      { method: "DELETE" },
    );
  });
  it.each(["instance", "client", "job", "singleton", "terminal"])(
    "refuses a mismatched or ineligible %s batch",
    async (fault) => {
      const changed = structuredClone(batch);
      if (fault === "instance") changed.instance_id = "other";
      if (fault === "client") changed.client_batch_id = "other";
      if (fault === "job") changed.children[0]!.job_id = "other";
      if (fault === "singleton") changed.children.pop();
      if (fault === "terminal")
        changed.children.forEach((child) => {
          child.state = "complete";
        });
      vi.mocked(getGenerationBatch).mockResolvedValue({
        kind: "found",
        batch: changed,
      });
      expect(await inspectQueueBatch(target, "instance-1", "job-1")).toBeNull();
      expect(apiFetchTo).not.toHaveBeenCalled();
    },
  );
  it("rechecks the original batch before cancelling, even when the same job is regrouped", async () => {
    const selection = await inspectQueueBatch(target, "instance-1", "job-1");
    vi.mocked(getQueueJob).mockResolvedValue({
      job: {
        id: "job-1",
        model: "video",
        state: "held",
        position: 0,
        started_at_unix_ms: 0,
        batch_id: "replacement",
        client_batch_id: "replacement",
      },
    });
    await expect(cancelInspectedQueueBatch(target, selection!)).rejects.toThrow(
      /changed/,
    );
    expect(apiFetchTo).not.toHaveBeenCalled();
  });
  it("stops when the caller's route or lifecycle changes during verification", async () => {
    const selection = await inspectQueueBatch(target, "instance-1", "job-1");
    let current = true;
    vi.mocked(getGenerationBatch).mockImplementation(async () => {
      current = false;
      return { kind: "found", batch: batch };
    });
    await expect(
      cancelInspectedQueueBatch(target, selection!, () => current),
    ).rejects.toThrow(/changed/);
    expect(apiFetchTo).not.toHaveBeenCalled();
  });
});
