import { describe, expect, it } from "vitest";
import type {
  GenerationBatchChild,
  GenerationBatchStatus,
} from "../api/generationAdmission";
import {
  buildGenerationBatchStatusRequest,
  createGenerationBatchTracker,
  generationAuthorityKey,
  mergeBulkGenerationBatchResponse,
  reduceGenerationLifecycle,
} from "./generationLifecycle";

function tracker(clientBatchId = "client-1", instanceId = "instance-1") {
  return createGenerationBatchTracker({
    hostId: "host-1",
    expectedInstanceId: instanceId,
    clientBatchId,
    submittedAtMs: 1,
  });
}

function child(
  state: GenerationBatchChild["state"] = "queued",
  updatedAtMs = 20,
  overrides: Partial<GenerationBatchChild> = {},
): GenerationBatchChild {
  return {
    index: 1,
    job_id: "job-1",
    state,
    created_at_ms: 10,
    updated_at_ms: updatedAtMs,
    ...overrides,
  };
}

function batch(
  state: GenerationBatchChild["state"] = "queued",
  updatedAtMs = 20,
  overrides: Partial<GenerationBatchStatus> = {},
): GenerationBatchStatus {
  return {
    id: "batch-1",
    client_batch_id: "client-1",
    instance_id: "instance-1",
    durable: true,
    children: [child(state, updatedAtMs)],
    ...overrides,
  };
}

function admitted(state: GenerationBatchChild["state"] = "queued") {
  return reduceGenerationLifecycle(tracker(), {
    type: "batch_snapshot",
    batch: batch(state),
  });
}

function onlyJob(state: ReturnType<typeof tracker>) {
  const jobs = Object.values(state.jobs);
  expect(jobs).toHaveLength(1);
  return jobs[0]!;
}

describe("canonical durable generation lifecycle", () => {
  it("keeps ambiguous admission separate from server lifecycle", () => {
    const uncertain = reduceGenerationLifecycle(tracker(), {
      type: "admission_uncertain",
      error: "response lost",
    });
    expect(uncertain.admission).toMatchObject({
      phase: "uncertain",
      lookup: "unchecked",
    });
    expect(uncertain.jobs).toEqual({});

    const recovered = reduceGenerationLifecycle(uncertain, {
      type: "batch_snapshot",
      batch: batch("accepted"),
    });
    expect(recovered.admission).toMatchObject({
      phase: "confirmed",
      lookup: "found",
    });
    expect(onlyJob(recovered).phase).toBe("accepted");
  });

  it("keys authority by host, instance, server batch and server job", () => {
    const base = {
      hostId: "a|b",
      instanceId: "c",
      batchId: "d",
      jobId: "e",
    };
    expect(generationAuthorityKey(base)).not.toBe(
      generationAuthorityKey({
        hostId: "a",
        instanceId: "b|c",
        batchId: "d",
        jobId: "e",
      }),
    );
    expect(onlyJob(admitted()).authority).toEqual({
      hostId: "host-1",
      instanceId: "instance-1",
      batchId: "batch-1",
      jobId: "job-1",
    });
  });

  it("makes admission recovery idempotent", () => {
    const first = admitted();
    const originalJob = onlyJob(first);
    const replay = reduceGenerationLifecycle(first, {
      type: "batch_snapshot",
      batch: batch(),
    });
    expect(onlyJob(replay)).toBe(originalJob);
    expect(replay.admission.phase).toBe("confirmed");
  });

  it("does not let a late transport error revoke confirmed admission", () => {
    const confirmed = admitted();
    expect(
      reduceGenerationLifecycle(confirmed, {
        type: "admission_uncertain",
        error: "late socket close",
      }),
    ).toBe(confirmed);
    expect(
      reduceGenerationLifecycle(confirmed, {
        type: "admission_rejected",
        error: "late callback",
      }),
    ).toBe(confirmed);
  });

  it("ignores stale events and accepts forward progress at an equal timestamp", () => {
    const queued = admitted();
    const stale = reduceGenerationLifecycle(queued, {
      type: "job_event",
      instanceId: "instance-1",
      batchId: "batch-1",
      clientBatchId: "client-1",
      child: child("accepted", 19),
    });
    expect(onlyJob(stale).phase).toBe("queued");

    const running = reduceGenerationLifecycle(queued, {
      type: "job_event",
      instanceId: "instance-1",
      batchId: "batch-1",
      clientBatchId: "client-1",
      child: child("running", 20),
    });
    expect(onlyJob(running).phase).toBe("running");
  });

  it("allows restart running to queued only under newer authority", () => {
    const running = reduceGenerationLifecycle(admitted(), {
      type: "job_event",
      instanceId: "instance-1",
      batchId: "batch-1",
      clientBatchId: "client-1",
      child: child("running", 30),
      revision: 4,
    });
    const staleRestart = reduceGenerationLifecycle(running, {
      type: "job_event",
      instanceId: "instance-1",
      batchId: "batch-1",
      clientBatchId: "client-1",
      child: child("queued", 40),
      revision: 3,
    });
    expect(onlyJob(staleRestart).phase).toBe("running");

    const restarted = reduceGenerationLifecycle(running, {
      type: "job_event",
      instanceId: "instance-1",
      batchId: "batch-1",
      clientBatchId: "client-1",
      child: child("queued", 31),
      revision: 5,
    });
    expect(onlyJob(restarted)).toMatchObject({
      phase: "queued",
      version: { revision: 5 },
    });
  });

  it("makes terminal states immutable and deduplicates terminal replay", () => {
    const complete = reduceGenerationLifecycle(admitted(), {
      type: "job_event",
      instanceId: "instance-1",
      batchId: "batch-1",
      clientBatchId: "client-1",
      child: child("complete", 30, {
        completed_at_ms: 30,
        result: { filename: "first.png" },
      }),
    });
    const terminal = onlyJob(complete);
    const duplicate = reduceGenerationLifecycle(complete, {
      type: "batch_snapshot",
      batch: batch("failed", 40, {
        children: [
          child("failed", 40, {
            terminal_error: { code: "LATE" },
            result: { filename: "wrong.png" },
          }),
        ],
      }),
    });
    expect(onlyJob(duplicate)).toBe(terminal);
    expect(onlyJob(duplicate)).toMatchObject({
      phase: "complete",
      result: { filename: "first.png" },
    });
  });

  it("preserves an incapable frozen-target failure from reconciliation", () => {
    const failed = reduceGenerationLifecycle(admitted(), {
      type: "batch_snapshot",
      batch: batch("failed", 40, {
        children: [
          child("failed", 40, {
            error: "selected machine cannot run this model",
            completed_at_ms: 40,
          }),
        ],
      }),
    });
    expect(onlyJob(failed)).toMatchObject({
      phase: "failed",
      error: "selected machine cannot run this model",
      completedAtMs: 40,
    });
  });

  it("fences instance mismatches without attaching foreign jobs", () => {
    const mismatch = reduceGenerationLifecycle(tracker(), {
      type: "batch_snapshot",
      batch: batch("queued", 20, { instance_id: "instance-2" }),
    });
    expect(mismatch.jobs).toEqual({});
    expect(mismatch.reconciliation).toEqual({
      required: true,
      reason: "instance_mismatch",
    });
  });

  it("forces snapshot reconciliation after a stream gap", () => {
    const gapped = reduceGenerationLifecycle(admitted(), {
      type: "event_gap",
      instanceId: "instance-1",
    });
    expect(gapped.reconciliation).toEqual({
      required: true,
      reason: "event_gap",
    });
    const ignored = reduceGenerationLifecycle(gapped, {
      type: "job_event",
      instanceId: "instance-1",
      batchId: "batch-1",
      clientBatchId: "client-1",
      child: child("running", 30),
    });
    expect(onlyJob(ignored).phase).toBe("queued");

    const repaired = reduceGenerationLifecycle(ignored, {
      type: "batch_snapshot",
      batch: batch("running", 30),
    });
    expect(repaired.reconciliation.required).toBe(false);
    expect(onlyJob(repaired).phase).toBe("running");
  });

  it("merges bulk outcomes in tracker order and distinguishes explicit missing", () => {
    const client2 = tracker("client-2");
    const client1 = tracker("client-1");
    const otherHost = {
      ...tracker("client-other"),
      hostId: "host-2",
    };
    const merged = mergeBulkGenerationBatchResponse(
      [client2, otherHost, client1],
      "host-1",
      {
        instance_id: "instance-1",
        batches: [batch()],
        missing: { client_batch_ids: ["client-2"], batch_ids: [] },
      },
    );
    expect(merged.trackers.map((item) => item.clientBatchId)).toEqual([
      "client-2",
      "client-other",
      "client-1",
    ]);
    expect(merged.trackers[0]?.admission).toMatchObject({
      phase: "pending",
      lookup: "missing",
    });
    expect(merged.trackers[0]?.reconciliation.required).toBe(false);
    expect(merged.trackers[1]).toBe(otherHost);
    expect(merged.trackers[2]?.admission.phase).toBe("confirmed");
    expect(merged.missingClientBatchIds).toEqual(["client-2"]);
  });

  it("builds one stable deduplicated bulk request without truncation", () => {
    const known = { ...admitted(), hostId: "host-1" };
    const request = buildGenerationBatchStatusRequest(
      [
        tracker("unknown-a"),
        known,
        tracker("unknown-a"),
        { ...known },
        tracker("unknown-b"),
        { ...tracker("other"), hostId: "host-2" },
      ],
      "host-1",
    );
    expect(request).toEqual({
      client_batch_ids: ["unknown-a", "unknown-b"],
      batch_ids: ["batch-1"],
    });
  });

  it("does not confuse incomplete bulk output with explicit missing", () => {
    const merged = mergeBulkGenerationBatchResponse(
      [tracker("client-unmentioned")],
      "host-1",
      {
        instance_id: "instance-1",
        batches: [],
        missing: { client_batch_ids: [], batch_ids: [] },
      },
    );
    expect(merged.trackers[0]?.admission.lookup).toBe("unchecked");
    expect(merged.trackers[0]?.reconciliation).toEqual({
      required: true,
      reason: "incomplete_response",
    });
  });

  it("fences every same-host tracker when a bulk response comes from a replacement", () => {
    const sameHost = tracker();
    const otherHost = { ...tracker("client-2"), hostId: "host-2" };
    const merged = mergeBulkGenerationBatchResponse(
      [sameHost, otherHost],
      "host-1",
      {
        instance_id: "replacement",
        batches: [],
        missing: { client_batch_ids: ["client-1"], batch_ids: [] },
      },
    );
    expect(merged.trackers[0]?.reconciliation.reason).toBe("instance_mismatch");
    expect(merged.trackers[1]).toBe(otherHost);
  });

  it("does not persist secrets, route URLs, request bodies, or media", () => {
    const serialized = JSON.stringify(tracker());
    for (const forbidden of [
      "apiKey",
      "baseUrl",
      "requests",
      "source_image",
      "id_image",
      "upload_handle",
    ]) {
      expect(serialized).not.toContain(forbidden);
    }
  });
});
