import { describe, expect, it } from "vitest";
import type { GenerateRequest } from "../lib/api/types";
import type { GenerationBatchStatus } from "@studio/api/generationAdmission";
import {
  buildMobileDurableHostStatusRequest,
  claimMobileDurableTerminalEffect,
  createMobileDurableGenerationRecovery,
  loadMobileDurableGenerationRecoveries,
  mergeMobileDurableHostStatus,
  mobileDurableJobs,
  mobileDurableAdmissionEffectKey,
  mobileDurableTerminalEffectsClaimed,
  reduceMobileDurableGenerationRecovery,
  resolveMobileDurableHost,
  saveMobileDurableGenerationRecoveries,
  mobileDurableGenerationRefusal,
} from "./mobileGenerationRecovery";

function request(index = 1): GenerateRequest {
  return {
    prompt: `print ${index}`,
    model: "flux-dev",
    width: 1024,
    height: 1024,
    steps: 20,
    guidance: 3.5,
    seed: index,
    batch_size: 1,
    output_format: "png",
  };
}

function recovery(clientBatchId = "client-1", requests = [request()]) {
  return createMobileDurableGenerationRecovery({
    hostId: "host-1",
    expectedInstanceId: "instance-1",
    clientBatchId,
    requests,
    submittedAtMs: 100,
  });
}

function batch(
  clientBatchId = "client-1",
  states: Array<"queued" | "running" | "complete" | "cancelled"> = ["queued"],
): GenerationBatchStatus {
  return {
    id: `server-${clientBatchId}`,
    client_batch_id: clientBatchId,
    instance_id: "instance-1",
    durable: true,
    children: states.map((state, offset) => ({
      index: offset + 1,
      job_id: `job-${clientBatchId}-${offset + 1}`,
      state,
      created_at_ms: 100,
      updated_at_ms: 110 + offset,
      ...(state === "complete"
        ? { completed_at_ms: 120, result: { filename: `${clientBatchId}-${offset + 1}.png` } }
        : {}),
    })),
  };
}

describe("mobile durable generation recovery", () => {
  const durableMedia = {
    protocol_version: 1,
    encrypted_at_rest: true,
    generate_request_media: true,
    identity: true,
    h3_references: false,
    private_h3: false,
  };
  const canonicalQueue = {
    heterogeneous_batch_max_outputs: 64,
  };
  const durableMediaV2 = {
    ...durableMedia,
    protocol_version: 2,
    private_h3: true,
  };

  it("admits singleton and Batch N while repeated submissions add no client-side queue cap", () => {
    expect(
      mobileDurableGenerationRefusal({
        queue: canonicalQueue,
        durableMedia,
        hostLabel: "Render",
        instanceId: "instance-1",
      }),
    ).toBeNull();
    const records = Array.from({ length: 257 }, (_, index) =>
      recovery(`client-${index}`, [request(index + 1)]),
    );
    expect(records).toHaveLength(257);
    const batchRequests = [request(1), request(2), request(3), request(4)];
    expect(recovery("batch", batchRequests).presentations).toHaveLength(batchRequests.length);
  });

  it("refuses a machine that has not reported its server instance", () => {
    for (const instanceId of [undefined, null, "", "   "]) {
      expect(
        mobileDurableGenerationRefusal({
          queue: canonicalQueue,
          durableMedia,
          hostLabel: "Render",
          instanceId,
        }),
      ).toBe("Render has not reported its server instance yet. Nothing was queued.");
    }
  });

  it("refuses a machine that does not advertise the durable generation queue", () => {
    expect(
      mobileDurableGenerationRefusal({
        queue: null,
        durableMedia,
        hostLabel: "Render",
        instanceId: "instance-1",
      }),
    ).toBe(
      "Render cannot queue this print: this machine does not advertise the durable generation queue. Nothing was queued.",
    );
  });

  it("admits a machine that advertises no durable request media", () => {
    // durable_media is the server's per-request refusal (a typed 503 the
    // client surfaces), never a client fence: a degraded media store still
    // admits every media-free print.
    expect(
      mobileDurableGenerationRefusal({
        queue: canonicalQueue,
        durableMedia: undefined,
        hostLabel: "Render",
        instanceId: "instance-1",
      }),
    ).toBeNull();
  });

  it("asks nothing about the request itself — the machine is the only client gate", () => {
    // The durable protocol carries source media, LoRAs, hdr_exr_dir, identity
    // photos and H3's ordered references. A client-side per-trait fence could
    // only refuse work the server would have taken, so the server's own typed
    // admission refusal is the single authority for anything it cannot take.
    expect(
      mobileDurableGenerationRefusal({
        queue: canonicalQueue,
        durableMedia: durableMediaV2,
        hostLabel: "Render",
        instanceId: "instance-1",
      }),
    ).toBeNull();
  });

  it("persists only byte-free identity and restores an ambiguous admission after restart", () => {
    let item: string | null = null;
    const storage = {
      getItem: () => item,
      setItem: (_key: string, value: string) => {
        item = value;
      },
    };
    const uncertain = {
      ...reduceMobileDurableGenerationRecovery(recovery(), {
        type: "admission_uncertain",
        error: "response lost",
      }),
      cancelRequestedChildIndexes: [1],
    };
    saveMobileDurableGenerationRecoveries(storage, [uncertain]);
    expect(item).not.toContain("print 1");
    expect(item).not.toContain("source_image");
    expect(item).not.toContain("apiKey");
    expect(item).not.toContain("baseUrl");
    const restored = loadMobileDurableGenerationRecoveries(storage);
    expect(restored[0]?.tracker.admission.phase).toBe("uncertain");
    expect(restored[0]?.cancelRequestedChildIndexes).toEqual([1]);
    expect(buildMobileDurableHostStatusRequest(restored, "host-1")).toEqual({
      client_batch_ids: ["client-1"],
    });
  });

  it.each(["QuotaExceededError", "SecurityError"])(
    "keeps a %s write failure out of the admission control flow",
    (name) => {
      const error = Object.assign(new Error("storage unavailable"), { name });
      expect(
        saveMobileDurableGenerationRecoveries(
          {
            setItem: () => {
              throw error;
            },
          },
          [recovery()],
        ),
      ).toBe(false);
    },
  );

  it("reconciles every tracked batch in one host request on wake", () => {
    const records = [recovery("a"), recovery("b"), recovery("c")];
    expect(buildMobileDurableHostStatusRequest(records, "host-1")).toEqual({
      client_batch_ids: ["a", "b", "c"],
    });
    const merged = mergeMobileDurableHostStatus(records, "host-1", {
      instance_id: "instance-1",
      batches: [batch("a"), batch("b", ["running"]), batch("c", ["complete"])],
      missing: { client_batch_ids: [], batch_ids: [] },
    });
    expect(merged.map((entry) => mobileDurableJobs(entry)[0]?.phase)).toEqual([
      "queued",
      "running",
      "complete",
    ]);
  });

  it("fences instance mismatch and event gaps until an authoritative snapshot", () => {
    const admitted = reduceMobileDurableGenerationRecovery(recovery(), {
      type: "batch_snapshot",
      batch: batch(),
    });
    const mismatch = reduceMobileDurableGenerationRecovery(admitted, {
      type: "event_gap",
      instanceId: "replacement",
    });
    expect(mismatch.tracker.reconciliation.reason).toBe("instance_mismatch");
    const gap = reduceMobileDurableGenerationRecovery(admitted, {
      type: "event_gap",
      instanceId: "instance-1",
    });
    expect(gap.tracker.reconciliation.reason).toBe("event_gap");
  });

  it("preserves a completion that races cancellation", () => {
    const completed = reduceMobileDurableGenerationRecovery(recovery(), {
      type: "batch_snapshot",
      batch: batch("client-1", ["complete"]),
    });
    const lateCancelled = reduceMobileDurableGenerationRecovery(completed, {
      type: "batch_snapshot",
      batch: batch("client-1", ["cancelled"]),
    });
    expect(mobileDurableJobs(lateCancelled)[0]?.phase).toBe("complete");
  });

  it("claims Photos and viewer effects once across duplicate terminal snapshots", () => {
    const completed = reduceMobileDurableGenerationRecovery(recovery(), {
      type: "batch_snapshot",
      batch: batch("client-1", ["complete"]),
    });
    const key = mobileDurableJobs(completed)[0]!.key;
    const first = claimMobileDurableTerminalEffect(completed, key, "photos");
    expect(first.claimed).toBe(true);
    expect(claimMobileDurableTerminalEffect(first.recovery, key, "photos").claimed).toBe(false);
    expect(claimMobileDurableTerminalEffect(first.recovery, key, "viewer").claimed).toBe(true);
  });

  it("releases only fully claimed terminal recovery records from durable storage", () => {
    const completed = reduceMobileDurableGenerationRecovery(recovery(), {
      type: "batch_snapshot",
      batch: batch("client-1", ["complete"]),
    });
    expect(mobileDurableTerminalEffectsClaimed(completed)).toBe(false);
    const key = mobileDurableJobs(completed)[0]!.key;
    const viewer = claimMobileDurableTerminalEffect(completed, key, "viewer").recovery;
    const photos = claimMobileDurableTerminalEffect(viewer, key, "photos").recovery;
    const gallery = claimMobileDurableTerminalEffect(photos, key, "gallery").recovery;
    expect(mobileDurableTerminalEffectsClaimed(gallery)).toBe(true);

    let rejected = reduceMobileDurableGenerationRecovery(recovery("rejected"), {
      type: "admission_rejected",
      error: "invalid request",
    });
    expect(mobileDurableTerminalEffectsClaimed(rejected)).toBe(false);
    rejected = claimMobileDurableTerminalEffect(
      rejected,
      mobileDurableAdmissionEffectKey(rejected),
      "viewer",
    ).recovery;
    expect(mobileDurableTerminalEffectsClaimed(rejected)).toBe(true);
  });

  it("resolves exact-host media credentials only after the instance fence", () => {
    const record = recovery();
    const exact = {
      id: "host-1",
      instanceId: "instance-1",
      baseUrl: "https://exact-host",
      apiKey: "native-secret",
    };
    expect(resolveMobileDurableHost(record, [exact])).toBe(exact);
    expect(resolveMobileDurableHost(record, [{ ...exact, instanceId: "replacement" }])).toBeNull();
    expect(JSON.stringify(record)).not.toContain("native-secret");
  });
});
