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
  useMobileDurableGenerationLifecycle,
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

  it("admits singleton and Batch N while repeated submissions add no client-side queue cap", () => {
    const queue = { heterogeneous_batch: true, durable_batch_outcomes: true };
    expect(
      useMobileDurableGenerationLifecycle({ queue, requests: [request()], chain: false }),
    ).toBe(true);
    const records = Array.from({ length: 257 }, (_, index) =>
      recovery(`client-${index}`, [request(index + 1)]),
    );
    expect(records).toHaveLength(257);
    const batchRequests = [request(1), request(2), request(3), request(4)];
    expect(recovery("batch", batchRequests).presentations).toHaveLength(batchRequests.length);
  });

  it("routes supported media durably only with the exact v1 host capability", () => {
    const queue = { heterogeneous_batch: true, durable_batch_outcomes: true };
    for (const media of [
      { id_image: "face" },
      { id_images: ["face-a", "face-b"] },
      { source_image: "source" },
      { mask_image: "mask" },
      { control_image: "control" },
      { edit_images: ["edit"] },
      { keyframes: [{ frame: 0, image: "keyframe" }] },
      { audio_file: "audio" },
      { audio_file_path: "/host/audio.wav" },
      { source_video: "video" },
      { source_video_path: "/host/source.mp4" },
      { extend_video_path: "/host/video.mp4" },
    ]) {
      expect(
        useMobileDurableGenerationLifecycle({
          queue,
          durableMedia,
          requests: [{ ...request(), ...media } as GenerateRequest],
          chain: false,
        }),
      ).toBe(true);
      expect(
        useMobileDurableGenerationLifecycle({
          queue,
          durableMedia: undefined,
          requests: [{ ...request(), ...media } as GenerateRequest],
          chain: false,
        }),
      ).toBe(false);
    }
    for (const excluded of [
      { references: [{ image: { authority: "inline", data: "h3" } }] },
      { source_image: "source", lora: { path: "adapter", scale: 1 } },
      { source_image: "source", loras: [] },
      { hdr_exr_dir: "/host/exr" },
    ])
      expect(
        useMobileDurableGenerationLifecycle({
          queue,
          durableMedia,
          requests: [{ ...request(), ...excluded } as GenerateRequest],
          chain: false,
        }),
      ).toBe(false);
    expect(useMobileDurableGenerationLifecycle({ queue, requests: [request()], chain: true })).toBe(
      false,
    );
    expect(
      useMobileDurableGenerationLifecycle({
        queue,
        durableMedia,
        requests: [{ ...request(), model: "minimax-h3-fl2va:official-bf16" }],
        chain: false,
      }),
    ).toBe(false);
    expect(
      useMobileDurableGenerationLifecycle({
        queue,
        durableMedia,
        requests: [{ ...request(), model: "hf:opaque-h3-checkpoint" }],
        chain: false,
        modelFamily: "minimax-h3",
      }),
    ).toBe(false);
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
