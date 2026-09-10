import { sha256 } from "@noble/hashes/sha2.js";
import { ApiError, apiFetchTo, apiJsonTo, type ApiTarget } from "./client";
import { getQueueJob } from "./queuePlan";
import {
  admitGenerationBatch,
  lookupGenerationBatchByClientId,
  isDefiniteGenerationAdmissionRejection,
  type GenerationBatchStatus,
} from "./generationAdmission";
import {
  prepareReferenceUploadBatch,
  type ReferenceUploadCapabilities,
  type ReferenceUploadRequest,
} from "./referenceUploads";

export interface QueueTransferHost {
  id: string;
  label: string;
  instanceId: string;
  target: ApiTarget;
  ready: boolean;
  gpuCount?: number | undefined;
  queueDepth?: number | null | undefined;
}

/** Stable across retries and app restarts. Never contains a credential. */
export async function queueTransferId(
  source: QueueTransferHost,
  jobId: string,
  destination: QueueTransferHost,
): Promise<string> {
  const digest = sha256(
    new TextEncoder().encode(
      JSON.stringify([
        "mold.queue-transfer.v1",
        source.instanceId,
        jobId,
        destination.instanceId,
      ]),
    ),
  );
  digest[6] = (digest[6]! & 15) | 80;
  digest[8] = (digest[8]! & 63) | 128;
  const hex = Array.from(digest.slice(0, 16), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

export async function sendHeldQueueJob(options: {
  source: QueueTransferHost;
  destination: QueueTransferHost;
  jobId: string;
  onProgress?: (message: string) => void;
}): Promise<{
  batch: GenerationBatchStatus;
  sourceRemoved: boolean;
  message: string;
}> {
  const { source, destination, jobId } = options;
  if (
    !source.ready ||
    !destination.ready ||
    source.instanceId === destination.instanceId
  ) {
    throw new Error("Choose another connected machine.");
  }
  const [sourceStatus, destinationStatus] = await Promise.all([
    apiJsonTo<{ instance_id: string }>(source.target, "/api/status"),
    apiJsonTo<{ instance_id: string }>(destination.target, "/api/status"),
  ]);
  if (
    sourceStatus.instance_id !== source.instanceId ||
    destinationStatus.instance_id !== destination.instanceId
  ) {
    throw new Error(
      "A machine's identity changed. Refresh the machines and try again.",
    );
  }
  const clientId = await queueTransferId(source, jobId, destination);
  options.onProgress?.(`Checking ${destination.label}…`);
  let lookup = await lookupGenerationBatchByClientId(
    destination.target,
    clientId,
  );
  let authority: Record<string, string> | null = null;
  try {
    const { job } = await getQueueJob(source.target, jobId);
    if (job.state !== "held" || !job.batch_id || !job.client_batch_id) {
      throw new Error(
        "This job is no longer held. Refresh the queue before sending it.",
      );
    }
    authority = {
      instance_id: source.instanceId,
      job_id: jobId,
      batch_id: job.batch_id,
      client_batch_id: job.client_batch_id,
    };
  } catch (error) {
    if (lookup.kind !== "found") throw error;
    // A previous successful transfer may already have removed the original.
    if (!(error instanceof ApiError && error.status === 404)) throw error;
  }
  let batch: GenerationBatchStatus;
  if (lookup.kind === "found") {
    batch = lookup.batch;
  } else {
    options.onProgress?.("Reading original settings and reference media…");
    const request = await apiJsonTo<ReferenceUploadRequest>(
      source.target,
      `/api/queue/${encodeURIComponent(jobId)}/transfer`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(authority),
      },
    );
    const capabilities = await apiJsonTo<{
      reference_uploads?: ReferenceUploadCapabilities;
    }>(destination.target, "/api/capabilities");
    options.onProgress?.(`Sending to ${destination.label}…`);
    const upload = await prepareReferenceUploadBatch({
      target: destination.target,
      expectedInstanceId: destination.instanceId,
      capabilities: capabilities.reference_uploads,
      requests: [request],
    });
    try {
      batch = await admitGenerationBatch(
        destination.target,
        {
          client_batch_id: clientId,
          requests: upload.requests,
        },
        undefined,
        undefined,
        destination.instanceId,
      );
    } catch (error) {
      if (isDefiniteGenerationAdmissionRejection(error)) {
        await upload.release();
        throw error;
      }
      lookup = await lookupGenerationBatchByClientId(
        destination.target,
        clientId,
      );
      if (lookup.kind !== "found") {
        throw new Error(
          `Acceptance by ${destination.label} is not confirmed. The original remains held. Retry this same destination to check safely.`,
        );
      }
      batch = lookup.batch;
    }
  }
  if (
    batch.instance_id !== destination.instanceId ||
    batch.client_batch_id !== clientId ||
    !batch.durable ||
    batch.children.length !== 1
  ) {
    throw new Error(
      "The destination returned an unexpected job identity. The original remains held.",
    );
  }
  if (
    batch.children[0]!.state === "failed" ||
    batch.children[0]!.state === "cancelled"
  ) {
    throw new Error(
      `The destination job ${batch.children[0]!.state}. The original remains held; choose another machine or inspect the destination.`,
    );
  }
  if (authority) {
    options.onProgress?.("Removing the held original…");
    try {
      await apiFetchTo(
        source.target,
        `/api/queue/${encodeURIComponent(jobId)}/transfer/complete`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(authority),
        },
      );
    } catch {
      return {
        batch,
        sourceRemoved: false,
        message: `Sent to ${destination.label}. The original could not be removed; check ${source.label} before retrying it.`,
      };
    }
  }
  return {
    batch,
    sourceRemoved: true,
    message: `Sent to ${destination.label}. The original was removed from ${source.label}'s queue.`,
  };
}
