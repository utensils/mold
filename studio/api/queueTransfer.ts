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
  preRenderTransfer?: boolean | undefined;
  transferIdentity?: string | undefined;
  generates?: boolean | undefined;
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
  if (
    source.preRenderTransfer === true &&
    destination.preRenderTransfer !== true
  ) {
    throw new Error(
      "Update the destination server before moving waiting jobs; it must support safe transfer recovery.",
    );
  }
  if (
    source.transferIdentity &&
    source.transferIdentity === destination.transferIdentity
  )
    throw new Error(
      "Choose another connected machine; these addresses share the same queue owner.",
    );
  if (
    source.preRenderTransfer === true &&
    !destination.transferIdentity?.trim()
  )
    throw new Error(
      "Update or refresh the destination before moving jobs; its durable transfer identity is unavailable.",
    );
  const destinationBinding =
    source.preRenderTransfer === true
      ? destination.transferIdentity!
      : destination.instanceId;
  destination.transferIdentity ?? destination.instanceId;
  let clientId = await queueTransferId(source, jobId, destination);
  if (source.preRenderTransfer === true) {
    const prior = await apiJsonTo<{
      transfer_id?: string;
      destination_transfer_identity?: string;
    } | null>(
      source.target,
      `/api/queue/${encodeURIComponent(jobId)}/transfer/reservation`,
    );
    if (prior?.transfer_id) {
      if (prior.destination_transfer_identity !== destinationBinding) {
        throw new Error(
          "The original is reserved for another machine. Retry that destination to reconcile acceptance before moving it elsewhere.",
        );
      }
      clientId = prior.transfer_id;
    } else {
      clientId = crypto.randomUUID();
    }
  }
  options.onProgress?.(`Checking ${destination.label}…`);
  let lookup = await lookupGenerationBatchByClientId(
    destination.target,
    clientId,
  );
  let authority: Record<string, string> | null = null;
  let reserved = false;
  const reservation = () => ({
    ...authority,
    transfer_id: clientId,
    destination_transfer_identity: destinationBinding,
  });
  const releaseReservation = async (abortReceipt?: string) => {
    if (!reserved && !abortReceipt) return;
    await apiFetchTo(
      source.target,
      `/api/queue/${encodeURIComponent(jobId)}/transfer/release`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...reservation(),
          ...(abortReceipt ? { abort_receipt: abortReceipt } : {}),
        }),
      },
    );
  };
  try {
    const { job } = await getQueueJob(source.target, jobId);
    if (
      !(
        job.state === "held" ||
        (source.preRenderTransfer === true &&
          ["queued", "paused"].includes(job.state))
      ) ||
      (source.preRenderTransfer !== true &&
        (!job.batch_id || !job.client_batch_id))
    ) {
      throw new Error(
        "This job is already rendering or is no longer movable. Refresh the queue before sending it.",
      );
    }
    authority = {
      instance_id: source.instanceId,
      job_id: jobId,
      batch_id: job.batch_id ?? "",
      client_batch_id: job.client_batch_id ?? "",
      ...(source.preRenderTransfer === true
        ? {
            transfer_id: clientId,
            destination_transfer_identity: destinationBinding,
          }
        : {}),
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
    if (source.preRenderTransfer === true) {
      if (!authority)
        throw new Error("Source authority is unavailable. Refresh and retry.");
      options.onProgress?.("Reserving the original on its machine…");
      await apiJsonTo(
        source.target,
        `/api/queue/${encodeURIComponent(jobId)}/transfer/reserve`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(reservation()),
        },
      );
      reserved = true;
      authority.transfer_id = clientId;
      authority.destination_transfer_identity = destinationBinding;
    }
    options.onProgress?.("Reading original settings and reference media…");
    let upload: Awaited<ReturnType<typeof prepareReferenceUploadBatch>>;
    try {
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
      upload = await prepareReferenceUploadBatch({
        target: destination.target,
        expectedInstanceId: destination.instanceId,
        capabilities: capabilities.reference_uploads,
        requests: [request],
      });
    } catch (error) {
      await releaseReservation();
      throw error;
    }
    if (reserved) {
      await apiJsonTo(
        source.target,
        `/api/queue/${encodeURIComponent(jobId)}/transfer/seal`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...authority,
            transfer_id: clientId,
            destination_transfer_identity: destinationBinding,
          }),
        },
      );
    }
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
      if (reserved || isDefiniteGenerationAdmissionRejection(error)) {
        if (reserved) {
          const aborted = await apiJsonTo<{
            transfer_id: string;
            destination_transfer_identity: string;
            abort_receipt?: string | null;
          }>(destination.target, "/api/generation-transfers/abort", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              transfer_id: clientId,
              destination_transfer_identity: destinationBinding,
            }),
          });
          if (
            aborted.transfer_id !== clientId ||
            aborted.destination_transfer_identity !== destinationBinding
          )
            throw new Error(
              "Destination identity changed. Retry the same destination to reconcile acceptance.",
            );
          if (aborted.abort_receipt) {
            await upload.release();
            await releaseReservation(aborted.abort_receipt);
            throw error;
          }
          lookup = await lookupGenerationBatchByClientId(
            destination.target,
            clientId,
          );
          if (lookup.kind === "found") {
            batch = lookup.batch;
          } else
            throw new Error(
              "Destination acceptance is not confirmed. Retry Move to the same destination; the original remains reserved.",
            );
        } else {
          await upload.release();
          throw error;
        }
      }
      lookup = await lookupGenerationBatchByClientId(
        destination.target,
        clientId,
      );
      if (lookup.kind !== "found") {
        throw new Error(
          `Acceptance by ${destination.label} is not confirmed. The original is retained. Retry this same destination to check safely.`,
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
      "The destination returned an unexpected job identity. The original is retained.",
    );
  }
  if (
    batch.children[0]!.state === "failed" ||
    batch.children[0]!.state === "cancelled"
  ) {
    if (source.preRenderTransfer === true && authority) {
      const aborted = await apiJsonTo<{
        transfer_id: string;
        destination_transfer_identity: string;
        abort_receipt?: string | null;
      }>(destination.target, "/api/generation-transfers/abort", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transfer_id: clientId,
          destination_transfer_identity: destinationBinding,
        }),
      });
      if (
        aborted.transfer_id !== clientId ||
        aborted.destination_transfer_identity !== destinationBinding ||
        !aborted.abort_receipt
      )
        throw new Error(
          "Destination terminal failure could not be reconciled. Retry the same destination; original remains reserved.",
        );
      await releaseReservation(aborted.abort_receipt);
      throw new Error(
        `The destination job ${batch.children[0]!.state}. The original was restored; choose another machine or retry.`,
      );
    }
    throw new Error(
      `The destination job ${batch.children[0]!.state}. The original is retained; choose another machine or inspect the destination.`,
    );
  }
  if (authority) {
    options.onProgress?.("Removing the original…");
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
