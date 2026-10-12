import { apiFetchTo, type ApiTarget } from "./client";
import { getGenerationBatch } from "./generationAdmission";
import { getQueueJob } from "./queuePlan";

export interface QueueBatchSelection {
  instanceId: string;
  jobId: string;
  batchId: string;
  clientBatchId: string;
  liveCount: number;
}
const cancellingBatches = new Set<string>();

const liveStates = new Set(["accepted", "queued", "running", "held", "paused"]);

/** A selected job's durable authority, including work submitted by another client. */
export async function inspectQueueBatch(
  target: ApiTarget,
  instanceId: string,
  jobId: string,
): Promise<QueueBatchSelection | null> {
  if (!instanceId || !jobId) return null;
  const { job } = await getQueueJob(target, jobId);
  if (job.id !== jobId || !job.batch_id || !job.client_batch_id) return null;
  const lookup = await getGenerationBatch(target, job.batch_id);
  if (lookup.kind !== "found") return null;
  const batch = lookup.batch;
  const liveCount = batch.children.filter((child) =>
    liveStates.has(child.state),
  ).length;
  if (
    batch.id !== job.batch_id ||
    batch.instance_id !== instanceId ||
    batch.client_batch_id !== job.client_batch_id ||
    batch.children.length <= 1 ||
    !batch.children.some((child) => child.job_id === jobId) ||
    liveCount === 0
  )
    return null;
  return {
    instanceId,
    jobId,
    batchId: batch.id,
    clientBatchId: batch.client_batch_id,
    liveCount,
  };
}

/** Re-read the same authority immediately before mutation; never broaden a stale selection. */
export async function cancelInspectedQueueBatch(
  target: ApiTarget,
  selection: QueueBatchSelection,
  isCurrent: () => boolean = () => true,
): Promise<void> {
  const key = JSON.stringify([
    target.baseUrl,
    target.apiKey,
    target.connectionMode,
    selection.instanceId,
    selection.batchId,
    selection.clientBatchId,
  ]);
  if (cancellingBatches.has(key))
    throw new Error("This batch is already being cancelled.");
  cancellingBatches.add(key);
  try {
    const current = await inspectQueueBatch(
      target,
      selection.instanceId,
      selection.jobId,
    );
    if (
      !isCurrent() ||
      !current ||
      current.batchId !== selection.batchId ||
      current.clientBatchId !== selection.clientBatchId
    ) {
      throw new Error(
        "This batch or machine changed. Refresh its jobs and try again.",
      );
    }
    await apiFetchTo(
      target,
      `/api/generation-batches/${encodeURIComponent(selection.batchId)}`,
      { method: "DELETE" },
    );
  } finally {
    cancellingBatches.delete(key);
  }
}
