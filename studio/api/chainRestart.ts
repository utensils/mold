import { apiFetchTo, apiJsonTo } from "./client";
import {
  assertChainOwner,
  type ChainAdmission,
  type ChainOwner,
} from "./chainJobs";

export interface ChainRestartOptions {
  source_instance_id: string;
  source_revision: string;
  restart: { supported: boolean; reason?: string };
  send: { supported: boolean; reason?: string };
  discard?: { supported: boolean; reason?: string };
}
export interface ChainRestartExport {
  destination_instance_id: string;
  source_instance_id: string;
  source_revision: string;
  /** Wire-ready private payload. Never parse/re-encode full-range u64 seeds. */
  request_json: string;
}
export interface ChainRestartAdmission extends ChainAdmission {
  sourceJobId: string;
  sourceHostId: string;
  sourceInstanceId: string;
  sourceRevision: string;
  destinationInstanceId: string;
}
export function chainRestartOptions(
  owner: ChainOwner,
  id: string,
): Promise<ChainRestartOptions> {
  return apiJsonTo(
    owner.target,
    `/api/chain-jobs/${encodeURIComponent(id)}/restart-options`,
  );
}
export async function exportChainRestart(
  source: ChainOwner,
  jobId: string,
  options: ChainRestartOptions,
  destinationInstanceId: string,
): Promise<ChainRestartExport> {
  if (!source.instanceId || source.instanceId !== options.source_instance_id)
    throw new Error(
      "The source machine installation changed. Reopen this job.",
    );
  const exported = await apiJsonTo<ChainRestartExport>(
    source.target,
    `/api/chain-jobs/${encodeURIComponent(jobId)}/restart-export`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_instance_id: options.source_instance_id,
        source_revision: options.source_revision,
        destination_instance_id: destinationInstanceId,
      }),
    },
  );
  if (
    exported.destination_instance_id !== destinationInstanceId ||
    exported.source_instance_id !== options.source_instance_id ||
    exported.source_revision !== options.source_revision
  )
    throw new Error(
      "This video changed while its inputs were being read. Reopen the job.",
    );
  if (
    typeof exported.request_json !== "string" ||
    !exported.request_json.trim()
  )
    throw new Error(
      "Update the source machine before starting this video again.",
    );
  return exported;
}

/** Older hosts ignore unknown headers, so status alone is not enough. */
export async function assertChainRestartDestination(
  destination: ChainOwner,
): Promise<void> {
  await assertChainOwner(destination);
  const capabilities = await apiJsonTo<{ queue?: { chain_restart?: boolean } }>(
    destination.target,
    "/api/capabilities",
  );
  if (capabilities.queue?.chain_restart !== true)
    throw new Error(
      "Update the destination machine before starting this video again.",
    );
}
/** A new attempt leaves the original intact. Storage failure prevents POST;
 * an ambiguous reply retains the exact destination operation for reconciliation. */
export async function admitChainRestart(options: {
  source: ChainOwner;
  sourceJobId: string;
  options: ChainRestartOptions;
  destination: ChainOwner;
  requestJson: string;
  operationId: string;
  persist: (admission: ChainRestartAdmission) => void;
  isCurrent?: () => boolean;
}): Promise<ChainRestartAdmission> {
  const { source, destination } = options;
  const permission =
    source.instanceId === destination.instanceId
      ? options.options.restart
      : options.options.send;
  if (!permission.supported)
    throw new Error(
      permission.reason ?? "This video cannot be started again here.",
    );
  if (
    !source.instanceId ||
    source.instanceId !== options.options.source_instance_id
  )
    throw new Error(
      "The source machine installation changed. Reopen this job.",
    );
  await assertChainRestartDestination(destination);
  if (options.isCurrent?.() === false)
    throw new Error(
      "The source or destination machine changed before admission.",
    );
  const admission: ChainRestartAdmission = {
    operationId: options.operationId,
    hostId: destination.hostId,
    instanceId: destination.instanceId!,
    baseUrl: destination.target.baseUrl,
    ...(destination.target.connectionMode
      ? { connectionMode: destination.target.connectionMode }
      : {}),
    createdAt: Date.now(),
    sourceJobId: options.sourceJobId,
    sourceHostId: source.hostId,
    sourceInstanceId: source.instanceId,
    sourceRevision: options.options.source_revision,
    destinationInstanceId: destination.instanceId!,
  };
  options.persist(admission);
  await apiJsonTo(destination.target, "/api/chain-jobs", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-mold-operation-id": options.operationId,
      "x-mold-destination-instance": destination.instanceId!,
    },
    body: options.requestJson,
  });
  return admission;
}

/** This new route cannot degrade to a legacy DELETE that ignores revisions. */
export async function discardChainRecovery(
  source: ChainOwner,
  jobId: string,
  options: ChainRestartOptions,
  isCurrent: () => boolean = () => true,
): Promise<void> {
  if (options.discard?.supported !== true)
    throw new Error(
      options.discard?.reason ??
        "Update this machine before deleting video recovery data.",
    );
  if (!source.instanceId || source.instanceId !== options.source_instance_id)
    throw new Error(
      "The source machine installation changed. Reopen this video.",
    );
  await assertChainOwner(source);
  if (!isCurrent())
    throw new Error(
      "The video or machine changed before deleting recovery data.",
    );
  await apiFetchTo(
    source.target,
    `/api/chain-jobs/${encodeURIComponent(jobId)}/discard`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_instance_id: options.source_instance_id,
        source_revision: options.source_revision,
      }),
    },
  );
}

/** New-attempt cancellation must never fall back to a legacy host that ignores
 * the installation header. The exact operation remains recoverable on refusal. */
export async function cancelChainRestart(
  destination: ChainOwner,
  operationId: string,
  isCurrent: () => boolean = () => true,
): Promise<void> {
  await assertChainOwner(destination);
  const capabilities = await apiJsonTo<{
    queue?: { chain_restart_controls?: boolean };
  }>(destination.target, "/api/capabilities");
  if (capabilities.queue?.chain_restart_controls !== true)
    throw new Error(
      "Update this machine before stopping a restarted video safely.",
    );
  if (!isCurrent())
    throw new Error("The video or machine changed before cancellation.");
  await apiFetchTo(
    destination.target,
    `/api/chain-jobs/${encodeURIComponent(operationId)}/operations/${encodeURIComponent(operationId)}/cancel`,
    {
      method: "POST",
      headers: { "x-mold-destination-instance": destination.instanceId! },
    },
  );
}
