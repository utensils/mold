import { apiFetchTo, apiJsonTo } from "./client";
import {
  assertVideoJobOwner,
  checkedVideoJobPlan,
  type VideoJobAdmission,
  type VideoJobOwner,
} from "./videoJobs";
import type {
  VideoJob,
  VideoJobPlan,
  VideoJobSource,
} from "../lib/api/videoJobTypes";

export interface VideoRestartOptions {
  source_instance_id: string;
  source_revision: string;
  restart: { supported: boolean; reason?: string };
  send: { supported: boolean; reason?: string };
}
export interface VideoRestartPrefix {
  filename: string;
  archive_identity_sha256: string;
  sha256: string;
  size_bytes: string;
  source_ref: string;
  /** Forward unchanged: metadata may contain integers outside JS's range. */
  upload_spec_json: string;
  /** Same-installation immutable grant; remote hosts must relay instead. */
  retained_upload?: SourceUpload;
}
export interface VideoRestartExport {
  destination_instance_id: string;
  source_instance_id: string;
  source_revision: string;
  /** Private literal JSON. Never parse, rewrite or persist this body. */
  request_json: string;
  prefix?: VideoRestartPrefix;
}
export interface VideoRestartAdmission extends VideoJobAdmission {
  sourceJobId: string;
  sourceHostId: string;
  sourceInstanceId: string;
  sourceRevision: string;
}
type SourceUpload = NonNullable<VideoJobSource["retained_upload"]>;
const CHUNK = 4 * 1024 * 1024;
const U64 = 18_446_744_073_709_551_615n;
const changed = () =>
  new Error("The video or machine changed. Refresh and try again.");
function decimal(value: unknown): bigint {
  if (
    typeof value !== "string" ||
    !/^(0|[1-9]\d*)$/.test(value) ||
    BigInt(value) > U64
  )
    throw new Error("Invalid video byte count.");
  return BigInt(value);
}
function digest(value: unknown): value is string {
  return typeof value === "string" && /^[a-fA-F0-9]{64}$/.test(value);
}
function operation(value: string): void {
  if (
    !/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/.test(
      value,
    )
  )
    throw new Error("Invalid video operation identifier.");
}
function current(check: () => boolean): void {
  if (!check()) throw changed();
}
async function restartOwner(
  owner: VideoJobOwner,
  transfer = false,
): Promise<void> {
  await assertVideoJobOwner(owner);
  const caps = await apiJsonTo<{
    video_jobs?: { restart?: boolean; source_transfer?: boolean };
  }>(owner.target, "/api/capabilities");
  if (
    caps.video_jobs?.restart !== true ||
    (transfer && caps.video_jobs.source_transfer !== true)
  )
    throw new Error("Update this machine before starting a video again.");
}
function checkedExport(
  source: VideoJobOwner,
  destination: VideoJobOwner,
  value: VideoRestartExport,
): void {
  if (
    !source.instanceId ||
    !destination.instanceId ||
    value.source_instance_id !== source.instanceId ||
    value.destination_instance_id !== destination.instanceId
  )
    throw changed();
  if (
    typeof value.source_revision !== "string" ||
    !value.source_revision ||
    typeof value.request_json !== "string" ||
    !value.request_json.trim()
  )
    throw new Error("The original video request is unavailable.");
  if (value.prefix) {
    const p = value.prefix;
    if (
      typeof p.filename !== "string" ||
      !p.filename ||
      !digest(p.archive_identity_sha256) ||
      !digest(p.sha256) ||
      decimal(p.size_bytes) === 0n ||
      typeof p.source_ref !== "string" ||
      !p.source_ref ||
      typeof p.upload_spec_json !== "string" ||
      !p.upload_spec_json.trim()
    )
      throw new Error("The original video prefix is unavailable.");
    if (p.retained_upload) {
      const ref = p.retained_upload;
      if (
        source.instanceId !== destination.instanceId ||
        ref.sha256 !== p.sha256 ||
        ref.size_bytes !== p.size_bytes
      )
        throw changed();
      operation(ref.handle);
    }
  }
}
export async function videoRestartOptions(
  owner: VideoJobOwner,
  id: string,
): Promise<VideoRestartOptions> {
  await restartOwner(owner);
  const value = await apiJsonTo<VideoRestartOptions>(
    owner.target,
    `/api/video-jobs/${encodeURIComponent(id)}/restart-options`,
    { headers: { "x-mold-destination-instance": owner.instanceId! } },
  );
  if (
    value.source_instance_id !== owner.instanceId ||
    !value.source_revision ||
    typeof value.restart?.supported !== "boolean" ||
    typeof value.send?.supported !== "boolean"
  )
    throw changed();
  return value;
}
export async function exportVideoRestart(
  source: VideoJobOwner,
  id: string,
  options: VideoRestartOptions,
  destination: VideoJobOwner,
  isCurrent: () => boolean = () => true,
): Promise<VideoRestartExport> {
  if (
    source.instanceId !== options.source_instance_id ||
    !destination.instanceId
  )
    throw changed();
  await restartOwner(source);
  current(isCurrent);
  await restartOwner(destination);
  current(isCurrent);
  const permission =
    source.instanceId === destination.instanceId
      ? options.restart
      : options.send;
  if (!permission.supported)
    throw new Error(permission.reason ?? "This video cannot start again here.");
  const value = await apiJsonTo<VideoRestartExport>(
    source.target,
    `/api/video-jobs/${encodeURIComponent(id)}/restart-export`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-mold-destination-instance": source.instanceId!,
      },
      body: JSON.stringify({
        source_instance_id: options.source_instance_id,
        source_revision: options.source_revision,
        destination_instance_id: destination.instanceId,
      }),
    },
  );
  current(isCurrent);
  checkedExport(source, destination, value);
  if (value.source_revision !== options.source_revision) throw changed();
  return value;
}
async function boundedBytes(
  response: Response,
  expected: number,
): Promise<Uint8Array<ArrayBuffer>> {
  if (!response.body)
    throw new Error("The original video stream is unavailable.");
  const declared = response.headers.get("content-length");
  if (declared !== null && declared !== String(expected))
    throw new Error("The original video chunk has an unexpected size.");
  const reader = response.body.getReader();
  const bytes = new Uint8Array(expected);
  let count = 0;
  try {
    for (;;) {
      const value = await reader.read();
      if (value.done) break;
      if (value.value.length > expected - count)
        throw new Error("The original video chunk exceeds its bounded size.");
      bytes.set(value.value, count);
      count += value.value.length;
    }
    if (count !== expected)
      throw new Error("The original video chunk was truncated.");
    return bytes;
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
/** Constant-memory relay. UUID/progress can be retained; private media never enters a journal. */
export async function transferVideoRestartPrefix(options: {
  source: VideoJobOwner;
  sourceJobId: string;
  destination: VideoJobOwner;
  exported: VideoRestartExport;
  uploadId: string;
  isCurrent?: () => boolean;
  progress?: (bytes: string, total: string) => void;
}): Promise<SourceUpload | undefined> {
  const { source, destination, exported, uploadId } = options;
  const fresh = options.isCurrent ?? (() => true);
  checkedExport(source, destination, exported);
  operation(uploadId);
  current(fresh);
  const prefix = exported.prefix;
  if (!prefix) return undefined;
  await restartOwner(source, true);
  current(fresh);
  await restartOwner(destination, true);
  current(fresh);
  if (prefix.retained_upload) return prefix.retained_upload;
  const headers = { "x-mold-destination-instance": destination.instanceId! };
  const route = `/api/video-source-uploads/${uploadId}`;
  const status = await apiJsonTo<{
    handle: string;
    offset_bytes: string;
    completed: boolean;
  }>(destination.target, route, {
    method: "POST",
    headers: { ...headers, "Content-Type": "application/json" },
    body: prefix.upload_spec_json,
  });
  current(fresh);
  const size = decimal(prefix.size_bytes);
  let offset = decimal(status.offset_bytes);
  if (
    status.handle !== uploadId ||
    offset > size ||
    typeof status.completed !== "boolean" ||
    (status.completed && offset !== size)
  )
    throw new Error("The destination returned another upload.");
  while (offset < size) {
    current(fresh);
    const expected = Number(
      size - offset > BigInt(CHUNK) ? BigInt(CHUNK) : size - offset,
    );
    const response = await apiFetchTo(
      source.target,
      `/api/video-jobs/${encodeURIComponent(options.sourceJobId)}/restart-source?offset_bytes=${offset}`,
      {
        headers: {
          "x-mold-source-instance": source.instanceId!,
          "x-mold-video-source-ref": prefix.source_ref,
        },
      },
    );
    const bytes = await boundedBytes(response, expected);
    current(fresh);
    const uploaded = await apiJsonTo<{ handle: string; offset_bytes: string }>(
      destination.target,
      `${route}/chunks?offset_bytes=${offset}`,
      {
        method: "PUT",
        headers: { ...headers, "Content-Type": "application/octet-stream" },
        body: bytes,
      },
    );
    current(fresh);
    const next = offset + BigInt(bytes.length);
    if (uploaded.handle !== uploadId || decimal(uploaded.offset_bytes) !== next)
      throw new Error(
        "The destination upload changed. Keep its identifier and check the machine.",
      );
    offset = next;
    options.progress?.(String(offset), prefix.size_bytes);
  }
  const sealed = await apiJsonTo<SourceUpload>(
    destination.target,
    `${route}/complete`,
    { method: "POST", headers },
  );
  current(fresh);
  if (
    sealed.handle !== uploadId ||
    sealed.sha256 !== prefix.sha256 ||
    sealed.size_bytes !== prefix.size_bytes
  )
    throw new Error("The destination retained a different video prefix.");
  return sealed;
}
function sourceHeaders(
  destination: VideoJobOwner,
  prefix: VideoRestartPrefix | undefined,
  upload: SourceUpload | undefined,
): Record<string, string> {
  if (
    prefix &&
    (!upload ||
      upload.sha256 !== prefix.sha256 ||
      upload.size_bytes !== prefix.size_bytes)
  )
    throw new Error("Transfer the original video before starting again.");
  return {
    "Content-Type": "application/json",
    "x-mold-destination-instance": destination.instanceId!,
    ...(upload ? { "x-mold-video-source-upload": JSON.stringify(upload) } : {}),
  };
}
export async function validateVideoRestart(
  destination: VideoJobOwner,
  exported: VideoRestartExport,
  upload?: SourceUpload,
): Promise<VideoJobPlan> {
  await restartOwner(destination);
  if (destination.instanceId !== exported.destination_instance_id)
    throw changed();
  const value = checkedVideoJobPlan(
    await apiJsonTo(destination.target, "/api/video-jobs/validate", {
      method: "POST",
      headers: sourceHeaders(destination, exported.prefix, upload),
      body: exported.request_json,
    }),
  );
  if (!value.validation_revision)
    throw new Error("This video needs full validation before starting again.");
  return value;
}
export async function admitVideoRestart(options: {
  source: VideoJobOwner;
  sourceJobId: string;
  options: VideoRestartOptions;
  destination: VideoJobOwner;
  exported: VideoRestartExport;
  proof: string;
  operationId: string;
  upload?: SourceUpload;
  persist: (record: VideoRestartAdmission) => void;
  isCurrent?: () => boolean;
}): Promise<VideoJob> {
  const { source, destination, exported } = options;
  checkedExport(source, destination, exported);
  operation(options.operationId);
  if (
    exported.source_revision !== options.options.source_revision ||
    source.instanceId !== options.options.source_instance_id
  )
    throw changed();
  const permission =
    source.instanceId === destination.instanceId
      ? options.options.restart
      : options.options.send;
  if (!permission.supported || !options.proof)
    throw new Error(
      permission.reason ?? "Validate this video before starting again.",
    );
  const headers = sourceHeaders(destination, exported.prefix, options.upload);
  await restartOwner(destination);
  current(options.isCurrent ?? (() => true));
  options.persist({
    kind: "video_job",
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
    sourceInstanceId: source.instanceId!,
    sourceRevision: options.options.source_revision,
  });
  const accepted = await apiJsonTo<VideoJob>(
    destination.target,
    "/api/video-jobs",
    {
      method: "POST",
      headers: {
        ...headers,
        "x-mold-operation-id": options.operationId,
        "x-mold-video-validation-revision": options.proof,
      },
      body: exported.request_json,
    },
  );
  if (accepted.id !== options.operationId)
    throw new Error(
      "This machine returned a different video. Keep the original operation for recovery.",
    );
  return accepted;
}
