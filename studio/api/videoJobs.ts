import { exactSeed } from "../lib/videoSettingsReuse";
import { ApiError, apiFetchTo, apiJsonTo } from "./client";
import {
  assertChainOwner,
  type ChainAdmission,
  type ChainOwner,
} from "./chainJobs";
import type {
  VideoJob,
  VideoJobPlan,
  VideoJobRequest,
} from "../lib/api/videoJobTypes";

export type VideoJobOwner = ChainOwner;
export interface VideoJobAdmission extends ChainAdmission {
  kind: "video_job";
}
export interface PreparedVideoMedia {
  /** Request-bound, one-use authority. Keep only in memory until admission. */
  retainedMediaSession?: string;
}
export interface PreparedDurationVideo<Generation extends object> {
  request: VideoJobRequest<Generation>;
  media?: PreparedVideoMedia;
  /** Capture only lease cancellation tokens, never the original request bytes. */
  release?: () => Promise<void>;
}
const SCALAR_FIELDS = new Set([
  "model",
  "prompt",
  "width",
  "height",
  "steps",
  "guidance",
  "frames",
  "fps",
  "batch_size",
  "enable_audio",
  "pipeline",
  "offload",
  "seed",
  "strength",
  "ltx2_pipeline",
  "ltx2_temporal_upscale",
  "ltx2_spatial_upscale",
]);
/** Planning is display-only. Never upload images, endpoints, references or
 * audio while the user types or drags a duration/size control. */
export function planningVideoRequest<Generation extends object>(
  request: VideoJobRequest<Generation>,
): VideoJobRequest {
  return {
    duration_ms: request.duration_ms,
    ...(request.source ? { source: request.source } : {}),
    ...(request.scenes ? { scenes: request.scenes } : {}),
    generation: Object.fromEntries(
      Object.entries(request.generation).filter(
        ([key, value]) =>
          SCALAR_FIELDS.has(key) &&
          (value == null ||
            ["string", "number", "boolean"].includes(typeof value)),
      ),
    ),
  };
}
function validateRequest(request: VideoJobRequest<object>): void {
  if (
    !/^[1-9]\d*$/.test(request.duration_ms) ||
    BigInt(request.duration_ms) > 18_446_744_073_709_551_615n
  )
    throw new Error("Enter a positive duration in seconds.");
  const generation = request.generation as {
    seed?: unknown;
    batch_size?: unknown;
  };
  if (generation.seed != null && !exactSeed(generation.seed))
    throw new Error(
      "This seed cannot be represented exactly. Enter a seed within the supported browser range.",
    );
  if (generation.batch_size != null && generation.batch_size !== 1)
    throw new Error(
      "Prepare each video in the batch as its own movie before submission.",
    );
}
export async function assertVideoJobOwner(
  owner: VideoJobOwner,
  extending = false,
): Promise<void> {
  await assertChainOwner(owner);
  const caps = await apiJsonTo<{
    video_jobs?: {
      supported?: boolean;
      literal_prompts?: boolean;
      library_extend?: boolean;
    };
  }>(owner.target, "/api/capabilities");
  if (
    caps.video_jobs?.supported !== true ||
    caps.video_jobs.literal_prompts !== true ||
    (extending && caps.video_jobs.library_extend !== true)
  )
    throw new Error(
      "Update this machine before making videos with a requested duration.",
    );
}
export function checkedVideoJobPlan(value: unknown): VideoJobPlan {
  if (!value || typeof value !== "object")
    throw new Error("This machine returned an incompatible video plan.");
  const plan = value as Record<string, unknown>;
  const counts = [
    "clip_count",
    "part_count",
    "added_frames",
    "source_frames",
    "total_frames",
    "added_duration_ms",
    "total_duration_ms",
  ];
  const dimensions = ["width", "height", "fps", "single_clip_max_frames"];
  if (
    typeof plan.validation_revision !== "string" ||
    typeof plan.model !== "string" ||
    !["continuous", "blended_shots"].includes(String(plan.join_mode)) ||
    dimensions.some(
      (key) => !Number.isSafeInteger(plan[key]) || (plan[key] as number) < 1,
    ) ||
    !Number.isSafeInteger(plan.overlap_frames) ||
    (plan.overlap_frames as number) < 0 ||
    counts.some(
      (key) =>
        typeof plan[key] !== "string" ||
        !/^(0|[1-9]\d*)$/.test(plan[key] as string) ||
        (plan[key] as string).length > 20 ||
        BigInt(plan[key] as string) > 18_446_744_073_709_551_615n,
    ) ||
    !Array.isArray(plan.warnings) ||
    plan.warnings.some((warning) => typeof warning !== "string")
  )
    throw new Error(
      "This machine returned an incompatible video plan. Update it before continuing.",
    );
  return value as VideoJobPlan;
}
export async function validateVideoJob<Generation extends object>(
  owner: VideoJobOwner,
  request: VideoJobRequest<Generation>,
  options: {
    planningOnly?: boolean;
    isCurrent?: () => boolean;
    signal?: AbortSignal;
    retainedMediaSession?: string;
  } = {},
): Promise<VideoJobPlan> {
  validateRequest(request);
  await assertVideoJobOwner(owner, !!request.source);
  if (options.isCurrent?.() === false)
    throw new Error("The video or machine changed before validation.");
  return checkedVideoJobPlan(
    await apiJsonTo(
      owner.target,
      `/api/video-jobs/validate${options.planningOnly ? "?planning_only=true" : ""}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-mold-destination-instance": owner.instanceId!,
          ...(!options.planningOnly && options.retainedMediaSession
            ? { "X-Mold-Retained-Media-Session": options.retainedMediaSession }
            : {}),
        },
        body: JSON.stringify(
          options.planningOnly ? planningVideoRequest(request) : request,
        ),
        ...(options.signal ? { signal: options.signal } : {}),
      },
    ),
  );
}
/** The caller retains this safe scalar journal across an uncertain reply.
 * Admission never retries automatically or submits a chain request. */
export async function admitVideoJob<Generation extends object>(
  owner: VideoJobOwner,
  request: VideoJobRequest<Generation>,
  plan: VideoJobPlan,
  operationId: string,
  persist: (record: VideoJobAdmission) => void,
  isCurrent: () => boolean = () => true,
  media: PreparedVideoMedia = {},
): Promise<VideoJob> {
  if (!plan.validation_revision)
    throw new Error(
      "Validate this video before making it. The displayed plan cannot authorize admission.",
    );
  validateRequest(request);
  await assertVideoJobOwner(owner, !!request.source);
  if (!isCurrent())
    throw new Error("The video or machine changed before submission.");
  persist({
    kind: "video_job",
    operationId,
    hostId: owner.hostId,
    instanceId: owner.instanceId!,
    baseUrl: owner.target.baseUrl,
    ...(owner.target.connectionMode
      ? { connectionMode: owner.target.connectionMode }
      : {}),
    createdAt: Date.now(),
  });
  return apiJsonTo(owner.target, "/api/video-jobs", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-mold-operation-id": operationId,
      "x-mold-destination-instance": owner.instanceId!,
      ...(media.retainedMediaSession
        ? { "X-Mold-Retained-Media-Session": media.retainedMediaSession }
        : {}),
    },
    body: JSON.stringify({
      ...request,
      validation_revision: plan.validation_revision,
    }),
  });
}
function videoJobNotFound(cause: unknown): boolean {
  return (
    cause instanceof ApiError &&
    cause.status === 404 &&
    (cause.body as { code?: unknown } | null)?.code === "VIDEO_JOB_NOT_FOUND"
  );
}
/** `progress_phase` is additive: absent on an older server, and a value this
 * build does not know still reads as rendering. Only a non-string is dropped,
 * so a malformed phase never refuses an otherwise valid job. */
export function withCheckedProgressPhase(job: VideoJob): VideoJob {
  if (job.progress_phase == null || typeof job.progress_phase === "string")
    return job;
  const { progress_phase: _dropped, ...rest } = job;
  return rest;
}
export async function getVideoJob(
  owner: VideoJobOwner,
  id: string,
): Promise<VideoJob | null> {
  try {
    const job = await apiJsonTo<VideoJob>(
      owner.target,
      `/api/video-jobs/${encodeURIComponent(id)}`,
      { headers: { "x-mold-destination-instance": owner.instanceId! } },
    );
    if (job.id !== id)
      throw new Error(
        "This machine returned a different video job. Keep the original operation for recovery.",
      );
    return withCheckedProgressPhase(job);
  } catch (cause) {
    if (videoJobNotFound(cause)) return null;
    throw cause;
  }
}
export async function mutateVideoJob(
  owner: VideoJobOwner,
  id: string,
  action: "cancel" | "resume" | "delete",
  isCurrent: () => boolean = () => true,
): Promise<void> {
  await assertVideoJobOwner(owner);
  if (!isCurrent())
    throw new Error("The video or machine changed before this action.");
  if (action !== "cancel") {
    const job = await getVideoJob(owner, id);
    if (!isCurrent())
      throw new Error("The video or machine changed before this action.");
    if (!job && action === "delete") return;
    if (!job || !(action === "resume" ? job.can_resume : job.can_delete))
      throw new Error("This video changed. Refresh its Queue controls.");
    if (!isCurrent())
      throw new Error("The video or machine changed before this action.");
  }
  let response: Response;
  try {
    response = await apiFetchTo(
      owner.target,
      `/api/video-jobs/${encodeURIComponent(id)}${action === "delete" ? "" : `/${action}`}`,
      {
        method: action === "delete" ? "DELETE" : "POST",
        headers: { "x-mold-destination-instance": owner.instanceId! },
      },
    );
  } catch (cause) {
    if (action !== "delete" || !isCurrent()) throw cause;
    if (videoJobNotFound(cause)) return;
    // Reconcile ambiguous Delete replies with one read, never another mutation.
    if (cause instanceof ApiError && cause.status < 500 && cause.status !== 408)
      throw cause;
    await assertVideoJobOwner(owner);
    if (!isCurrent())
      throw new Error("The video or machine changed before this action.");
    const remaining = await getVideoJob(owner, id);
    if (!isCurrent())
      throw new Error("The video or machine changed before this action.");
    if (!remaining) return;
    throw cause;
  }
  if (action === "cancel") {
    let receipt: unknown;
    try {
      receipt = await response.json();
    } catch {
      /* Keep the operation uncertain. */
    }
    if (
      !receipt ||
      typeof receipt !== "object" ||
      (receipt as { accepted?: unknown }).accepted !== true
    )
      throw new Error(
        "This machine did not confirm the Stop request. Check the saved video attempt before retrying.",
      );
  }
}
