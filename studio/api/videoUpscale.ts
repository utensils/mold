import { apiJsonTo, type ApiTarget } from "./client";

export type VideoUpscaleJobState =
  | "queued"
  | "running"
  | "finalizing"
  | "paused"
  | "completed"
  | "failed"
  | "cancelled";

export interface VideoUpscaleJob {
  contract_version: number;
  id: string;
  state: VideoUpscaleJobState;
  source?: { kind: "library"; filename: string } | { kind: "upload"; handle: string };
  model: string;
  completed_frames: number;
  total_frames: number;
  output_filename?: string | null;
  error?: string | null;
  disclosure: string;
}

export function createFramewiseUpscale(
  target: ApiTarget,
  filename: string,
  model: string,
  tileSize?: number,
): Promise<VideoUpscaleJob> {
  return apiJsonTo(target, "/api/video-upscale-jobs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      source: { kind: "library", filename },
      model,
      ...(tileSize === undefined ? {} : { tile_size: tileSize }),
    }),
  });
}

export interface GalleryImageUpscaleResponse {
  filename: string;
  model: string;
  scale_factor: number;
}

export function upscaleLibraryImage(
  target: ApiTarget,
  filename: string,
  model: string,
  tileSize?: number,
): Promise<GalleryImageUpscaleResponse> {
  return apiJsonTo(target, "/api/gallery/upscale", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      filename,
      model,
      ...(tileSize === undefined ? {} : { tile_size: tileSize }),
    }),
  });
}

export function listFramewiseUpscales(target: ApiTarget): Promise<VideoUpscaleJob[]> {
  return apiJsonTo(target, "/api/video-upscale-jobs");
}

export async function findRecoverableFramewiseUpscale(
  target: ApiTarget,
  filename: string,
): Promise<VideoUpscaleJob | null> {
  return recoverableFramewiseUpscale(await listFramewiseUpscales(target), filename);
}

export function recoverableFramewiseUpscale(
  jobs: readonly VideoUpscaleJob[],
  filename: string,
): VideoUpscaleJob | null {
  return (
    jobs.find(
      (job) =>
        job.source?.kind === "library" &&
        job.source.filename === filename &&
        !["completed", "failed", "cancelled"].includes(job.state),
    ) ?? null
  );
}

export function getFramewiseUpscale(
  target: ApiTarget,
  id: string,
): Promise<VideoUpscaleJob> {
  return apiJsonTo(target, `/api/video-upscale-jobs/${encodeURIComponent(id)}`);
}

export function transitionFramewiseUpscale(
  target: ApiTarget,
  id: string,
  action: "pause" | "resume" | "cancel",
): Promise<VideoUpscaleJob> {
  const encoded = encodeURIComponent(id);
  return apiJsonTo(
    target,
    action === "cancel"
      ? `/api/video-upscale-jobs/${encoded}`
      : `/api/video-upscale-jobs/${encoded}/${action}`,
    { method: action === "cancel" ? "DELETE" : "POST" },
  );
}
