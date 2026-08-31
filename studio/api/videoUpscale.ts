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
