import type { VideoUpscaleJob } from "../api/videoUpscale";

export type LibraryUpscaleKind = "image" | "video";

export interface UpscalerChoice {
  name: string;
  downloaded?: boolean;
}

export function libraryUpscaleLabel(kind: LibraryUpscaleKind): string {
  return kind === "video" ? "Framewise upscale…" : "Upscale…";
}

export function defaultUpscaler(choices: UpscalerChoice[]): string {
  return (
    choices.find(
      (choice) =>
        choice.downloaded && /^real-esrgan-x4plus(?::|$)/.test(choice.name),
    )?.name ??
    choices.find((choice) => choice.downloaded)?.name ??
    choices.find((choice) => /^real-esrgan-x4plus(?::|$)/.test(choice.name))?.name ??
    choices[0]?.name ??
    "real-esrgan-x4plus:fp16"
  );
}

export function framewiseProgress(job: VideoUpscaleJob): number | null {
  if (job.total_frames <= 0) return null;
  return Math.min(1, Math.max(0, job.completed_frames / job.total_frames));
}

export function framewiseStatus(job: VideoUpscaleJob): string {
  switch (job.state) {
    case "queued":
      return "Queued";
    case "running":
      return job.total_frames > 0
        ? `Upscaling frame ${Math.min(job.completed_frames + 1, job.total_frames)} of ${job.total_frames}`
        : "Preparing source video";
    case "finalizing":
      return "Finalizing video";
    case "paused":
      return "Paused — ready to resume";
    case "completed":
      return "Complete";
    case "failed":
      return job.error || "Framewise upscale failed";
    case "cancelled":
      return "Cancelled";
  }
}
