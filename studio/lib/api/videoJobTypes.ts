/** New video protocol: prompt strings are literal JSON and u64 counters are
 * decimal strings. They must never pass through JavaScript Number. */
export interface VideoJobSource {
  filename: string;
  expected_archive_identity_sha256?: string;
  retained_upload?: { handle: string; sha256: string; size_bytes: string };
}
export interface VideoJobScene {
  prompt: string;
  duration_ms: string;
}
export interface VideoJobRequest<
  Generation extends object = Record<string, unknown>,
> {
  generation: Generation;
  duration_ms: string;
  source?: VideoJobSource;
  scenes?: VideoJobScene[];
  validation_revision?: string;
}
export interface VideoJobPlan {
  validation_revision: string;
  model: string;
  width: number;
  height: number;
  fps: number;
  single_clip_max_frames: number;
  overlap_frames: number;
  join_mode: "continuous" | "blended_shots";
  clip_count: string;
  part_count: string;
  added_frames: string;
  source_frames: string;
  total_frames: string;
  added_duration_ms: string;
  total_duration_ms: string;
  warnings: string[];
}
export type VideoJobState =
  | "queued"
  | "running"
  | "paused"
  | "completed"
  | "failed"
  | "interrupted"
  | "cancelled";
/** What a running parent is really doing. Absent on an older server (read it
 * as rendering); a value this build does not know also reads as rendering. */
export type VideoJobProgressPhase = "waiting" | "rendering" | "joining";
export interface VideoJob {
  id: string;
  state: VideoJobState;
  /** Present only while `state == running`. */
  progress_phase?: VideoJobProgressPhase | (string & {}) | null;
  plan: VideoJobPlan;
  completed_clips: string;
  active_child_id?: string | null;
  output_filename?: string | null;
  error?: string | null;
  error_detail?: string | null;
  created_at_ms: number;
  updated_at_ms: number;
  can_cancel: boolean;
  can_resume: boolean;
  can_delete: boolean;
}
