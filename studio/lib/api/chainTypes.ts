/**
 * Canonical chain / sequence wire types shared by every surface.
 *
 * These mirror the Rust types in `mold-core` (`chain.rs`, `chain_job.rs`)
 * exactly; the per-surface `types.ts` files re-export from here rather than
 * keeping divergent copies (desktop's `ChainLimits` was missing
 * `supports_sequence` before this module existed). `ChainRequest` seeds are
 * JSON numbers; full-range u64 values in job metadata, retakes, amendments,
 * and TOML projections use decimal strings where noted below.
 */

import type { SequenceTransition } from "../sequence";

export interface ChainStageWire {
  prompt: string;
  frames: number;
  /** Base64 PNG/JPEG for this clip's opening frame; stage 0 only today. */
  source_image?: string | null;
  negative_prompt?: string | null;
  seed_offset?: number | null;
  transition?: SequenceTransition;
  fade_frames?: number | null;
  loras?: Array<{ path: string; scale: number; name?: string | null }>;
}

export interface ChainRequestWire {
  model: string;
  stages: ChainStageWire[];
  motion_tail_frames?: number;
  width: number;
  height: number;
  fps?: number;
  seed?: number | null;
  steps: number;
  guidance: number;
  strength?: number;
  /** Durable chain jobs currently accept only MP4. */
  output_format?: "mp4";
  enable_audio?: boolean | null;
  original_prompt?: string | null;
  batch_id?: string | null;
  batch_index?: number | null;
  batch_count?: number | null;
  output_mode?: "one-shot" | "sequence" | null;
}

export interface ChainValidationStage {
  prompt: string;
  frames: number;
  output_frames: number;
  transition: SequenceTransition;
  fade_frames?: number | null;
  has_source_image: boolean;
  has_negative_prompt: boolean;
}

/** Read-only server normalization returned by
 * `POST /api/generate/chain/validate`. */
export interface ChainValidationResponse {
  model: string;
  width: number;
  height: number;
  fps: number;
  motion_tail_frames: number;
  stage_count: number;
  estimated_total_frames: number;
  estimated_duration_ms: number;
  stages: ChainValidationStage[];
  warnings: string[];
  vram_estimate?: {
    worst_case_bytes: number;
    fits: boolean;
  } | null;
}

/**
 * Per-clip provenance recorded into a stitched chain output's gallery
 * metadata (`OutputMetadata.chain.stages`) — the durable record of what each
 * clip asked for, which is how the Library reloads a sequence's rail. Mirrors
 * `mold_core::chain::ChainStageMetadata`. `seed` is the EFFECTIVE per-stage
 * seed as a decimal string (full-range u64), not the request's offset.
 */
export interface ChainStageMetadata {
  prompt: string;
  frames: number;
  transition: SequenceTransition;
  fade_frames?: number | null;
  seed?: string | null;
  loras?: Array<{ path: string; scale: number; name?: string | null }>;
}

/** Structured multi-clip execution provenance on `OutputMetadata` (additive;
 * present for authored sequences and auto-chained One shots). The additive
 * `output_mode` field is authoritative for restoring the authoring surface. */
export interface ChainOutputMetadata {
  stage_count: number;
  motion_tail_frames: number;
  stages: ChainStageMetadata[];
}

/** `mold.chain.v1` TOML projection echoed by the server (`ChainJobDetail.script`). */
export interface ChainScriptChain {
  model: string;
  motion_tail_frames?: number;
  width?: number;
  height?: number;
  fps?: number;
  seed?: string | null;
  steps?: number;
  guidance?: number;
  strength?: number;
  enable_audio?: boolean | null;
}

export interface ChainScriptStage {
  prompt: string;
  frames?: number;
  transition?: SequenceTransition;
  fade_frames?: number | null;
  negative_prompt?: string | null;
  source_image_path?: string | null;
  source_image_b64?: string | null;
  seed_offset?: string | null;
  loras?: Array<{ path: string; scale: number; name?: string | null }>;
}

export interface ChainScript {
  schema?: string;
  chain: ChainScriptChain;
  stages: ChainScriptStage[];
}

export type ChainJobState =
  "queued" | "running" | "interrupted" | "failed" | "completed" | "cancelled";
export type ChainExecutionPhase = "queued" | "running" | "finalizing";

export type ChainStageState = "pending" | "running" | "completed" | "failed";

export interface ChainJobSummary {
  id: string;
  state: ChainJobState;
  model: string;
  stage_count: number;
  current_stage: number;
  created_at_unix_ms: number;
  updated_at_unix_ms: number;
  error?: string | null;
  ephemeral?: boolean;
  /** Additive server truth for active work. Parent state can be `running`
   * while its next clip is still waiting for a scheduler lease. */
  execution_phase?: ChainExecutionPhase | null;
}

/** Internal long-video compatibility shims share the chain runner but are
 * still one-shot prints. Older servers listed those temporary records beside
 * authored sequences, so every client filters defensively at its list
 * boundary instead of letting a shim switch Create into Sequence mode. */
export function isAuthoredSequenceJob(job: ChainJobSummary): boolean {
  return job.ephemeral !== true;
}

export interface ChainJobStageDetail {
  idx: number;
  state: ChainStageState;
  seed?: string | null;
  frames_emitted?: number | null;
  generation_time_ms?: number | null;
  has_preview?: boolean | undefined;
  has_media?: boolean | undefined;
  cache_ready?: boolean | undefined;
  error?: string | null;
}

/** Mirrors `mold_core::chain_job::FinalizeRecord`. */
export interface ChainJobFinalizeRecord {
  /** Job-relative MP4 artifact amend/retake decode; never fetchable. */
  output: string;
  /** Gallery filename in the requested format; absent without gallery output. */
  gallery_filename?: string | null;
  at_unix_ms?: number;
  take?: number;
}

export interface ChainJobDetail extends ChainJobSummary {
  /**
   * Every finalize record the job has published, oldest first. The last
   * one names the gallery file a client attaching after settlement fetches.
   */
  finalizes?: ChainJobFinalizeRecord[];
  stages: ChainJobStageDetail[];
  /** Effective script with retakes/amends applied — the composer's edit source. */
  script?: ChainScript | null;
  /** Amend history (additive; absent on older servers). */
  amends?: ChainAmendRecord[];
}

export interface ChainJobListing {
  jobs: ChainJobSummary[];
}

export interface CreateChainJobResponse {
  job_id: string;
}

export interface ChainAmendRecord {
  at_unix_ms: number;
  previous_request_json: string;
  preserved_stages: number;
}

/** `POST /api/chain-jobs/:id/amend` — full edited stage list; the server
 * recomputes the preserved prefix itself and re-renders only dirty stages. */
export interface AmendRequest {
  stages: ChainStageWire[];
  motion_tail_frames?: number | null;
  fps?: number | null;
  seed?: string | null;
  steps?: number | null;
  guidance?: number | null;
  strength?: number | null;
  enable_audio?: boolean | null;
}

export interface AmendResponse extends ChainJobSummary {
  preserved_stages: number;
}

export type ChainJobEvent =
  | { type: "snapshot"; job: ChainJobDetail }
  | { type: "stage_start"; stage_idx: number }
  | { type: "denoise_step"; stage_idx: number; step: number; total: number }
  | {
      type: "stage_done";
      stage_idx: number;
      has_preview?: boolean;
      has_media?: boolean;
      cache_ready?: boolean;
    }
  | { type: "yielded" }
  | { type: "finalizing"; total_frames?: number }
  | {
      type: "finalized";
      output?: string;
      take?: number;
      gallery_filename?: string | null;
    }
  | { type: "state_changed"; state: ChainJobState; error?: string | null };

export interface ChainLimits {
  model: string;
  frames_per_clip_cap: number;
  fps?: number | null;
  frames_per_clip_runtime_seconds?: number | null;
  frames_per_clip_recommended: number;
  max_stages: number;
  max_total_frames: number;
  fade_frames_max: number;
  transition_modes: string[];
  quantization_family: string;
  supports_audio: boolean;
  supports_sequence: boolean;
  sequence_unsupported_reason?: string | null;
}
