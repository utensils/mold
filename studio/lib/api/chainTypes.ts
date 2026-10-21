import type {
  GenerationReference,
  GenerationReferenceMetadata,
} from "../generationReferences";
import type { KeyframeConditionWire, KeyframeMetadata } from "./serverTypes";
import type { OutputFormat } from "../generated/generationProfileV1";
/**
 * Canonical chain / sequence wire types shared by every surface.
 *
 * These mirror the Rust types in `mold-core` (`chain.rs`, `chain_job.rs`)
 * with additive compatibility for older hosts; per-surface `types.ts` files
 * re-export from here rather than keeping divergent copies. `ChainRequest` seeds accept checked decimal strings as well as safe JSON
 * numbers; decimal strings preserve full-range u64 values in browser clients.
 */

/**
 * How one clip joins the next on the wire. The compact editor defaults to
 * smooth; the API and CLI additionally expose explicit cuts and fades.
 */
export type ChainTransition = "smooth" | "cut" | "fade";

export interface ChainContinuation {
  job_id: string;
  finalize_take: number;
}

export interface ChainStageWire {
  prompt: string;
  frames: number;
  /** Base64 PNG/JPEG for this clip's authored opening frame. */
  source_image?: string | null;
  generation_references?: GenerationReference[] | null;
  keyframes?: KeyframeConditionWire[] | null;
  negative_prompt?: string | null;
  seed_offset?: number | string | null;
  transition?: ChainTransition;
  fade_frames?: number | null;
  loras?: Array<{ path: string; scale: number; name?: string | null }>;
}

export interface ChainRequestWire {
  model: string;
  stages?: ChainStageWire[];
  continue_from?: ChainContinuation;
  ephemeral?: boolean;
  title?: string | null;
  tags?: string[];
  collection?: { id?: string; name?: string };
  prompt?: string;
  total_frames?: number;
  clip_frames?: number;
  source_image?: string | null;
  placement?: import("./serverTypes").DevicePlacement | null;
  motion_tail_frames?: number;
  source_fit?: import("../sourceFit").SourceFitPolicy | null;
  width: number;
  height: number;
  fps?: number;
  seed?: number | string | null;
  steps: number;
  guidance: number;
  strength?: number;
  /** Durable chain jobs currently accept only MP4. */
  output_format?: OutputFormat;
  enable_audio?: boolean | null;
  original_prompt?: string | null;
  prompt_transform?: {
    operation: "expand" | "remix";
    root_prompt?: string;
    source_prompt: string;
    source_kind: import("./serverTypes").RemixSourceKind;
    task: import("../expandTask").ExpandTask;
    dimensions?: import("./serverTypes").RemixDimension[];
  };
  batch_id?: string | null;
  batch_index?: number | null;
  batch_count?: number | null;
  output_mode?: "one-shot" | "sequence" | null;
}

/**
 * Per-clip provenance recorded into a stitched chain output's gallery
 * metadata (`OutputMetadata.chain.stages`) — the durable record of what each
 * clip asked for, which is how the Library reloads a sequence's rail. Mirrors
 * `mold_core::chain::ChainStageMetadata`. `seed` is the EFFECTIVE per-stage
 * seed as a decimal string (full-range u64), not the request's offset.
 */
export interface ChainStageMetadata {
  generation_references?: GenerationReferenceMetadata[];
  keyframes?: KeyframeMetadata[];
  prompt: string;
  frames: number;
  transition: ChainTransition;
  fade_frames?: number | null;
  seed?: string | null;
  loras?: Array<{ path: string; scale: number; name?: string | null }>;
}

/** Structured multi-clip execution provenance on `OutputMetadata` (additive;
 * present for authored sequences and auto-chained One shots). The additive
 * `output_mode` field is authoritative for restoring the authoring surface. */
export interface ChainOutputMetadata {
  continue_from?: ChainContinuation;
  stage_count: number;
  motion_tail_frames: number;
  stages: ChainStageMetadata[];
}

/** `mold.chain.v1` TOML projection echoed by the server (`ChainJobDetail.script`). */
export interface ChainScriptChain {
  continue_from?: ChainContinuation;
  model: string;
  width: number;
  height: number;
  fps: number;
  seed?: number | string | null;
  steps: number;
  guidance: number;
  strength: number;
  motion_tail_frames: number;
  source_fit?: import("../sourceFit").SourceFitPolicy | null;
  output_format: OutputFormat;
  enable_audio?: boolean | null;
}

export interface ChainScript {
  schema: string;
  chain: ChainScriptChain;
  stage: ChainStageWire[];
}

export type ChainJobState =
  | "queued"
  | "running"
  | "paused"
  | "interrupted"
  | "failed"
  | "completed"
  | "cancelled";
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
