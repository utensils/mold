/**
 * Shared browser-safe mirror of the server's automatic chain-routing rules.
 * Keep the constants and branch structure aligned with
 * `crates/mold-cli/src/commands/chain.rs` and `ChainRequest::normalise`.
 *
 * `LTX2_DEFAULT_CLIP_FRAMES` is the *routing* clip size, not the model's
 * ceiling: LTX-2's real single-request limit is a 20 s runtime budget that
 * moves with fps (see `./videoBudget`), and 97 is simply the clip size that
 * fits comfortably on one consumer GPU.
 */

import { ltx2MaxFramesAtFps, maxFramesForFamilyAtFps } from "./videoBudget";

export const LTX2_DEFAULT_CLIP_FRAMES = 97;
export const MAX_CHAIN_STAGES = 16;

// 17 pixel frames become three LTX-2 latent frames of carryover under the
// VAE's 8x causal temporal compression. Keep this in sync with the CLI's
// `--motion-tail` default.
export const DEFAULT_MOTION_TAIL = 17;

export type ChainRoutingDecision =
  | {
      kind: "single";
      /** Options that forced an otherwise automatic chain to remain one
       * render so their request semantics are preserved. */
      preservedAutoChainFields?: AutoChainUnsupportedField[];
    }
  | {
      kind: "chain";
      clipFrames: number;
      motionTail: number;
      stageCount: number;
    }
  | { kind: "reject"; reason: string };

export type GenerateRoutingRequest = {
  frames?: number | null;
  fps?: number | null;
  model: string;
  negative_prompt?: string | null;
  loras?: readonly unknown[] | null;
  lora?: unknown;
  audio_file?: string | null;
  audio_file_path?: string | null;
  source_video?: string | null;
  source_video_path?: string | null;
  extend_video?: string | null;
  extend_video_path?: string | null;
  extend_overlap_frames?: number | null;
  keyframes?: readonly unknown[] | null;
  pipeline?: unknown;
  ic_lora_control?: string | null;
  retake_range?: unknown;
  spatial_upscale?: unknown;
  temporal_upscale?: string | null;
  guidance_overrides?: object | null;
};

/** Model-row fields that refine routing beyond the request wire itself. */
export type GenerateRoutingModel = {
  default_frames?: number | null | undefined;
  source_image?: string | null | undefined;
  /** `/api/models.max_frames` — the HOST's single-request ceiling, which
   * outranks this module's family fallback exactly as it does in
   * `maxVideoFrames`. Without it the two disagreed on a model whose row
   * advertises more than the client constant, and the router refused a frame
   * count the duration control offered and the server accepts. */
  max_frames?: number | null | undefined;
};

export type AutoChainUnsupportedField =
  | "negative_prompt"
  | "loras"
  | "audio_file"
  | "source_video"
  | "extend_video"
  | "keyframes"
  | "pipeline"
  | "ic_lora_control"
  | "retake_range"
  | "spatial_upscale"
  | "temporal_upscale"
  | "guidance_overrides";

export const AUTO_CHAIN_FIELD_LABELS: Readonly<
  Record<AutoChainUnsupportedField, string>
> = {
  negative_prompt: "negative prompt",
  loras: "LoRAs or camera motion",
  audio_file: "conditioning audio",
  source_video: "source video",
  extend_video: "video continuation",
  keyframes: "keyframes",
  pipeline: "pipeline selection",
  ic_lora_control: "reference control",
  retake_range: "retake range",
  spatial_upscale: "spatial upscale",
  temporal_upscale: "temporal upscale",
  guidance_overrides: "guidance overrides",
};

/** Fields the auto-expand chain form cannot carry without silently changing
 * the request. Canonical authored Sequences support a wider per-clip schema. */
export function unsupportedAutoChainFields(
  req: GenerateRoutingRequest,
): AutoChainUnsupportedField[] {
  const unsupported: AutoChainUnsupportedField[] = [];
  if (req.negative_prompt?.trim()) unsupported.push("negative_prompt");
  if ((req.loras?.length ?? 0) > 0 || req.lora) unsupported.push("loras");
  if (req.audio_file || req.audio_file_path) unsupported.push("audio_file");
  if (req.source_video || req.source_video_path)
    unsupported.push("source_video");
  if (
    req.extend_video ||
    req.extend_video_path ||
    req.extend_overlap_frames != null
  ) {
    unsupported.push("extend_video");
  }
  if ((req.keyframes?.length ?? 0) > 0) unsupported.push("keyframes");
  if (req.pipeline) unsupported.push("pipeline");
  if (req.ic_lora_control) unsupported.push("ic_lora_control");
  if (req.retake_range) unsupported.push("retake_range");
  if (req.spatial_upscale) unsupported.push("spatial_upscale");
  if (req.temporal_upscale) unsupported.push("temporal_upscale");
  const hasGuidanceOverride = Object.values(req.guidance_overrides ?? {}).some(
    (value) =>
      Array.isArray(value)
        ? value.length > 0
        : value !== null && value !== undefined,
  );
  if (hasGuidanceOverride) unsupported.push("guidance_overrides");
  return unsupported;
}

export function autoChainFieldList(
  fields: readonly AutoChainUnsupportedField[],
): string {
  const labels = fields.map((field) => AUTO_CHAIN_FIELD_LABELS[field]);
  if (labels.length <= 1) return labels[0] ?? "selected options";
  if (labels.length === 2) return `${labels[0]} and ${labels[1]}`;
  return `${labels.slice(0, -1).join(", ")}, and ${labels.at(-1)}`;
}

/** Families that may turn a one-shot into a context-preserving chain.
 * Legacy LTX-Video remains available to explicitly authored sequences, but
 * its one-shot router stays single up to the engine ceiling. */
const AUTO_CHAIN_CAPABLE_FAMILIES: ReadonlySet<string> = new Set([
  "ltx2",
  "wan",
]);

/**
 * Families where latent context crosses a clip boundary for every checkpoint.
 *
 * Wan is deliberately absent: it has no latent motion tail, and its smooth
 * handoff is last-frame *image* conditioning, which only an image-conditioned
 * checkpoint accepts (#783). Wan's carryover therefore comes from the model's
 * advertised `source_image` contract via {@link wanCarriesContext}, never from
 * this set — a T2V-only wan checkpoint must never be offered "Continue
 * motion".
 */
const FAMILIES_WITH_CONTEXT_HANDOFF: ReadonlySet<string> = new Set(["ltx2"]);

/**
 * Whether a wan checkpoint carries context across a clip boundary.
 *
 * Mirrors `mold_inference::chain::wan_carryover`. `Required` is the A14B I2V
 * 36-channel concat and `Optional` the TI2V-5B latent inpaint; both can be
 * seeded from the previous clip. `Unsupported` is text-to-video only, and an
 * unclassified checkpoint is "unknown", never an assumed handoff.
 */
export function wanCarriesContext(
  sourceImage: string | null | undefined,
): boolean {
  return sourceImage === "required" || sourceImage === "optional";
}

/**
 * The refusal a **one-shot** auto-chain earns when no context can cross its seam.
 *
 * Byte-identical mirror of `mold_core::chain::text_only_auto_chain_refusal`,
 * which is the single authority: the CLI router calls it and the server renders
 * it at `POST /api/chain-jobs` for the same ephemeral job this router submits,
 * so a user meets one sentence whichever door they came through. The shared
 * fixture `tests/fixtures/wan/surface-parity-v1.json` pins the template and
 * both sides read it (`chainRouting.wan.test.ts`, and the Rust test in
 * `mold-core`'s `chain.rs`).
 *
 * A wan checkpoint whose advertised `source_image` contract is `unsupported`
 * has no conditioning channel, so nothing crosses a clip boundary: every stage
 * re-derives the scene from the same prompt and seed and the "longer" video is
 * the same clip repeated with a visible reset at each seam (#1508). An
 * unclassified contract is "unknown", never a declared refusal.
 *
 * Only the automatic split reaches here. An authored Sequence builds its own
 * stages and is untouched: repeated clips there are what the author asked for.
 */
/**
 * The single-request frame ceiling of a model that cannot be auto-chained.
 *
 * For every other video model the routing clip size and the single-request
 * ceiling are different numbers: the ceiling is what one denoise may ask for,
 * and anything past the clip becomes an automatic sequence. A wan tier that
 * declares `source_image: "unsupported"` has no sequence to become — the split
 * is refused — so its clip size IS its ceiling, and a control that offered
 * more was offering a value submit would turn away. `null` for every model
 * this does not apply to, which is the only thing callers should key on.
 */
export function textOnlyAutoChainSingleClipCeiling(
  family: string | null | undefined,
  model: string,
  sourceImage: string | null | undefined,
  tierDefault: number | null | undefined,
): number | null {
  const normalizedFamily = canonicalizeFamily(family);
  if (normalizedFamily !== "wan" || sourceImage !== "unsupported") {
    return null;
  }
  return wanRoutingClipFrames(model, tierDefault);
}

export function textOnlyAutoChainRefusal(
  family: string | null | undefined,
  model: string,
  sourceImage: string | null | undefined,
  totalFrames: number,
  clipFrames: number,
): string | null {
  if (totalFrames <= clipFrames) return null;
  const normalizedFamily = canonicalizeFamily(family);
  if (normalizedFamily === "wan" && sourceImage === "unsupported") {
    return (
      `'${model}' is text-to-video and cannot continue motion across a clip ` +
      `boundary, so rendering ${totalFrames} frames would repeat the same ` +
      `~${clipFrames}-frame clip rather than extend it. Reduce the frame count ` +
      `to ${clipFrames} or fewer for one continuous clip, or use an ` +
      `image-to-video tier (wan22-i2v-a14b, wan22-ti2v-5b:turbo), which ` +
      `seeds each continuation with the previous clip's final frame.`
    );
  }
  if (normalizedFamily === "ltx-video") {
    return (
      `'${model}' is legacy LTX-Video and cannot continue motion across a clip ` +
      `boundary, so rendering ${totalFrames} frames would repeat the same ` +
      `~${clipFrames}-frame clip rather than extend it. Reduce the frame count ` +
      `to ${clipFrames} or fewer for one continuous clip, or use LTX-2.3 or ` +
      `LTX-2.5, which carries context into each continuation.`
    );
  }
  return null;
}

/**
 * Auto-chaining clip length for a wan render.
 *
 * Wan's per-clip ceiling is the family's flat 257-frame request cap, but the
 * routing default is a VRAM envelope, not a ceiling: the A14B pair measures
 * near the 24 GB limit well before 257 frames, while the single-expert 5B has
 * room for its own shipped 121. Both values sit on wan's `4k+1` grid.
 */
export const WAN_DEFAULT_CLIP_FRAMES = 53;
export const WAN_SINGLE_EXPERT_CLIP_FRAMES = 121;

/**
 * Pixel frames a wan continuation duplicates from the clip before it.
 *
 * The handoff seeds the continuation with the previous clip's final frame, so
 * it re-renders exactly that one frame. Mirrors `WAN_HANDOFF_DUPLICATED_FRAMES`
 * in `wan/pipeline.rs` and `./sequence`.
 */
export const WAN_HANDOFF_DUPLICATED_FRAMES = 1;

/**
 * Wan's per-checkpoint routing clip size — the clip size ONE generation
 * renders. Exported because sequence authoring must bound its per-clip picker
 * by the same value the router splits work into, even when the host advertises
 * only the family's looser ceiling.
 */
export function wanDefaultClipFrames(model: string): number {
  return /a14b/i.test(model)
    ? WAN_DEFAULT_CLIP_FRAMES
    : WAN_SINGLE_EXPERT_CLIP_FRAMES;
}

/**
 * Mirror of `mold_core::chain::wan_default_clip_frames`: the checkpoint's own
 * recorded default frame count over the family floor. The floor alone is
 * what `wanDefaultClipFrames` answers when no manifest is in hand; a tier
 * whose default was raised past it (A14B Q5/Q4 at 81, Q8 at 73) renders that
 * default as one clip, so the browser must not clamp it back to 53. Pass
 * `/api/models.default_frames` or chain-limits' `frames_per_clip_recommended`
 * (the same default, snapped to the grid) as `tierDefault`.
 */
export function wanRoutingClipFrames(
  model: string,
  tierDefault: number | null | undefined,
): number {
  const floor = wanDefaultClipFrames(model);
  return tierDefault != null && tierDefault > floor ? tierDefault : floor;
}

function canonicalizeFamily(family: string | null | undefined): string {
  const normalized = (family ?? "").trim().toLowerCase();
  return normalized === "ltx-2" ? "ltx2" : normalized;
}

/** Keep LTX-2 temporal x2 on ordinary generation: it expands one latent
 * render and is not a multi-clip chain. */
export function decideGenerateRequestRouting(
  req: GenerateRoutingRequest,
  family: string | null | undefined,
  model: GenerateRoutingModel | null = null,
  motionTail: number = DEFAULT_MOTION_TAIL,
): ChainRoutingDecision {
  const frames = req.frames;
  if (
    canonicalizeFamily(family) === "ltx2" &&
    req.temporal_upscale === "x2" &&
    frames
  ) {
    // Temporal x2 halves BOTH the stage-1 frame count and the stage-1 fps, so
    // it renders the same runtime at half the frame rate. It buys temporal
    // resolution, never extra duration — the ceiling is the same either way.
    const cap = ltx2MaxFramesAtFps(req.fps);
    return frames <= cap
      ? { kind: "single" }
      : {
          kind: "reject",
          reason: `Temporal x2 renders the same duration as a plain request, so it is still capped at ${cap} frames at ${req.fps ?? 24} fps. Reduce the frame count or raise fps.`,
        };
  }
  const decision = decideChainRouting(
    frames,
    family,
    req.model,
    motionTail,
    req.fps,
    model?.source_image,
    model?.default_frames,
    model?.max_frames,
  );
  if (decision.kind !== "chain") return decision;

  const unsupported = unsupportedAutoChainFields(req);
  if (unsupported.length === 0) return decision;

  // Same preference as the non-chain-capable branch above: the host is
  // authoritative about its own single-request ceiling and the family table is
  // only the fallback. Reading the table alone here let one half of this
  // function trust a row's advertised `max_frames` while the other half did
  // not, so a row whose ceiling differs from its family's saw two caps.
  const singleShotCap =
    model?.max_frames ??
    maxFramesForFamilyAtFps(canonicalizeFamily(family), req.fps);
  const frameCount = frames ?? 0;
  if (singleShotCap !== null && frameCount <= singleShotCap) {
    return { kind: "single", preservedAutoChainFields: unsupported };
  }

  const fps = Math.max(1, Math.floor(req.fps ?? 24));
  const options = autoChainFieldList(unsupported);
  const determiner = unsupported.length === 1 ? "that option" : "those options";
  return {
    kind: "reject",
    reason: `${frameCount} frames exceeds the ${singleShotCap ?? 97}-frame single-shot limit at ${fps} fps, and automatic chaining can’t preserve ${options}. Reduce Frames, remove ${determiner}, or author a Sequence with compatible per-clip settings.`,
  };
}

export function decideChainRouting(
  frames: number | null | undefined,
  family: string | null | undefined,
  model: string,
  motionTail: number = DEFAULT_MOTION_TAIL,
  fps: number | null | undefined = undefined,
  /**
   * The model's advertised `source_image` contract. Only wan reads it, and
   * only to decide whether the seam can carry context (#783); omitting it
   * keeps the conservative independent-clip behaviour, which is also what an
   * older server that does not advertise the field gets.
   */
  sourceImage: string | null | undefined = undefined,
  /** `/api/models.default_frames`; wan tiers may raise their one-generation
   * routing size above the family floor. */
  tierDefault: number | null | undefined = undefined,
  /** `/api/models.max_frames`. The host is authoritative about its own
   * single-request ceiling; the family table below is only the fallback for a
   * row that does not advertise one. */
  advertisedMaxFrames: number | null | undefined = undefined,
): ChainRoutingDecision {
  if (!frames || frames <= 0) return { kind: "single" };

  const normalizedFamily = canonicalizeFamily(family);
  // Every LTX-2 pipeline renders sequence clips, so capability is a property
  // of the family. The old `model.includes("distilled")` test refused a dev
  // checkpoint the server chains, and only tolerated opaque catalog IDs by
  // special-casing them.
  const isChainCapable = AUTO_CHAIN_CAPABLE_FAMILIES.has(normalizedFamily);

  if (!isChainCapable) {
    // Non-chainable models still get their family's own single-request
    // ceiling; only fall back to the routing default when the family
    // publishes none.
    const cap =
      advertisedMaxFrames ??
      maxFramesForFamilyAtFps(normalizedFamily, fps) ??
      LTX2_DEFAULT_CLIP_FRAMES;
    if (frames <= cap) return { kind: "single" };
    return {
      kind: "reject",
      reason: `Model '${model}' does not support chained video generation. Reduce frames to ${cap} or less.`,
    };
  }

  const isWan = normalizedFamily === "wan";
  const clipFrames = isWan
    ? wanRoutingClipFrames(model, tierDefault)
    : LTX2_DEFAULT_CLIP_FRAMES;
  if (frames <= clipFrames) return { kind: "single" };

  // A family that carries nothing across a seam cannot be auto-chained into
  // a longer video — it would render the same clip again.
  // The sentence is `mold-core`'s, rendered identically here so the Studio,
  // the CLI, and the server's own 422 read the same.
  const textOnly = textOnlyAutoChainRefusal(
    normalizedFamily,
    model,
    sourceImage,
    frames,
    clipFrames,
  );
  if (textOnly) return { kind: "reject", reason: textOnly };

  // Families without context handoff are forced to zero by the server. Wan's
  // answer is per checkpoint, so it comes from the advertised source-image
  // contract rather than from the family set.
  // Wan's seam duplicates exactly one frame — the one it was seeded with — so
  // the caller's LTX-shaped tail does not apply to it in either direction.
  const effectiveMotionTail = isWan
    ? wanCarriesContext(sourceImage)
      ? WAN_HANDOFF_DUPLICATED_FRAMES
      : 0
    : FAMILIES_WITH_CONTEXT_HANDOFF.has(normalizedFamily)
      ? motionTail
      : 0;

  if (effectiveMotionTail >= clipFrames) {
    return {
      kind: "reject",
      reason: `motion tail (${effectiveMotionTail}) must be strictly less than clip frames (${clipFrames}).`,
    };
  }

  // The first clip emits `clipFrames`; each continuation contributes the
  // clip minus its trimmed motion tail.
  const effective = clipFrames - effectiveMotionTail;
  const remainder = frames - clipFrames;
  const stageCount = 1 + Math.ceil(remainder / effective);

  if (stageCount > MAX_CHAIN_STAGES) {
    const maxFrames = clipFrames + (MAX_CHAIN_STAGES - 1) * effective;
    return {
      kind: "reject",
      reason: `Chained video supports at most ${maxFrames} frames (${MAX_CHAIN_STAGES} clips) for this model. Reduce the frame count.`,
    };
  }

  return {
    kind: "chain",
    clipFrames,
    motionTail: effectiveMotionTail,
    stageCount,
  };
}
