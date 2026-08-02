/**
 * Server-wire → studio normalization for a durable sequence job's echoed
 * script, shared by desktop, web, and iPhone.
 *
 * Today's server serializes `ChainJobDetail.script` from Rust's
 * `ChainScript`: stages ride under the serde rename `stage`, a stage's
 * opening image is the canonical `source_image`, and seeds are numeric.
 * The shared studio contract (`./api/chainTypes`) reads `stages`,
 * `source_image_b64`, and decimal-string seeds — u64 exceeds `Number`'s safe
 * range, so seeds only stay exact as strings.
 *
 * Both wire generations (and the studio shape itself, which older servers
 * and re-imported exports produce) must feed `chainScriptToClips` correctly,
 * so this reader is deliberately defensive: it takes `unknown`, whitelists
 * the fields the studio contract owns, and returns `null` rather than a
 * half-built script when there is nothing editable.
 */

import type {
  ChainJobStageDetail,
  ChainScript,
  ChainScriptStage,
} from "./api/chainTypes";
import type { SequenceTransition } from "./sequence";

function num(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function str(v: unknown): string | undefined {
  return typeof v === "string" && v.length > 0 ? v : undefined;
}

function seedString(v: unknown): string | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return String(v);
  if (typeof v === "string" && v.trim() !== "") return v;
  return undefined;
}

function asTransition(v: unknown): SequenceTransition | undefined {
  return v === "smooth" || v === "cut" || v === "fade" ? v : undefined;
}

function loras(
  value: unknown,
): Array<{ path: string; scale: number; name?: string | null }> | undefined {
  if (!Array.isArray(value)) return undefined;
  const rows = value.flatMap((raw) => {
    if (raw == null || typeof raw !== "object") return [];
    const row = raw as Record<string, unknown>;
    const path = str(row.path);
    const scale = num(row.scale);
    if (!path || scale === undefined) return [];
    const name = str(row.name);
    return [{ path, scale, ...(name ? { name } : {}) }];
  });
  return rows.length ? rows : undefined;
}

/**
 * Read a server-echoed chain script into the studio `ChainScript` shape.
 * Returns `null` when the payload carries nothing editable (absent script,
 * no `[chain]` table, or no model).
 */
export function normalizeServerChainScript(raw: unknown): ChainScript | null {
  if (raw == null || typeof raw !== "object") return null;
  const doc = raw as Record<string, unknown>;
  const chainRaw = doc.chain;
  if (chainRaw == null || typeof chainRaw !== "object") return null;
  const c = chainRaw as Record<string, unknown>;
  const model = str(c.model);
  if (!model) return null;

  const stagesRaw = Array.isArray(doc.stages)
    ? (doc.stages as Record<string, unknown>[])
    : Array.isArray(doc.stage)
      ? (doc.stage as Record<string, unknown>[])
      : [];

  const stages: ChainScriptStage[] = stagesRaw.map((s) => {
    const stage: ChainScriptStage = { prompt: str(s.prompt) ?? "" };
    const frames = num(s.frames);
    if (frames !== undefined) stage.frames = frames;
    const transition = asTransition(s.transition);
    if (transition !== undefined) stage.transition = transition;
    const fadeFrames = num(s.fade_frames);
    if (fadeFrames !== undefined) stage.fade_frames = fadeFrames;
    const negative = str(s.negative_prompt);
    if (negative !== undefined) stage.negative_prompt = negative;
    const image = str(s.source_image_b64) ?? str(s.source_image);
    if (image !== undefined) stage.source_image_b64 = image;
    const seedOffset = seedString(s.seed_offset);
    if (seedOffset !== undefined) stage.seed_offset = seedOffset;
    const stageLoras = loras(s.loras);
    if (stageLoras) stage.loras = stageLoras;
    return stage;
  });

  const script: ChainScript = {
    schema: "mold.chain.v1",
    chain: { model },
    stages,
  };
  const width = num(c.width);
  if (width !== undefined) script.chain.width = width;
  const height = num(c.height);
  if (height !== undefined) script.chain.height = height;
  const fps = num(c.fps);
  if (fps !== undefined) script.chain.fps = fps;
  const steps = num(c.steps);
  if (steps !== undefined) script.chain.steps = steps;
  const guidance = num(c.guidance);
  if (guidance !== undefined) script.chain.guidance = guidance;
  const strength = num(c.strength);
  if (strength !== undefined) script.chain.strength = strength;
  const motionTail = num(c.motion_tail_frames);
  if (motionTail !== undefined) script.chain.motion_tail_frames = motionTail;
  const seed = seedString(c.seed);
  if (seed !== undefined) script.chain.seed = seed;
  if (c.enable_audio === true) script.chain.enable_audio = true;
  return script;
}

/** Leading completed, disk-backed stages — the edit session's cache ceiling.
 * Older servers do not advertise `cache_ready`; completed remains the
 * backward-compatible best signal there. */
export function countLeadingCompletedStages(
  stages: readonly ChainJobStageDetail[],
): number {
  const ordered = [...stages].sort((a, b) => a.idx - b.idx);
  let count = 0;
  for (const stage of ordered) {
    if (stage.state !== "completed" || stage.cache_ready === false) break;
    count += 1;
  }
  return count;
}
