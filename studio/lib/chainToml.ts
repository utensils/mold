/**
 * `mold.chain.v1` TOML ↔ [`ChainScript`] for every surface's Import/Export
 * .toml file tools (desktop, web, and iPhone). Field names match the
 * canonical files `mold run --script` reads (top-level `schema`, one
 * `[chain]` table, `[[stage]]` tables — `stage` is serde's rename of the
 * Rust `stages` field), so old exports keep importing and new exports keep
 * running.
 *
 * Two Rust-side constraints shape this module:
 * - mold-core's `ChainScriptChain` has NO serde defaults — `strength` and
 *   `output_format` (which the studio `ChainScript` doesn't carry) must
 *   still be emitted or the CLI rejects the file. Writing is hand-rolled
 *   rather than smol-toml `stringify`, which collapses `1.0` to the integer
 *   `1` and would trip mold-core's f64 fields.
 * - Seeds are u64. They ride as decimal strings in TS
 *   ([`ChainScriptChain.seed`]) because u64 exceeds `Number`'s safe range,
 *   and cross the TOML boundary as bare integers via `BigInt`, so no
 *   precision is lost in either direction.
 */

import { parse as parseToml } from "smol-toml";
import type { ChainScript, ChainScriptStage } from "./api/chainTypes";
import type { SequenceTransition } from "./sequence";

export const CHAIN_SCHEMA = "mold.chain.v1";

/**
 * TOML integer/float → `number`. `integersAsBigInt: "asNeeded"` only
 * promotes values past `Number.MAX_SAFE_INTEGER`, so the bigint branch only
 * fires on absurd (but parseable) documents — clamping to a finite number
 * beats silently dropping the field.
 */
function num(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "bigint") return Number(v);
  return undefined;
}

function str(v: unknown): string | undefined {
  return typeof v === "string" ? v : undefined;
}

function asTransition(v: unknown): SequenceTransition {
  return v === "cut" || v === "fade" ? v : "smooth";
}

/** u64-safe decimal string from a TOML integer (number, bigint, or string). */
function seedString(v: unknown): string | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return String(v);
  if (typeof v === "bigint") return v.toString();
  if (typeof v === "string" && v.trim() !== "") return v;
  return undefined;
}

/** Escape a string for a TOML basic (double-quoted) string. */
function tomlString(s: string): string {
  const escaped = s
    .replace(/\\/g, "\\\\")
    .replace(/"/g, '\\"')
    .replace(/\n/g, "\\n")
    .replace(/\t/g, "\\t");
  return `"${escaped}"`;
}

/** Floats keep a decimal point so Rust's f64 fields always see a TOML float. */
function tomlFloat(n: number): string {
  return Number.isInteger(n) ? `${n}.0` : String(n);
}

/** Decimal-string (or numeric) seed → bare TOML integer, u64-exact. */
function tomlInt(value: string | number): string {
  return BigInt(value).toString();
}

/**
 * Serialize a studio `ChainScript` to canonical `mold.chain.v1` TOML,
 * omitting fields that are null/undefined.
 */
export function serializeChainScript(
  script: ChainScript & { chain: { strength?: number; output_format?: string } },
): string {
  const c = script.chain;
  const lines: string[] = [`schema = ${tomlString(CHAIN_SCHEMA)}`, "", "[chain]"];
  lines.push(`model = ${tomlString(c.model)}`);
  if (c.width != null) lines.push(`width = ${c.width}`);
  if (c.height != null) lines.push(`height = ${c.height}`);
  if (c.fps != null) lines.push(`fps = ${c.fps}`);
  if (c.seed != null) lines.push(`seed = ${tomlInt(c.seed)}`);
  if (c.steps != null) lines.push(`steps = ${c.steps}`);
  if (c.guidance != null) lines.push(`guidance = ${tomlFloat(c.guidance)}`);
  lines.push(`strength = ${tomlFloat(c.strength ?? 1.0)}`);
  if (c.motion_tail_frames != null) lines.push(`motion_tail_frames = ${c.motion_tail_frames}`);
  lines.push(`output_format = ${tomlString(c.output_format ?? "mp4")}`);
  if (c.enable_audio != null) lines.push(`enable_audio = ${c.enable_audio}`);

  for (const stage of script.stages) {
    lines.push("", "[[stage]]");
    lines.push(`prompt = ${tomlString(stage.prompt)}`);
    if (stage.frames != null) lines.push(`frames = ${stage.frames}`);
    lines.push(`transition = ${tomlString(stage.transition ?? "smooth")}`);
    if ((stage.transition ?? "smooth") === "fade" && stage.fade_frames != null) {
      lines.push(`fade_frames = ${stage.fade_frames}`);
    }
    if (stage.negative_prompt) {
      lines.push(`negative_prompt = ${tomlString(stage.negative_prompt)}`);
    }
    if (stage.source_image_b64) {
      lines.push(`source_image_b64 = ${tomlString(stage.source_image_b64)}`);
    }
    if (stage.seed_offset != null) lines.push(`seed_offset = ${tomlInt(stage.seed_offset)}`);
  }

  return `${lines.join("\n")}\n`;
}

/**
 * Extract a stage's inline starting image. Mirrors mold-core's
 * `resolve_stage_source_image`: authoring `source_image_b64` and canonical
 * `source_image` are both accepted, at most one image field may be set, and
 * `source_image_path` is rejected — the composer has no script folder to
 * resolve it against.
 */
function stageSourceImage(s: Record<string, unknown>, idx: number): string | undefined {
  const present = ["source_image", "source_image_path", "source_image_b64"].filter(
    (key) => s[key] != null,
  );
  if (present.length > 1) {
    throw new Error(
      `Stage ${idx + 1}: set at most one of source_image, source_image_path, source_image_b64.`,
    );
  }
  if (s.source_image_path != null) {
    throw new Error(
      `Stage ${idx + 1} uses source_image_path, which needs the script's folder to resolve. ` +
        "Inline the image as source_image_b64 to open it here.",
    );
  }
  const value = s.source_image_b64 ?? s.source_image;
  if (value == null) return undefined;
  if (typeof value !== "string") {
    throw new Error(`Stage ${idx + 1}: ${present[0]} must be a string.`);
  }
  return value.length > 0 ? value : undefined;
}

/**
 * Parse `mold.chain.v1` TOML into a studio `ChainScript`. Defensive:
 * unknown fields are ignored; malformed TOML, foreign schemas, and missing
 * `[chain]` / `[[stage]]` tables throw friendly errors.
 */
export function parseChainScript(text: string): ChainScript {
  let doc: Record<string, unknown>;
  try {
    // u64 seeds overflow JS numbers; "asNeeded" promotes only those to bigint.
    doc = parseToml(text, { integersAsBigInt: "asNeeded" }) as Record<string, unknown>;
  } catch (e) {
    throw new Error(`Couldn't parse the TOML: ${e instanceof Error ? e.message : String(e)}`);
  }

  const schema = doc.schema;
  if (schema != null && schema !== CHAIN_SCHEMA) {
    throw new Error(`Unsupported chain schema "${String(schema)}" (expected ${CHAIN_SCHEMA}).`);
  }
  const chain = doc.chain;
  if (chain == null || typeof chain !== "object") {
    throw new Error("Chain TOML is missing its [chain] table.");
  }
  const c = chain as Record<string, unknown>;
  // The canonical key is `stage` (Rust serde rename); the natural plural is
  // accepted too so hand-written documents open.
  const stageTables = doc.stage ?? doc.stages;
  const rawStages = Array.isArray(stageTables) ? (stageTables as Record<string, unknown>[]) : [];
  if (rawStages.length === 0) {
    throw new Error("Chain TOML is missing its [[stage]] entries.");
  }

  const stages: ChainScriptStage[] = rawStages.map((s, idx) => {
    const stage: ChainScriptStage = {
      prompt: str(s.prompt) ?? "",
      transition: idx === 0 ? "smooth" : asTransition(s.transition),
    };
    const frames = num(s.frames);
    if (frames !== undefined) stage.frames = frames;
    const fadeFrames = num(s.fade_frames);
    if (fadeFrames !== undefined) stage.fade_frames = fadeFrames;
    const negative = str(s.negative_prompt);
    if (negative) stage.negative_prompt = negative;
    const image = stageSourceImage(s, idx);
    if (image !== undefined) stage.source_image_b64 = image;
    const seedOffset = seedString(s.seed_offset);
    if (seedOffset !== undefined) stage.seed_offset = seedOffset;
    return stage;
  });

  const script: ChainScript = {
    schema: CHAIN_SCHEMA,
    chain: { model: str(c.model) ?? "" },
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
  const motionTail = num(c.motion_tail_frames);
  if (motionTail !== undefined) script.chain.motion_tail_frames = motionTail;
  const seed = seedString(c.seed);
  if (seed !== undefined) script.chain.seed = seed;
  if (c.enable_audio === true) script.chain.enable_audio = true;
  return script;
}
