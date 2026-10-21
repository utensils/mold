import type { VideoDurationAuthoring } from "./videoDurationAuthoring";
const MAX_U64 = 18_446_744_073_709_551_615n;
function decimal(value: unknown, zero = false): string {
  if (
    typeof value !== "string" ||
    !/^(0|[1-9]\d*)$/.test(value) ||
    value.length > 20 ||
    BigInt(value) > MAX_U64 ||
    (!zero && value === "0")
  )
    throw new Error(
      "This video's authoring settings are incomplete. Update its machine before using these settings.",
    );
  return value;
}
function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error(
      "This video's authoring settings are unavailable. Update its machine before using these settings.",
    );
  return value as Record<string, unknown>;
}
/** Published authoring describes NEW footage; it never reconnects a previous prefix. */
export function videoSettingsForReuse(
  metadata: object,
): { draft: VideoDurationAuthoring; frames: number; seed: string } | null {
  const m = metadata as Record<string, unknown>;
  if (!m.video_job) return null;
  const job = record(m.video_job);
  if (job.authoring == null)
    throw new Error(
      "This older video did not save its original duration/scenes. Use Start again from its saved job, or create a draft and choose duration explicitly.",
    );
  const a = record(job.authoring);
  if (new TextEncoder().encode(JSON.stringify(a)).byteLength > 1024 * 1024)
    throw new Error(
      "This video's saved authoring settings exceed the supported size.",
    );
  const durationMs = decimal(a.duration_ms),
    seed = decimal(a.seed, true);
  if (
    !Number.isSafeInteger(a.boundary_frames) ||
    (a.boundary_frames as number) < 1 ||
    (a.boundary_frames as number) > 0xffffffff
  )
    throw new Error(
      "This video's authoring settings have an invalid clip boundary.",
    );
  const frames = a.input_frames ?? a.boundary_frames;
  if (
    !Number.isSafeInteger(frames) ||
    (frames as number) < 1 ||
    (frames as number) > 0xffffffff
  )
    throw new Error(
      "This video's authoring settings have an invalid frame boundary.",
    );
  if (a.scenes != null && !Array.isArray(a.scenes))
    throw new Error("This video's scene settings are unavailable.");
  const raw = (a.scenes ?? []) as unknown[];
  if (raw.length > 4096)
    throw new Error("This video has too many scenes to restore.");
  const scenes = raw.map((value) => {
    const s = record(value);
    if (
      typeof s.prompt !== "string" ||
      !s.prompt.trim() ||
      new TextEncoder().encode(s.prompt).byteLength > 77_000
    )
      throw new Error("This video's scene prompt is missing.");
    return {
      id: crypto.randomUUID(),
      prompt: s.prompt,
      durationMs: decimal(s.duration_ms),
    };
  });
  if (
    scenes.length &&
    scenes.reduce((n, s) => n + BigInt(s.durationMs), 0n) !== BigInt(durationMs)
  )
    throw new Error(
      "This video's scene durations do not match its requested duration.",
    );
  return {
    frames: frames as number,
    seed,
    draft: {
      durationMs,
      scenesEnabled: scenes.length > 0,
      scenes,
      ...(job.source_frames && decimal(job.source_frames, true) !== "0"
        ? {
            reuseNotice:
              "These settings make new footage. The original video is not attached.",
          }
        : {}),
    },
  };
}
/** JSON can carry a full u64 as a checked decimal string without rounding. */
export function exactSeed(value: unknown): value is number | string {
  return typeof value === "number"
    ? Number.isSafeInteger(value) && value >= 0
    : typeof value === "string" &&
        /^(0|[1-9]\d*)$/.test(value) &&
        value.length <= 20 &&
        BigInt(value) <= MAX_U64;
}
export function incrementExactSeed(
  seed: number | string,
  offset: number,
): number | string {
  if (!exactSeed(seed) || !Number.isSafeInteger(offset) || offset < 0)
    throw new Error("Choose a valid seed and batch size.");
  const next = BigInt(seed) + BigInt(offset);
  if (next > MAX_U64)
    throw new Error(
      "The seed plus batch size exceeds the supported seed range.",
    );
  return typeof seed === "string" || next > BigInt(Number.MAX_SAFE_INTEGER)
    ? String(next)
    : Number(next);
}
export function seedFromInput(value: string): number | string | null {
  if (value === "") return null;
  if (exactSeed(value) && BigInt(value) <= BigInt(Number.MAX_SAFE_INTEGER))
    return Number(value);
  return value;
}

/** Prefer retained decimal authority; never present an already-rounded JSON u64 as exact. */
export function publishedSeed(metadata: object): number | string | null {
  const m = metadata as Record<string, unknown>;
  const job = m.video_job as { authoring?: { seed?: unknown } } | undefined;
  const chain = m.chain as { stages?: Array<{ seed?: unknown }> } | undefined;
  for (const value of [job?.authoring?.seed, chain?.stages?.[0]?.seed, m.seed])
    if (exactSeed(value)) return value;
  return null;
}
