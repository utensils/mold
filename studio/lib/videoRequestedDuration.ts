/** Milliseconds remain independent of any model's generation frame grid.
 * Only the selected server decides the bounded clips and final trim. */
export function parseRequestedVideoDuration(seconds: string): string | null {
  const text = seconds.trim();
  if (!/^(?:\d+(?:\.\d{0,3})?|\.\d{1,3})$/.test(text)) return null;
  const [whole = "", fraction = ""] = text.split(".");
  const millis = BigInt(whole || "0") * 1000n + BigInt(fraction.padEnd(3, "0"));
  return millis > 0n && millis <= 18_446_744_073_709_551_615n
    ? millis.toString()
    : null;
}
export function formatRequestedVideoDuration(millis: string): string {
  if (!/^[1-9]\d*$/.test(millis)) return "";
  const value = BigInt(millis);
  const fractional = String(value % 1000n)
    .padStart(3, "0")
    .replace(/0+$/, "");
  return `${value / 1000n}${fractional ? `.${fractional}` : ""}`;
}
export interface VideoDurationSummary {
  fps: number;
  single_clip_max_frames: number;
  clip_count: string;
}
function validRate(plan: { fps: number; single_clip_max_frames: number }) {
  return (
    Number.isFinite(plan.fps) &&
    plan.fps > 0 &&
    Number.isSafeInteger(plan.single_clip_max_frames) &&
    plan.single_clip_max_frames >= 1
  );
}
/** The longest video the model makes in one generation, in exact seconds
 * (`345` frames at 24 fps → `14.375`). */
export function singleGenerationSeconds(plan: {
  fps: number;
  single_clip_max_frames: number;
}): string | null {
  if (!validRate(plan)) return null;
  return (
    formatRequestedVideoDuration(
      String(Math.floor((plan.single_clip_max_frames * 1000) / plan.fps)),
    ) || null
  );
}
/** The longest scene that always stays one clip: one generation less the
 * frames a following clip overlaps, floored to a tenth of a second so a scene
 * of exactly this length never spills into a second clip. */
export function oneClipSceneSeconds(plan: {
  fps: number;
  single_clip_max_frames: number;
  overlap_frames?: number;
}): string | null {
  if (!validRate(plan) || !Number.isSafeInteger(plan.overlap_frames))
    return null;
  const tenths = Math.floor(
    ((plan.single_clip_max_frames - plan.overlap_frames!) * 10) / plan.fps,
  );
  if (tenths <= 0) return null;
  return tenths % 10 === 0
    ? String(tenths / 10)
    : `${Math.floor(tenths / 10)}.${tenths % 10}`;
}
