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
