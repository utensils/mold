import type { VideoClipDraft } from "./chainDraft";

/** Text-only UI drafts. Media and credentials stay with their existing owners.
 * The retired sequence sweeper deliberately does not own this new namespace. */
const KEY = "mold.video-clips.drafts.v1";
interface SavedDraft {
  key: string;
  clips: VideoClipDraft[];
  updatedAt: number;
}
function read(): SavedDraft[] {
  const value: unknown = JSON.parse(localStorage.getItem(KEY) ?? "[]");
  if (!Array.isArray(value))
    throw new Error("Saved video clip drafts are unreadable.");
  return value.filter(
    (row): row is SavedDraft =>
      row &&
      typeof row.key === "string" &&
      Array.isArray(row.clips) &&
      row.clips.length <= 16 &&
      row.clips.every(
        (clip: VideoClipDraft) =>
          clip &&
          typeof clip.id === "string" &&
          typeof clip.prompt === "string" &&
          Number.isSafeInteger(clip.frames) &&
          clip.frames > 0,
      ),
  );
}
export function loadVideoClipsDraft(key: string): VideoClipDraft[] | null {
  return read().find((draft) => draft.key === key)?.clips ?? null;
}
export function saveVideoClipsDraft(
  key: string,
  clips: readonly VideoClipDraft[],
): void {
  const rows = read().filter((draft) => draft.key !== key);
  // Write only explicit text fields, even if a caller later extends its draft.
  rows.push({
    key,
    clips: clips.map(({ id, prompt, frames }) => ({ id, prompt, frames })),
    updatedAt: Date.now(),
  });
  localStorage.setItem(KEY, JSON.stringify(rows.slice(-8)));
}
