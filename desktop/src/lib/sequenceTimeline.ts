/**
 * What the clip timeline says and what it asks. The timeline, the lane it
 * renders and the composer above it are three components that must never name
 * the same thing two ways, and the timeline's dialogs are rendered by the
 * workbench rather than by the timeline itself.
 */

/**
 * What a scene is called: its own words, or its place in the clip. The lane's
 * block title and the timeline's menus and messages all read it, so a scene
 * cannot be named two things on one screen. `clipLabel` in `ui/lib/duration.ts`
 * is the web filmstrip's own wording and stays.
 */
export function sceneLabel(prompt: string, index: number, maxLength?: number): string {
  const written = prompt.trim();
  if (!written) return index === 0 ? "Opening scene" : `Scene ${index + 1}`;
  if (maxLength !== undefined && written.length > maxLength) {
    return `${written.slice(0, maxLength - 1)}…`;
  }
  return written;
}

/**
 * Why Generate refuses in clip mode with no video style to make one with. The
 * timeline raises it while it is mounted; New image answers with the same
 * sentence while the inventory is empty and the timeline is not there at all.
 */
export const SEQUENCE_NEEDS_STYLE = "Pick a video style first.";

/**
 * A destructive question the timeline needs answered. The timeline decides it
 * and owns the state behind it, but the DIALOG is rendered by the workbench:
 * the bench strip declares `container-type: size`, which makes it the
 * containing block for every absolutely positioned descendant, so a dialog
 * inside it would centre in the strip instead of over the app.
 */
export interface SequenceConfirmation {
  title: string;
  message: string;
  confirmLabel: string;
  confirm: () => void;
  cancel: () => void;
}
