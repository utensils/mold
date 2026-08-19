/*
 * Copying a notification out of the bell. The bell is the durable record of
 * every toast — a full server error body lands there — and that text is only
 * useful if it can leave the app: into an issue, a chat, a search. The app
 * shells disable text selection on their chrome, so a deliberate copy control
 * is the only way out, and this is the one place that decides what it yields.
 */
import type { NotificationEntry } from "../stores/notifications";

/**
 * The clipboard payload for one entry: the message, its full untruncated
 * supporting copy, then the origin/time line the row shows. `timeLabel` comes
 * from the caller because the rendered time is locale-formatted there.
 */
export function notificationClipboardText(
  entry: Pick<NotificationEntry, "text" | "description" | "hostLabel">,
  timeLabel?: string | null,
): string {
  const meta = [entry.hostLabel, timeLabel].filter(
    (part): part is string => !!part && part.length > 0,
  );
  return [entry.text, entry.description ?? "", meta.join(" · ")]
    .filter((line) => line.length > 0)
    .join("\n");
}

/**
 * Write text to the clipboard, degrading to the legacy `execCommand` path.
 * `navigator.clipboard` is absent in insecure contexts (a LAN server reached
 * over plain http is the normal way mold is used) and can reject when the
 * document is not focused, so the fallback is load-bearing rather than legacy
 * politeness. Returns false when nothing could be copied.
 */
export async function copyTextToClipboard(text: string): Promise<boolean> {
  if (!text) return false;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // Fall through to the textarea path.
  }
  // Selecting the staging node steals focus, and removing it would then strand
  // focus on <body> — restoring it keeps a keyboard user's place in the list.
  const previouslyFocused = document.activeElement;
  const area = document.createElement("textarea");
  try {
    area.value = text;
    // Off-screen but focusable; readOnly keeps the mobile keyboard away.
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.top = "-1000px";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    return document.execCommand?.("copy") ?? false;
  } catch {
    return false;
  } finally {
    // A throwing execCommand must not strand the staging node in the document;
    // every failed Copy would otherwise add another invisible textarea.
    area.remove();
    if (previouslyFocused instanceof HTMLElement) previouslyFocused.focus();
  }
}
