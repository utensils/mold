/**
 * Native OS notifications for background completion events. Foreground runs
 * get toasts instead, so every helper no-ops unless the window is hidden or
 * unfocused. In a plain browser (no Tauri) these are all no-ops.
 */
import { inTauri, ipc } from "./ipc";
import { useAppPrefsStore } from "../stores/appPrefs";

let permissionResolved = false;
let granted = false;

async function ensurePermission(): Promise<boolean> {
  if (!inTauri()) return false;
  if (permissionResolved) return granted;
  permissionResolved = true;
  const { isPermissionGranted, requestPermission } =
    await import("@tauri-apps/plugin-notification");
  granted = await isPermissionGranted();
  if (!granted) granted = (await requestPermission()) === "granted";
  return granted;
}

/** True when the app is backgrounded — the moment a notification earns its keep. */
export function appIsBackground(): boolean {
  return document.hidden || !document.hasFocus();
}

async function notify(title: string, body?: string): Promise<void> {
  if (!appIsBackground()) return;
  try {
    if (!useAppPrefsStore().notifications) return;
  } catch {
    /* store not ready (early boot) — default on */
  }
  if (await ipc.sendNativeNotification(title, body)) return;
  if (!(await ensurePermission())) return;
  const { sendNotification } = await import("@tauri-apps/plugin-notification");
  sendNotification(body != null ? { title, body } : { title });
}

function dispatchNotification(title: string, body?: string): void {
  void notify(title, body).catch(() => {
    /* notifications are best effort */
  });
}

export function notifyGenerated(prompt: string): void {
  dispatchNotification(`Generated — ${prompt.trim().slice(0, 40)}`);
}
export function notifyGenerationFailed(message: string): void {
  dispatchNotification("Generation failed", message.slice(0, 80));
}
export function notifyChainFinished(frames: number): void {
  dispatchNotification(`Chain finished · ${frames} frames`);
}
export function notifyPulled(model: string): void {
  dispatchNotification(`Pulled ${model}`);
}
