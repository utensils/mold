import { invoke } from "@tauri-apps/api/core";
import { isNativeIOSRuntime } from "./platform";

export interface MobileBackgroundTaskLease {
  readonly active: boolean;
  release(): Promise<void>;
}

const BACKGROUND_TASK_SAFETY_RELEASE_MS = 25_000;

const NO_BACKGROUND_TASK: MobileBackgroundTaskLease = {
  active: false,
  release: () => Promise.resolve(),
};

/**
 * Keep WKWebView network and preprocessing work runnable for iOS's finite
 * background grace period. Failure to acquire the assertion is non-fatal: the
 * host still owns already-admitted jobs, and foreground recovery remains the
 * authority after iOS suspends the WebView.
 */
export async function beginMobileBackgroundTask(name: string): Promise<MobileBackgroundTaskLease> {
  if (!isNativeIOSRuntime()) return NO_BACKGROUND_TASK;

  let token: string;
  try {
    token = await invoke<string>("begin_mobile_background_task", { name });
  } catch (error) {
    console.warn("iOS background execution was unavailable:", error);
    return NO_BACKGROUND_TASK;
  }

  let released = false;
  let safetyRelease: ReturnType<typeof setTimeout> | null = null;
  const release = async () => {
    if (released) return;
    released = true;
    if (safetyRelease !== null) clearTimeout(safetyRelease);
    safetyRelease = null;
    try {
      await invoke("end_mobile_background_task", { token });
    } catch (error) {
      // Expiration removes the native token first. A later WebView cleanup
      // remains best-effort and must never turn accepted work into a failure.
      console.warn("iOS background execution cleanup failed:", error);
    }
  };
  safetyRelease = setTimeout(() => {
    safetyRelease = null;
    void release();
  }, BACKGROUND_TASK_SAFETY_RELEASE_MS);
  return {
    active: true,
    release,
  };
}
