import type { ConnectionInfo } from "./ipc";

/** Only the native embedded engine can be restarted by this app. */
export function shouldRestartEmbeddedEngine(
  mode: ConnectionInfo["mode"],
  consecutiveFailures: number,
): boolean {
  return mode === "local" && consecutiveFailures >= 2;
}
