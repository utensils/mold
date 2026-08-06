/** The mobile bundle can also run in a desktop browser for development. Only
 * the native iOS build may replace local downloads with the system share sheet. */
export function isNativeIOSRuntime(): boolean {
  return import.meta.env.TAURI_ENV_PLATFORM === "ios";
}
