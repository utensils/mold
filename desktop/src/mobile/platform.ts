/** The mobile bundle can also run in a desktop browser for development. */
export function isNativeIOSRuntime(platform = import.meta.env.TAURI_ENV_PLATFORM): boolean {
  return platform === "ios";
}

export function isNativeAndroidRuntime(platform = import.meta.env.TAURI_ENV_PLATFORM): boolean {
  return platform === "android";
}
