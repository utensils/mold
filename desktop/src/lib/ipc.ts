/**
 * Typed wrappers around Tauri IPC. In a plain browser (`bun run dev` /
 * desktop-ui against a running `mold serve`) there is no Tauri runtime, so
 * every wrapper degrades to a sensible fallback and the app stays usable.
 */

export interface ConnectionInfo {
  mode: "off" | "local" | "external" | "remote";
  baseUrl: string | null;
  apiKey: string | null;
}

export type Theme = "system" | "dark" | "light";
export type ThemeFamily = "safelight" | "mold";

export interface AppSettings {
  mode: "local" | "remote" | "off";
  remoteUrl: string | null;
  remoteApiKey: string | null;
  lastRoute: string | null;
  /** Env applied to the embedded engine at start (Performance knobs). */
  engineEnv: Record<string, string>;
  theme: Theme;
  /** Visual palette. Appearance remains independent so every family supports system/dark/light. */
  themeFamily: ThemeFamily;
  notifications: boolean;
  dockBadge: boolean;
  restoreLastRoute: boolean;
}

export interface HostTest {
  ok: boolean;
  version: string | null;
  error: string | null;
}

/** A mold server discovered on the local network via mDNS. */
export interface DiscoveredHost {
  name: string;
  url: string;
  host: string;
  port: number;
  version: string | null;
  authRequired: boolean;
  isThisMachine: boolean;
}

export const inTauri = (): boolean =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(cmd, args);
}

/** Engine the browser build talks to when there is no Tauri backend. */
const browserFallbackConnection = (): ConnectionInfo => ({
  mode: "external",
  baseUrl: import.meta.env.VITE_MOLD_HOST ?? "http://localhost:7680",
  apiKey: import.meta.env.VITE_MOLD_API_KEY ?? null,
});

const browserFallbackSettings = (): AppSettings => ({
  mode: "local",
  remoteUrl: null,
  remoteApiKey: null,
  lastRoute: null,
  engineEnv: {},
  theme: "system",
  themeFamily: "mold",
  notifications: true,
  dockBadge: true,
  restoreLastRoute: false,
});

export const ipc = {
  getConnection(): Promise<ConnectionInfo> {
    if (!inTauri()) return Promise.resolve(browserFallbackConnection());
    return invoke<ConnectionInfo>("get_connection");
  },
  startLocalEngine(): Promise<ConnectionInfo> {
    if (!inTauri()) return Promise.resolve(browserFallbackConnection());
    return invoke<ConnectionInfo>("start_local_engine");
  },
  stopLocalEngine(): Promise<ConnectionInfo> {
    if (!inTauri()) return Promise.resolve(browserFallbackConnection());
    return invoke<ConnectionInfo>("stop_local_engine");
  },
  setRemoteHost(url: string, apiKey: string | null): Promise<ConnectionInfo> {
    if (!inTauri()) return Promise.resolve(browserFallbackConnection());
    return invoke<ConnectionInfo>("set_remote_host", { url, apiKey });
  },
  testRemoteHost(url: string, apiKey: string | null): Promise<HostTest> {
    if (!inTauri()) return Promise.resolve({ ok: true, version: null, error: null });
    return invoke<HostTest>("test_remote_host", { url, apiKey });
  },
  /** Browse the LAN for advertised mold servers; empty in a plain browser. */
  discoverServers(timeoutMs?: number): Promise<DiscoveredHost[]> {
    if (!inTauri()) return Promise.resolve([]);
    return invoke<DiscoveredHost[]>("discover_servers", { timeoutMs });
  },
  appSettingsGet(): Promise<AppSettings> {
    if (!inTauri()) return Promise.resolve(browserFallbackSettings());
    return invoke<AppSettings>("app_settings_get");
  },
  appSettingsSet(settings: AppSettings): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("app_settings_set", { settings });
  },
  /** Where the local engine writes gallery files; null on remote hosts. */
  getOutputDir(): Promise<string | null> {
    if (!inTauri()) return Promise.resolve(null);
    return invoke<string | null>("get_output_dir");
  },
  revealOutputFile(filename: string): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("reveal_output_file", { filename });
  },
  /** macOS dock badge; null clears it. */
  setDockBadge(count: number | null): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("set_dock_badge", { count }).catch(() => {});
  },
  /** Open the engine's log directory in Finder. */
  openLogsDir(): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("open_logs_dir");
  },
  /** Keychain-backed secrets (file fallback in dev). Names are allowlisted. */
  secretGet(name: SecretName): Promise<string | null> {
    if (!inTauri()) return Promise.resolve(null);
    return invoke<string | null>("secret_get", { name });
  },
  secretSet(name: SecretName, value: string): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("secret_set", { name, value });
  },
  secretClear(name: SecretName): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("secret_clear", { name });
  },
  /** Native folder picker; null when cancelled (or in a plain browser). */
  async pickDirectory(title: string): Promise<string | null> {
    if (!inTauri()) return null;
    const { open } = await import("@tauri-apps/plugin-dialog");
    const picked = await open({ directory: true, multiple: false, title });
    return typeof picked === "string" ? picked : null;
  },
};

export type SecretName = "hf-token" | "civitai-token" | "remote-api-key";
