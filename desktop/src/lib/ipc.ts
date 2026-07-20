import type {
  RunPodCreateInput,
  RunPodNetworkVolume,
  RunPodNetworkVolumeCreateInput,
  RunPodNetworkVolumeUpdateInput,
  RunPodOverview,
  RunPodPod,
} from "./runpod";
import type { GalleryImage, OutputMetadata } from "./api/types";
import type { DesktopImageImport } from "./desktopImageDrop";
import type { Theme, ThemeFamily } from "./theme";

export type { Theme, ThemeFamily } from "./theme";

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

export interface LocalServerInfo {
  kind: "embedded" | "external";
  baseUrl: string;
  apiKey: string | null;
  port: number;
}

export type UpdateChannel = "stable" | "nightly";

export interface UpdateCandidate {
  /** Opaque backend-issued identifier. The webview never supplies update URLs. */
  id: string;
  version: string;
  publishedAt: string | null;
  notes: string | null;
}

export interface UpdateCheckResult {
  supported: boolean;
  channel: UpdateChannel;
  currentVersion: string;
  checkedAt: string;
  candidate: UpdateCandidate | null;
}

export type UpdateProgressPhase = "downloading" | "verifying" | "staging" | "installing";

export interface UpdateProgress {
  candidateId: string;
  phase: UpdateProgressPhase;
  downloadedBytes: number | null;
  totalBytes: number | null;
}

/** A remote host the app has connected to before (most recent first). */
export interface SavedHost {
  /** URL-derived slug; its API key lives at secret `remote-api-key.<id>`. */
  id: string;
  name: string | null;
  url: string;
  lastUsedMs: number | null;
  /** Stable server-installation UUID; used to dedupe the same box reached by a
   *  different address. Absent on older servers / entries saved before it. */
  instanceId?: string | null;
}

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
  runpodIncludeHfToken: boolean;
  runpodNetworkVolumeId: string | null;
  uiScalePercent: number;
  /** Signed desktop release stream. Nightly follows builds from main. */
  updateChannel: UpdateChannel;
  /** Remote hosts the app remembers, most recently used first. */
  savedHosts: SavedHost[];
  /** Legacy reconnect markers; every saved host is now attempted at boot. */
  connectedHostIds: string[];
  /** Sticky generation-target host id; null routes automatically. */
  generateTargetHost: string | null;
  /** Also save generations from remote hosts into this Mac's gallery. */
  saveRemoteOutputs: boolean;
  /** Persisted sidebar width in px; null uses the panel default. */
  navRailWidth: number | null;
  /** Persisted Generate-inspector width in px; null uses the panel default. */
  generateParamsWidth: number | null;
}

export interface HostTest {
  ok: boolean;
  version: string | null;
  error: string | null;
  /** Stable server-installation UUID from `/api/status`; absent on older servers. */
  instanceId?: string | null;
  /** Server-reported hostname from `/api/status`; absent on older servers. */
  hostname?: string | null;
}

/** A mold server discovered on the local network via mDNS. */
export interface DiscoveredHost {
  name: string;
  url: string;
  host: string;
  port: number;
  version: string | null;
  authRequired: boolean;
  /** Stable server-installation UUID from the mDNS `id` TXT record; absent on older servers. */
  instanceId?: string | null;
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
  runpodIncludeHfToken: false,
  runpodNetworkVolumeId: null,
  uiScalePercent: 100,
  updateChannel: "stable",
  savedHosts: [],
  connectedHostIds: [],
  generateTargetHost: null,
  saveRemoteOutputs: true,
  navRailWidth: null,
  generateParamsWidth: null,
});

export const ipc = {
  getConnection(): Promise<ConnectionInfo> {
    if (!inTauri()) return Promise.resolve(browserFallbackConnection());
    return invoke<ConnectionInfo>("get_connection");
  },
  ensureLocalServer(): Promise<LocalServerInfo> {
    if (!inTauri()) {
      const fallback = browserFallbackConnection();
      return Promise.resolve({
        kind: "external",
        baseUrl: fallback.baseUrl!,
        apiKey: fallback.apiKey,
        port: Number(new URL(fallback.baseUrl!).port || 7680),
      });
    }
    return invoke<LocalServerInfo>("ensure_local_server");
  },
  startLocalEngine(): Promise<ConnectionInfo> {
    if (!inTauri()) return Promise.resolve(browserFallbackConnection());
    return invoke<ConnectionInfo>("start_local_engine");
  },
  stopLocalEngine(): Promise<ConnectionInfo> {
    if (!inTauri()) return Promise.resolve(browserFallbackConnection());
    return invoke<ConnectionInfo>("stop_local_engine");
  },
  /** Drop a host from the saved list and delete its stored API key. */
  forgetRemoteHost(id: string): Promise<SavedHost[]> {
    if (!inTauri()) return Promise.resolve([]);
    return invoke<SavedHost[]>("forget_remote_host", { id });
  },
  testRemoteHost(url: string, apiKey: string | null): Promise<HostTest> {
    if (!inTauri())
      return Promise.resolve({
        ok: true,
        version: null,
        error: null,
        instanceId: null,
        hostname: null,
      });
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
  checkForUpdates(channel: UpdateChannel): Promise<UpdateCheckResult> {
    if (!inTauri()) {
      return Promise.resolve({
        supported: false,
        channel,
        currentVersion: "dev",
        checkedAt: new Date().toISOString(),
        candidate: null,
      });
    }
    return invoke<UpdateCheckResult>("check_for_updates", { channel });
  },
  installPendingUpdate(candidateId: string): Promise<void> {
    if (!inTauri()) return Promise.reject(new Error("Updates require a signed desktop build."));
    return invoke<void>("install_pending_update", { candidateId });
  },
  async onUpdaterProgress(listener: (progress: UpdateProgress) => void): Promise<() => void> {
    if (!inTauri()) return () => {};
    const { listen } = await import("@tauri-apps/api/event");
    return listen<UpdateProgress>("updater-progress", ({ payload }) => listener(payload));
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
  localGalleryList(): Promise<GalleryImage[]> {
    if (!inTauri()) return Promise.resolve([]);
    return invoke<GalleryImage[]>("local_gallery_list");
  },
  localGalleryDelete(filename: string): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("local_gallery_delete", { filename });
  },
  /** Read a native OS-dropped PNG/JPEG plus any embedded Mold metadata. */
  importSourceImage(path: string): Promise<DesktopImageImport> {
    if (!inTauri()) return Promise.reject(new Error("Native file drops require the desktop app."));
    return invoke<DesktopImageImport>("import_source_image", { path });
  },
  /**
   * Open the platform file picker for still-image conditioning inputs. The
   * native dialog avoids the WebView file-control stall, and the backend
   * validates the chosen files by decoding them rather than trusting this
   * suffix filter.
   */
  async pickSourceImages(multiple: boolean): Promise<DesktopImageImport[] | null> {
    if (!inTauri()) return null;
    const { open } = await import("@tauri-apps/plugin-dialog");
    const picked = await open({
      title: "Choose image",
      multiple,
      filters: [{ name: "PNG or JPEG images", extensions: ["png", "jpg", "jpeg"] }],
    });
    if (!picked) return null;
    const paths = Array.isArray(picked) ? picked : [picked];
    return Promise.all(
      paths.map((path) => invoke<DesktopImageImport>("import_source_image", { path })),
    );
  },
  /** Write encoded output bytes (base64) into this Mac's output dir. */
  saveOutputBytes(
    filename: string,
    dataB64: string,
    metadata?: OutputMetadata | null,
  ): Promise<string> {
    if (!inTauri()) return Promise.reject(new Error("Local saves require the desktop app."));
    return invoke<string>("save_output_bytes", { filename, dataB64, metadata: metadata ?? null });
  },
  /** Stash the exact img2img source bytes under their sha256 (fire-and-forget
   * at submit) so Reuse settings can restore uploads that live nowhere else. */
  sourceStashPut(sha256: string, dataB64: string): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("source_stash_put", { sha256, dataB64 });
  },
  /** Read a stashed source back by sha256; null when absent (or pruned). */
  sourceStashGet(sha256: string): Promise<string | null> {
    if (!inTauri()) return Promise.resolve(null);
    return invoke<string | null>("source_stash_get", { sha256 });
  },
  clipboardWriteImage(bytes: Uint8Array): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("clipboard_write_image", { bytes });
  },
  /** macOS dock badge; null clears it. */
  setDockBadge(count: number | null): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("set_dock_badge", { count }).catch(() => {});
  },
  /** Use a platform-native notification when it can preserve Mold's app icon. */
  sendNativeNotification(title: string, body?: string): Promise<boolean> {
    if (!inTauri()) return Promise.resolve(false);
    return invoke<boolean>("send_native_notification", { title, body: body ?? null }).catch(
      () => false,
    );
  },
  /** Open the engine's log directory in Finder. */
  openLogsDir(): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("open_logs_dir");
  },
  runpodOverview(): Promise<RunPodOverview> {
    if (!inTauri()) {
      return Promise.resolve({
        configured: false,
        credentialSource: null,
        account: null,
        pods: [],
        gpus: [],
        datacenters: [],
        networkVolumes: [],
      });
    }
    return invoke<RunPodOverview>("runpod_overview");
  },
  runpodCreate(input: RunPodCreateInput): Promise<RunPodPod> {
    if (!inTauri())
      return Promise.reject(new Error("RunPod provisioning requires the desktop app."));
    return invoke<RunPodPod>("runpod_create", { input });
  },
  runpodNetworkVolumeCreate(input: RunPodNetworkVolumeCreateInput): Promise<RunPodNetworkVolume> {
    if (!inTauri())
      return Promise.reject(new Error("RunPod volume management requires the desktop app."));
    return invoke<RunPodNetworkVolume>("runpod_network_volume_create", { input });
  },
  runpodNetworkVolumeUpdate(input: RunPodNetworkVolumeUpdateInput): Promise<RunPodNetworkVolume> {
    if (!inTauri())
      return Promise.reject(new Error("RunPod volume management requires the desktop app."));
    return invoke<RunPodNetworkVolume>("runpod_network_volume_update", { input });
  },
  runpodNetworkVolumeDelete(id: string): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("runpod_network_volume_delete", { id });
  },
  runpodStart(id: string): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("runpod_start", { id });
  },
  runpodStop(id: string): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("runpod_stop", { id });
  },
  runpodDelete(id: string): Promise<void> {
    if (!inTauri()) return Promise.resolve();
    return invoke<void>("runpod_delete", { id });
  },
  /** Secrets in an owner-only local file (secrets.json). Names are allowlisted. */
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

export type SecretName =
  | "hf-token"
  | "civitai-token"
  | "remote-api-key"
  | "runpod-api-key"
  | "desktop-local-api-key"
  | `remote-api-key.${string}`;
