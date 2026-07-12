import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useConnectionStore } from "./connection";
import { useToastStore } from "./toasts";
import { ipc } from "../lib/ipc";

vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: vi.fn(),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
    startLocalEngine: vi.fn(),
    stopLocalEngine: vi.fn(),
    setRemoteHost: vi.fn(),
    testRemoteHost: vi.fn(),
    getConnection: vi.fn(),
    secretGet: vi.fn().mockResolvedValue(null),
  },
}));

const mocked = vi.mocked(ipc);

const local = { mode: "local" as const, baseUrl: "http://127.0.0.1:49152", apiKey: "k" };
const remote = { mode: "remote" as const, baseUrl: "http://studio.local:7680", apiKey: null };
const defaults = {
  mode: "local" as const,
  remoteUrl: null,
  remoteApiKey: null,
  lastRoute: null,
  engineEnv: {},
  theme: "system" as const,
  themeFamily: "safelight" as const,
  notifications: true,
  dockBadge: true,
  restoreLastRoute: false,
  runpodIncludeHfToken: false,
  runpodNetworkVolumeId: null,
  uiScalePercent: 100,
  updateChannel: "stable" as const,
  savedHosts: [],
  connectedHostIds: [],
  generateTargetHost: null,
};

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
});

describe("connection store", () => {
  it("boots the local engine when settings say local", async () => {
    mocked.appSettingsGet.mockResolvedValue(defaults);
    mocked.startLocalEngine.mockResolvedValue(local);
    const store = useConnectionStore();
    await store.init();
    expect(store.ready).toBe(true);
    expect(store.baseUrl).toBe(local.baseUrl);
    expect(store.mode).toBe("local");
  });

  it("reconnects a saved remote host", async () => {
    mocked.appSettingsGet.mockResolvedValue({
      ...defaults,
      mode: "remote",
      remoteUrl: remote.baseUrl,
    });
    mocked.setRemoteHost.mockResolvedValue(remote);
    const store = useConnectionStore();
    await store.init();
    expect(mocked.setRemoteHost).toHaveBeenCalledWith(remote.baseUrl, null);
    expect(store.mode).toBe("remote");
  });

  it("falls back to the local engine when the saved remote host is unreachable", async () => {
    mocked.appSettingsGet.mockResolvedValue({
      ...defaults,
      mode: "remote",
      remoteUrl: remote.baseUrl,
    });
    mocked.setRemoteHost.mockRejectedValue("Can't reach http://studio.local:7680");
    mocked.startLocalEngine.mockResolvedValue(local);
    const store = useConnectionStore();
    await store.init();
    // Launch must never dead-end on an unreachable host: fall back cleanly.
    expect(store.ready).toBe(true);
    expect(store.mode).toBe("local");
    const toasts = useToastStore();
    expect(toasts.items.map((t) => t.message).join(" ")).toContain("studio.local");
  });

  it("keeps the remote preference for the next launch after falling back", async () => {
    mocked.appSettingsGet.mockResolvedValue({
      ...defaults,
      mode: "remote",
      remoteUrl: remote.baseUrl,
    });
    mocked.setRemoteHost.mockRejectedValue("unreachable");
    mocked.startLocalEngine.mockResolvedValue(local);
    const store = useConnectionStore();
    await store.init();
    // The fallback is for this session only — settings.mode stays "remote"
    // so the preferred host is retried on the next launch.
    expect(mocked.appSettingsSet).not.toHaveBeenCalled();
  });

  it("surfaces engine-start failures as error state", async () => {
    mocked.appSettingsGet.mockResolvedValue(defaults);
    mocked.startLocalEngine.mockRejectedValue("The engine didn't start.");
    const store = useConnectionStore();
    await store.init();
    expect(store.status).toBe("error");
    expect(store.error).toContain("didn't start");
    expect(store.ready).toBe(false);
  });

  it("init is idempotent once ready", async () => {
    mocked.appSettingsGet.mockResolvedValue(defaults);
    mocked.startLocalEngine.mockResolvedValue(local);
    const store = useConnectionStore();
    await store.init();
    await store.init();
    expect(mocked.startLocalEngine).toHaveBeenCalledTimes(1);
  });
});
