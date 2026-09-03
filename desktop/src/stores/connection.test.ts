import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useConnectionStore } from "./connection";
import { ipc } from "../lib/ipc";

vi.mock("../lib/ipc", () => ({
  inTauri: () => false,
  ipc: {
    appSettingsGet: vi.fn(),
    appSettingsSet: vi.fn().mockResolvedValue(undefined),
    ensureLocalServer: vi.fn(),
    startLocalEngine: vi.fn(),
    stopLocalEngine: vi.fn(),
    testRemoteHost: vi.fn(),
    getConnection: vi.fn(),
    secretGet: vi.fn().mockResolvedValue(null),
  },
}));

const mocked = vi.mocked(ipc);

const local = { mode: "local" as const, baseUrl: "http://127.0.0.1:49152", apiKey: "k" };
const remote = { mode: "remote" as const, baseUrl: "http://studio.local:7680", apiKey: null };
const localServer = {
  kind: "embedded" as const,
  baseUrl: "http://127.0.0.1:7680",
  apiKey: "local-key",
  port: 7680,
};
const defaults = {
  mode: "local" as const,
  remoteUrl: null,
  remoteApiKey: null,
  lastRoute: null,
  engineEnv: {},
  theme: "mocha" as const,
  matchSystem: false,
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
  saveRemoteOutputs: true,
  navRailWidth: null,
  generateParamsWidth: null,
  sidebarCollapsed: false,
};

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  mocked.ensureLocalServer.mockResolvedValue(localServer);
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

  it("brings the built-in engine online regardless of a legacy remote setting", async () => {
    // Remote-primary is retired; the Rust boot migration has already flipped
    // mode to local, but even a stale mode:remote must just boot local.
    mocked.appSettingsGet.mockResolvedValue({
      ...defaults,
      mode: "remote",
      remoteUrl: remote.baseUrl,
    });
    mocked.startLocalEngine.mockResolvedValue(local);
    const store = useConnectionStore();
    await store.init();
    expect(store.ready).toBe(true);
    expect(store.mode).toBe("local");
  });

  it("errors cleanly when the local server can't start", async () => {
    mocked.appSettingsGet.mockResolvedValue(defaults);
    mocked.ensureLocalServer.mockRejectedValue("No local port available");
    const store = useConnectionStore();
    await store.init();
    expect(store.status).toBe("error");
    expect(store.localStatus).toBe("error");
    expect(store.localError).toContain("No local port available");
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

  it("coalesces overlapping restart requests into one stop/start cycle", async () => {
    let finishStop!: (value: typeof local) => void;
    mocked.stopLocalEngine.mockImplementation(
      () => new Promise((resolve) => (finishStop = resolve)),
    );
    mocked.startLocalEngine.mockResolvedValue(local);
    mocked.appSettingsGet.mockResolvedValue(defaults);
    mocked.appSettingsSet.mockResolvedValue(undefined);

    const store = useConnectionStore();
    store.info = local;
    store.status = "ready";
    store.localInfo = localServer;
    store.localStatus = "ready";

    const first = store.restartEngine();
    const repeated = store.restartEngine();

    expect(store.status).toBe("starting");
    expect(mocked.stopLocalEngine).toHaveBeenCalledTimes(1);

    finishStop(local);
    await expect(first).resolves.toBe("restarted");
    await expect(repeated).resolves.toBe("coalesced");
    expect(mocked.ensureLocalServer).toHaveBeenCalledTimes(1);
    expect(mocked.startLocalEngine).toHaveBeenCalledTimes(1);
    expect(store.ready).toBe(true);
  });
});
