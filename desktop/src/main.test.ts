import { beforeEach, describe, expect, it, vi } from "vitest";

const setup = vi.hoisted(() => {
  const app = { use: vi.fn(), mount: vi.fn() };
  app.use.mockReturnValue(app);
  return { app, install: vi.fn<() => Promise<void>>() };
});
vi.mock("vue", () => ({ createApp: () => setup.app }));
vi.mock("pinia", () => ({ createPinia: () => ({}) }));
vi.mock("./App.vue", () => ({ default: {} }));
vi.mock("./mobile/MobileApp.vue", () => ({ default: {} }));
vi.mock("./router", () => ({ router: {} }));
vi.mock("./lib/secureNative", () => ({ installSecureNative: setup.install }));
vi.mock("@studio/api/secureMedia", () => ({ configureSecureMediaWorker: vi.fn() }));
vi.mock("@studio/lib/retireSequenceStorage", () => ({ retireSequenceStorage: vi.fn() }));
vi.mock("./lib/theme", () => ({ installSystemThemeSync: vi.fn() }));
vi.mock("./mobile/settings", () => ({
  applyMobileSettings: vi.fn(),
  loadMobileSettings: () => ({}),
}));
vi.mock("./mobile/platform", () => ({
  isNativeAndroidRuntime: () => false,
  isNativeIOSRuntime: () => false,
}));

const entries = [
  { name: "desktop", load: () => import("./main") },
  { name: "mobile", load: () => import("./mobile/main") },
];

describe.each(entries)("$name startup", ({ load }) => {
  beforeEach(() => {
    vi.resetModules();
    vi.clearAllMocks();
    setup.app.use.mockReturnValue(setup.app);
  });

  it("mounts after installing hooks while secure credential warm-up remains blocked", async () => {
    setup.install.mockReturnValue(new Promise(() => {}));
    await load();
    expect(setup.install).toHaveBeenCalledOnce();
    expect(setup.app.mount).toHaveBeenCalledWith("#app");
    expect(setup.install.mock.invocationCallOrder[0]).toBeLessThan(
      setup.app.mount.mock.invocationCallOrder[0]!,
    );
  });

  it("keeps the mounted interface available when credential warm-up fails", async () => {
    let reject!: (error: Error) => void;
    setup.install.mockReturnValue(
      new Promise((_, fail) => {
        reject = fail;
      }),
    );
    await load();
    expect(setup.app.mount).toHaveBeenCalledOnce();
    reject(new Error("Secure storage is unavailable"));
    await Promise.resolve();
    expect(setup.app.mount).toHaveBeenCalledOnce();
  });
});
