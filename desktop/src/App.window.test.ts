import { describe, expect, it } from "vitest";
import tauriConfig from "../src-tauri/tauri.conf.json";
import linuxConfig from "../src-tauri/tauri.linux.conf.json";
import macosConfig from "../src-tauri/tauri.macos.conf.json";
import source from "./App.vue?raw";

describe("main window launch", () => {
  it("keeps the macOS linkage hook out of Linux bundles", () => {
    expect(tauriConfig.build).not.toHaveProperty("beforeBundleCommand");
    expect(linuxConfig).not.toHaveProperty("build");
    expect(macosConfig.build?.beforeBundleCommand).toContain("macos");
  });

  it("opens maximized without entering full screen", () => {
    const mainWindow = tauriConfig.app.windows[0];

    expect(mainWindow?.maximized).toBe(true);
    expect(mainWindow?.fullscreen).toBe(false);
    expect(source).toMatch(/await appWindow\.maximize\(\);\s*await appWindow\.show\(\);/);
  });

  it("keeps native Linux decorations and scopes overlay chrome to macOS", () => {
    expect(linuxConfig.app.windows[0]?.decorations).toBe(true);
    expect(linuxConfig.app.windows[0]?.visible).toBe(true);
    expect(macosConfig.app.windows[0]?.visible).toBe(false);
    expect(macosConfig.app.windows[0]?.titleBarStyle).toBe("Overlay");
    expect(tauriConfig.app.windows[0]).not.toHaveProperty("trafficLightPosition");
  });
});
