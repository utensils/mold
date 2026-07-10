import { describe, expect, it } from "vitest";
import tauriConfig from "../src-tauri/tauri.conf.json";
import source from "./App.vue?raw";

describe("main window launch", () => {
  it("opens maximized without entering full screen", () => {
    const mainWindow = tauriConfig.app.windows[0];

    expect(mainWindow?.maximized).toBe(true);
    expect(mainWindow?.fullscreen).toBe(false);
    expect(source).toMatch(/await appWindow\.maximize\(\);\s*await appWindow\.show\(\);/);
  });
});
