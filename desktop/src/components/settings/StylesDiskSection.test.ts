/**
 * Settings ▸ Styles & disk. The meter reads the primary's own `models_disk`
 * and its reading shares one unit with the machine card and the status bar,
 * so the same bytes cannot read two ways in two places.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

const pickDirectory = vi.fn((_title: string) => Promise.resolve("/Volumes/Big/models"));
vi.mock("../../lib/ipc", () => ({
  inTauri: () => true,
  ipc: { pickDirectory: (title: string) => pickDirectory(title) },
}));
const setConfig = vi.fn(() => Promise.resolve());
vi.mock("../../lib/api/config", () => ({
  fetchConfig: vi.fn(() => Promise.resolve([])),
  fetchProfiles: vi.fn(() => Promise.resolve({ profiles: [], active: "default" })),
  setConfig: (...args: unknown[]) => setConfig(...(args as [])),
  resetConfig: vi.fn(() => Promise.resolve()),
  setProfile: vi.fn(() => Promise.resolve()),
}));

import StylesDiskSection from "./StylesDiskSection.vue";
import { useHostStatusStore } from "../../stores/hostStatus";
import { useSettingsConfigStore } from "../../stores/settingsConfig";

beforeEach(() => {
  setActivePinia(createPinia());
  pickDirectory.mockClear();
  setConfig.mockClear();
});

describe("StylesDiskSection", () => {
  it("hides the meter until the primary reports a disk", () => {
    const wrapper = mount(StylesDiskSection);
    expect(wrapper.find("[data-test='settings-disk-meter']").exists()).toBe(false);
  });

  it("states the reading as one used/total pair, decimal, with the unit once", () => {
    const hostStatus = useHostStatusStore();
    hostStatus.status = {
      models_disk: { total_bytes: 36_000_000_000, free_bytes: 23_900_000_000 },
    } as never;

    const meter = mount(StylesDiskSection).get("[data-test='settings-disk-meter']");
    expect(meter.text()).toContain("Disk for styles");
    expect(meter.text()).toContain("12.1 / 36.0 GB");
    expect(meter.get("[role='meter']").attributes("aria-valuenow")).toBe("34");
  });

  /*
   * The shared kit's path control takes its picker by INJECTION — studio may
   * not reference Tauri at all — so a browser gets an editable field and only
   * a caller that hands one down gets Choose…. This is the app doing that.
   */
  it("still opens the native folder picker and saves what it returns", async () => {
    const config = useSettingsConfigStore();
    config.available = true;
    config.rows = [
      {
        key: "models_dir",
        value: "/old/models",
        source: "file",
        env_var: null,
        restart_required: false,
      },
    ];
    const wrapper = mount(StylesDiskSection);
    const choose = wrapper.findAll("button").find((b) => b.text() === "Choose…");
    expect(choose, "the native folder picker button").toBeTruthy();

    await choose!.trigger("click");
    await flushPromises();
    expect(pickDirectory).toHaveBeenCalledWith("Where styles are kept");
    expect(setConfig).toHaveBeenCalledWith("models_dir", "/Volumes/Big/models");
  });
});
