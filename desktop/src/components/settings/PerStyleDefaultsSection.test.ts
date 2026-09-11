import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: { pickDirectory: vi.fn(() => Promise.resolve(null)) },
}));
const setConfig = vi.fn(() => Promise.resolve());
vi.mock("../../lib/api/config", () => ({
  fetchConfig: vi.fn(() => Promise.resolve([])),
  fetchProfiles: vi.fn(() => Promise.resolve({ profiles: [], active: "default" })),
  setConfig: (...args: unknown[]) => setConfig(...(args as [])),
  resetConfig: vi.fn(() => Promise.resolve()),
  setProfile: vi.fn(() => Promise.resolve()),
}));

import PerStyleDefaultsSection from "./PerStyleDefaultsSection.vue";
import { useModelStore } from "../../stores/models";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import type { ConfigRow, ModelEntry } from "../../lib/api/types";

function row(key: string, value: ConfigRow["value"]): ConfigRow {
  return { key, value, source: "db", env_var: null, restart_required: false };
}

/** One style's worth of rows, the way `/api/config` reports them. */
function styleRows(style: string): ConfigRow[] {
  return [
    row(`models.${style}.default_steps`, 28),
    row(`models.${style}.default_guidance`, 3.5),
    row(`models.${style}.negative_prompt`, null),
  ];
}

beforeEach(() => {
  setActivePinia(createPinia());
  setConfig.mockClear();
});

/*
 * hal9000 reports 104 `models.<style>.<field>` rows over 13 styles. Flat, that
 * is most of the Settings page and says nothing about any one style; here it is
 * 13 collapsed disclosures that open to the fields.
 */
describe("Settings ▸ Per-style defaults", () => {
  it("is one collapsed row per style, not one per field", () => {
    const config = useSettingsConfigStore();
    config.rows = [...styleRows("sd1.5"), ...styleRows("flux-dev:q4")];
    const wrapper = mount(PerStyleDefaultsSection);
    expect(wrapper.findAll("details")).toHaveLength(2);
    expect(wrapper.findAll("[data-test='per-style-row-default_steps']")).toHaveLength(0);
  });

  it("leads with the style's friendly name when this machine knows it", async () => {
    const config = useSettingsConfigStore();
    config.rows = styleRows("sd1.5");
    useModelStore().all = [
      {
        name: "sd1.5",
        family: "sd",
        description: "Stable Diffusion 1.5",
        downloaded: true,
      } as ModelEntry,
    ];
    const wrapper = mount(PerStyleDefaultsSection);
    await flushPromises();
    expect(wrapper.get("[data-test='per-style-name']").text()).toBe("Stable Diffusion 1.5");
    // The runnable id stays on the row, because it is what `mold config set`
    // takes and what `mold config list` prints.
    expect(wrapper.get("[data-test='per-style-id']").text()).toBe("sd1.5");
  });

  it("falls back to the id for a style this machine has never listed", () => {
    useSettingsConfigStore().rows = styleRows("some-other-style");
    const wrapper = mount(PerStyleDefaultsSection);
    expect(wrapper.get("[data-test='per-style-name']").text()).toBe("some-other-style");
  });

  it("writes the full engine key when a field is edited", async () => {
    const config = useSettingsConfigStore();
    config.rows = styleRows("sd1.5");
    const wrapper = mount(PerStyleDefaultsSection);
    const details = wrapper.get("details");
    (details.element as HTMLDetailsElement).open = true;
    await details.trigger("toggle");
    const field = wrapper.get("[data-test='per-style-row-default_steps'] input");
    await field.setValue("32");
    await field.trigger("blur");
    await flushPromises();
    expect(setConfig).toHaveBeenCalledWith("models.sd1.5.default_steps", 32);
  });

  it("offers a filter only once there are more styles than fit at a glance", () => {
    const config = useSettingsConfigStore();
    config.rows = ["a", "b", "c"].flatMap(styleRows);
    expect(mount(PerStyleDefaultsSection).find("[data-test='per-style-filter']").exists()).toBe(
      false,
    );

    setActivePinia(createPinia());
    const many = useSettingsConfigStore();
    many.rows = ["a", "b", "c", "d", "e", "f", "g", "h", "i"].flatMap(styleRows);
    const wrapper = mount(PerStyleDefaultsSection);
    expect(wrapper.find("[data-test='per-style-filter']").exists()).toBe(true);
    expect(wrapper.findAll("details")).toHaveLength(9);
  });

  it("narrows to the styles the filter names", async () => {
    const config = useSettingsConfigStore();
    config.rows = [
      "alpha",
      "beta",
      "gamma",
      "delta",
      "epsilon",
      "zeta",
      "eta",
      "theta",
      "iota",
    ].flatMap(styleRows);
    const wrapper = mount(PerStyleDefaultsSection);
    await wrapper.get("[data-test='per-style-filter']").setValue("eta");
    // beta, theta, zeta and eta all carry it.
    expect(wrapper.findAll("details").length).toBeLessThan(9);
    expect(wrapper.text()).toContain("eta");
    expect(wrapper.text()).not.toContain("alpha");
  });

  it("says so when no style has been tuned", () => {
    useSettingsConfigStore().rows = [row("models_dir", "/models")];
    const wrapper = mount(PerStyleDefaultsSection);
    expect(wrapper.findAll("details")).toHaveLength(0);
    expect(wrapper.find("[data-test='per-style-empty']").exists()).toBe(true);
  });
});
