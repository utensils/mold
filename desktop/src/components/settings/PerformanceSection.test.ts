import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: { appSettingsGet: vi.fn(), appSettingsSet: vi.fn() },
}));

import PerformanceSection from "./PerformanceSection.vue";
import { useConnectionStore } from "../../stores/connection";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import type { ConfigRow } from "../../lib/api/types";

const SCHEDULER_KEYS = [
  "scheduler.replan_debounce_ms",
  "scheduler.replan_max_delay_ms",
  "scheduler.warm_wait_max_ms",
];

function configRow(key: string, value: number): ConfigRow {
  return {
    key,
    value,
    source: "db",
    env_var: null,
    restart_required: false,
  } as ConfigRow;
}

function mountSection(mode: "local" | "external" | "remote") {
  const conn = useConnectionStore();
  conn.info = { mode, baseUrl: "http://127.0.0.1:7680", apiKey: null };
  conn.status = "ready";
  const config = useSettingsConfigStore();
  config.rows = SCHEDULER_KEYS.map((key) => configRow(key, 250));
  return mount(PerformanceSection);
}

beforeEach(() => {
  setActivePinia(createPinia());
});

describe("Speed & memory", () => {
  it("renders the scheduler keys it declares, on every kind of engine", async () => {
    // The schema puts these three in this section; without a render site,
    // searching "replan" narrowed the nav to Speed & memory and showed
    // nothing at all.
    for (const mode of ["local", "external", "remote"] as const) {
      setActivePinia(createPinia());
      const wrapper = mountSection(mode);
      await flushPromises();
      expect(wrapper.findAll("[data-test='performance-engine-row']")).toHaveLength(3);
      expect(wrapper.text()).toContain("Queue replan debounce");
      expect(wrapper.text()).toContain("Maximum replan delay");
      expect(wrapper.text()).toContain("Maximum warm-model wait");
    }
  });

  it("calls a reused local mold serve what it is, not a shared or remote server", async () => {
    const wrapper = mountSection("external");
    await flushPromises();
    const note = wrapper.get("[data-test='performance-note']").text();
    expect(note).toContain("already running on this device");
    expect(note).not.toContain("shared or remote server");
  });

  it("says a remote machine manages its own environment", async () => {
    const wrapper = mountSection("remote");
    await flushPromises();
    expect(wrapper.get("[data-test='performance-note']").text()).toContain("another machine");
  });

  it("keeps the environment knobs for the engine the app itself starts", async () => {
    const wrapper = mountSection("local");
    await flushPromises();
    expect(wrapper.find("[data-test='performance-note']").exists()).toBe(false);
    expect(wrapper.text()).toContain("MOLD_STEP_PREVIEW");
  });
});
