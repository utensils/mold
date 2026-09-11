import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../../lib/ipc", () => ({
  inTauri: () => false,
  ipc: { pickDirectory: vi.fn(() => Promise.resolve(null)) },
}));
vi.mock("../../lib/api/config", () => ({
  fetchConfig: vi.fn(() => Promise.resolve([])),
  fetchProfiles: vi.fn(() => Promise.resolve({ profiles: [], active: "default" })),
  setConfig: vi.fn(() => Promise.resolve()),
  resetConfig: vi.fn(() => Promise.resolve()),
  setProfile: vi.fn(() => Promise.resolve()),
}));

import CloudSection from "./CloudSection.vue";
import { useSettingsConfigStore } from "../../stores/settingsConfig";
import type { ConfigRow } from "../../lib/api/types";

function row(key: string, value: ConfigRow["value"]): ConfigRow {
  return { key, value, source: "db", env_var: null, restart_required: false };
}

beforeEach(() => {
  setActivePinia(createPinia());
});

/*
 * Cloud GPUs is where the seventeen `runpod.*` / `lambda.*` keys went. They
 * used to be most of the Advanced dump, labelled by raw key under the sentence
 * "Server-provided configuration key." — which is now reserved for keys newer
 * than this client.
 */
describe("Settings ▸ Cloud GPUs", () => {
  it("renders every cloud key the engine reports, under its plain label", () => {
    const config = useSettingsConfigStore();
    config.rows = [
      row("runpod.api_key", "<set>"),
      row("runpod.default_gpu", "A40"),
      row("lambda.endpoint", "https://cloud.lambda.ai/api/v1"),
    ];
    const wrapper = mount(CloudSection);
    expect(wrapper.findAll("[data-test='cloud-row']")).toHaveLength(3);
    expect(wrapper.text()).not.toContain("Server-provided configuration key.");
    expect(wrapper.text()).not.toContain("runpod.default_gpu");
  });

  it("hides a key this engine does not report rather than drawing an empty row", () => {
    const config = useSettingsConfigStore();
    config.rows = [row("runpod.api_key", "<set>")];
    expect(mount(CloudSection).findAll("[data-test='cloud-row']")).toHaveLength(1);
  });

  it("says so when the engine reports no cloud keys at all", () => {
    useSettingsConfigStore().rows = [row("models_dir", "/models")];
    const wrapper = mount(CloudSection);
    expect(wrapper.findAll("[data-test='cloud-row']")).toHaveLength(0);
    expect(wrapper.find("[data-test='cloud-empty']").exists()).toBe(true);
  });

  it("reads a stored API key as present and never renders the sentinel", () => {
    // The engine answers a stored key with the literal `<set>`, which is not a
    // value anyone may see, edit, or send back.
    useSettingsConfigStore().rows = [row("runpod.api_key", "<set>")];
    const wrapper = mount(CloudSection);
    expect(wrapper.text()).not.toContain("<set>");
    expect(wrapper.text()).toContain("set");
    expect(wrapper.find("[data-test='secret-edit']").exists()).toBe(true);
  });
});
