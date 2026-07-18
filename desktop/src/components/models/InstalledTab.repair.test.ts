import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import type { ModelComponentStatus, ModelEntry } from "../../lib/api/types";

const { fetchModelComponents, loadModel, removeModel, unloadModel } = vi.hoisted(() => ({
  fetchModelComponents: vi.fn(),
  loadModel: vi.fn(),
  removeModel: vi.fn(),
  unloadModel: vi.fn(),
}));
vi.mock("../../lib/api/models", () => ({
  fetchModelComponents,
  loadModel,
  removeModel,
  unloadModel,
}));

const { startCatalogDownload } = vi.hoisted(() => ({
  startCatalogDownload: vi.fn(),
}));
vi.mock("../../lib/api/catalog", () => ({ startCatalogDownload }));
vi.mock("../../lib/openExternal", () => ({ openExternal: vi.fn() }));

import InstalledTab from "./InstalledTab.vue";
import { useModelStore } from "../../stores/models";
import { useToastStore } from "../../stores/toasts";

function model(part: Partial<ModelEntry> = {}): ModelEntry {
  return {
    name: "sdxl-base:fp16",
    family: "sdxl",
    size_gb: 6.9,
    is_loaded: false,
    hf_repo: "stabilityai/stable-diffusion-xl-base-1.0",
    default_steps: 30,
    default_guidance: 7,
    default_width: 1024,
    default_height: 1024,
    description: "",
    downloaded: true,
    disk_usage_bytes: 6_900_000_000,
    ...part,
  };
}

function component(part: Partial<ModelComponentStatus> = {}): ModelComponentStatus {
  return {
    kind: "vae",
    name: "vae",
    present: true,
    repair_model: "sdxl-base:fp16",
    ...part,
  };
}

async function mountWithComponents(components: ModelComponentStatus[]) {
  setActivePinia(createPinia());
  useModelStore().all = [model()];
  fetchModelComponents.mockResolvedValue({ model: "sdxl-base:fp16", components });
  const wrapper = mount(InstalledTab, { props: {} });
  await flushPromises();
  // Expand the Info section to reveal the per-component rows.
  const info = wrapper.findAll("button").find((b) => b.text() === "Info");
  await info?.trigger("click");
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  vi.clearAllMocks();
  startCatalogDownload.mockResolvedValue(undefined);
});

describe("InstalledTab component repair", () => {
  it("offers Repair only on missing components that carry a repair_model", async () => {
    const wrapper = await mountWithComponents([
      component({ name: "transformer", kind: "transformer", present: true }),
      component({ name: "vae", kind: "vae", present: false }),
      component({ name: "orphan", kind: "clip", present: false, repair_model: null }),
    ]);

    const repairs = wrapper.findAll("[data-test='component-repair']");
    expect(repairs).toHaveLength(1);
  });

  it("repairs a missing component by re-running the catalog download on the owning host", async () => {
    const wrapper = await mountWithComponents([component({ name: "vae", present: false })]);

    await wrapper.get("[data-test='component-repair']").trigger("click");
    await flushPromises();

    // The download is keyed on the server-provided repair_model and goes to
    // the same API target the component listing came from (the owning host).
    expect(startCatalogDownload).toHaveBeenCalledWith("sdxl-base:fp16");
    expect(useToastStore().items.some((t) => /repair/i.test(t.message))).toBe(true);
  });

  it("surfaces repair failures as an error toast", async () => {
    startCatalogDownload.mockRejectedValue(new Error("host offline"));
    const wrapper = await mountWithComponents([component({ present: false })]);

    await wrapper.get("[data-test='component-repair']").trigger("click");
    await flushPromises();

    expect(useToastStore().items.some((t) => t.kind === "error")).toBe(true);
  });
});
