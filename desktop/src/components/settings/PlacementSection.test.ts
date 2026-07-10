import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import PlacementSection from "./PlacementSection.vue";
import { useModelStore } from "../../stores/models";
import type { ModelEntry } from "../../lib/api/types";

const putModelPlacement = vi.fn().mockResolvedValue(undefined);
const deleteModelPlacement = vi.fn().mockResolvedValue(undefined);
vi.mock("../../lib/api/placement", () => ({
  putModelPlacement: (...a: unknown[]) => putModelPlacement(...a),
  deleteModelPlacement: (...a: unknown[]) => deleteModelPlacement(...a),
}));

const apiJson = vi.fn((path: string) => {
  if (path === "/api/resources") {
    return Promise.resolve({
      hostname: "test",
      timestamp: 0,
      gpus: [
        { ordinal: 0, name: "RTX 4090", backend: "cuda", vram_total: 0, vram_used: 0 },
        { ordinal: 1, name: "A100", backend: "cuda", vram_total: 0, vram_used: 0 },
      ],
      system_ram: { total: 0, used: 0, used_by_mold: 0, used_by_other: 0 },
    });
  }
  return Promise.resolve([]);
});
vi.mock("../../lib/api/client", () => ({
  apiJson: (...a: unknown[]) => apiJson(...(a as [string])),
}));

function model(name: string, family: string): ModelEntry {
  return {
    name,
    family,
    size_gb: 1,
    is_loaded: false,
    hf_repo: "x/y",
    default_steps: 20,
    default_guidance: 3.5,
    default_width: 1024,
    default_height: 1024,
    description: "",
    downloaded: true,
  };
}

async function mountWithModels(models: ModelEntry[]) {
  setActivePinia(createPinia());
  const store = useModelStore();
  store.all = models;
  const wrapper = mount(PlacementSection);
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  putModelPlacement.mockClear();
  deleteModelPlacement.mockClear();
  apiJson.mockClear();
});

describe("PlacementSection", () => {
  it("renders Auto/CPU plus one option per GPU in the text-encoders select", async () => {
    const wrapper = await mountWithModels([model("flux-dev:q4", "flux")]);
    const options = wrapper
      .find('[data-test="placement-text-encoders"]')
      .findAll("option")
      .map((o) => o.attributes("value"));
    expect(options).toEqual(["auto", "cpu", "gpu:0", "gpu:1"]);
  });

  it("shows the tier-2 advanced disclosure only for tier-2 families", async () => {
    const tier2 = await mountWithModels([model("flux-dev:q4", "flux")]);
    expect(tier2.find('[data-test="placement-advanced-toggle"]').exists()).toBe(true);

    const nonTier2 = await mountWithModels([model("sd15:fp16", "sd1.5")]);
    expect(nonTier2.find('[data-test="placement-advanced-toggle"]').exists()).toBe(false);
  });

  it("reveals per-component selects when the disclosure is expanded", async () => {
    const wrapper = await mountWithModels([model("flux-dev:q4", "flux")]);
    expect(wrapper.find('[data-test="placement-advanced"]').exists()).toBe(false);
    await wrapper.find('[data-test="placement-advanced-toggle"]').trigger("click");
    expect(wrapper.find('[data-test="placement-advanced"]').exists()).toBe(true);
    // transformer, vae, and the four nullable encoder rows.
    expect(wrapper.find("#placement-transformer").exists()).toBe(true);
    expect(wrapper.find("#placement-vae").exists()).toBe(true);
    expect(wrapper.find("#placement-t5").exists()).toBe(true);
  });

  it("Save as default calls the api with the selected model and placement", async () => {
    const wrapper = await mountWithModels([model("flux-dev:q4", "flux")]);
    await wrapper.find('[data-test="placement-text-encoders"]').setValue("cpu");
    await wrapper.find('[data-test="placement-save"]').trigger("click");
    await flushPromises();
    expect(putModelPlacement).toHaveBeenCalledTimes(1);
    const [name, placement] = putModelPlacement.mock.calls[0]!;
    expect(name).toBe("flux-dev:q4");
    expect(placement).toEqual({ text_encoders: { kind: "cpu" }, advanced: null });
  });

  it("Clear calls deleteModelPlacement for the selected model", async () => {
    const wrapper = await mountWithModels([model("flux-dev:q4", "flux")]);
    await wrapper.find('[data-test="placement-clear"]').trigger("click");
    await flushPromises();
    expect(deleteModelPlacement).toHaveBeenCalledWith("flux-dev:q4");
  });

  it("shows an empty-state message when no models are installed", async () => {
    const wrapper = await mountWithModels([]);
    expect(wrapper.text()).toContain("No installed models");
    expect(wrapper.find('[data-test="placement-model"]').exists()).toBe(false);
  });
});
