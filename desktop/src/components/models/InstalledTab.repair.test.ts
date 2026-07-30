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

const { startCatalogDownload, fetchCatalogDetail } = vi.hoisted(() => ({
  startCatalogDownload: vi.fn(),
  fetchCatalogDetail: vi.fn().mockRejectedValue(new Error("no detail in tests")),
}));
vi.mock("../../lib/api/catalog", () => ({ startCatalogDownload, fetchCatalogDetail }));
vi.mock("../../lib/openExternal", () => ({ openExternal: vi.fn() }));

import InstalledTab from "./InstalledTab.vue";
import { useConnectionStore } from "../../stores/connection";
import { useHostsStore } from "../../stores/hosts";
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
  // Clicking the row opens the shared detail drawer, which lists components.
  await wrapper.get("[data-test='row-title']").trigger("click");
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  vi.clearAllMocks();
  startCatalogDownload.mockResolvedValue(undefined);
  fetchCatalogDetail.mockRejectedValue(new Error("no detail in tests"));
});

describe("InstalledTab model info drawer", () => {
  it("labels installed rows with model kind and explicit mature classification", () => {
    setActivePinia(createPinia());
    const wrapper = mount(InstalledTab, {
      props: {
        entries: [
          model({
            name: "cv:8001",
            family: "flux",
            display_name: "Portrait adapter",
            kind: "lora",
            modality: "image",
            nsfw: true,
          }),
        ],
      },
    });

    expect(wrapper.get('[data-test="model-kind-badge"]').text()).toBe("LoRA");
    expect(wrapper.get('[data-test="model-nsfw-badge"]').text()).toBe("18+ NSFW");
    expect(wrapper.get('[data-test="model-table-row"]').attributes("aria-label")).toContain(
      "LoRA, 18+ NSFW",
    );
  });

  it("opens the shared detail drawer with the component listing from the owning host", async () => {
    const wrapper = await mountWithComponents([
      component({ name: "transformer", kind: "transformer", present: true }),
      component({ name: "vae", kind: "vae", present: false }),
    ]);

    const drawer = wrapper.get("[data-test='catalog-detail-drawer']");
    expect(drawer.text()).toContain("sdxl-base:fp16");
    // Presence summary + per-component rows in the ON THIS HOST section.
    expect(drawer.get("[data-test='component-list']").text()).toContain("1/2 present");
    expect(fetchModelComponents).toHaveBeenCalledWith("sdxl-base:fp16", undefined);
  });

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
    // the same API target the component listing came from (the owning host —
    // the local primary here, so no explicit target or forwarding).
    expect(startCatalogDownload).toHaveBeenCalledWith("sdxl-base:fp16", undefined, false);
    expect(useToastStore().items.some((t) => /repair/i.test(t.message))).toBe(true);
  });

  it("surfaces repair failures as an error toast", async () => {
    startCatalogDownload.mockRejectedValue(new Error("host offline"));
    const wrapper = await mountWithComponents([component({ present: false })]);

    await wrapper.get("[data-test='component-repair']").trigger("click");
    await flushPromises();

    expect(useToastStore().items.some((t) => t.kind === "error")).toBe(true);
  });

  it("confirms the owning host before repairing a model installed on multiple hosts", async () => {
    setActivePinia(createPinia());
    const connection = useConnectionStore();
    connection.info = {
      mode: "local",
      baseUrl: "http://127.0.0.1:7680",
      apiKey: "local-key",
    };
    connection.status = "ready";
    useHostsStore().extras.push({
      id: "studio-7680",
      label: "Studio GPU",
      url: "http://studio:7680",
      apiKey: "studio-key",
      status: "ready",
      error: null,
      instanceId: null,
    });

    const wrapper = mount(InstalledTab, {
      attachTo: document.body,
      props: { entries: [{ ...model(), hostIds: ["local", "studio-7680"] }] },
    });
    await wrapper.get("[data-test='row-title']").trigger("click");
    await flushPromises();
    await wrapper.get("[data-test='drawer-repair']").trigger("click");
    await flushPromises();

    const dialog = document.body.querySelector<HTMLElement>(
      "[data-test='download-target-dialog']",
    )!;
    expect(dialog.textContent).toContain("Choose where to repair");
    expect(dialog.textContent).toContain("This device");
    expect(dialog.textContent).toContain("Studio GPU");
    expect(startCatalogDownload).not.toHaveBeenCalled();

    document.body
      .querySelector<HTMLButtonElement>("[data-test='download-target-studio-7680']")!
      .click();
    await flushPromises();

    expect(startCatalogDownload).toHaveBeenCalledWith(
      "sdxl-base:fp16",
      { baseUrl: "http://studio:7680", apiKey: "studio-key" },
      true,
    );
    wrapper.unmount();
  });
});
