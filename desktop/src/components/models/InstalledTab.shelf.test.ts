import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import type { ModelEntry, ServerStatus } from "../../lib/api/types";

const { loadModel, removeModel, unloadModel } = vi.hoisted(() => ({
  loadModel: vi.fn(),
  removeModel: vi.fn(),
  unloadModel: vi.fn(),
}));
vi.mock("../../lib/api/models", () => ({
  fetchModelComponents: vi.fn().mockRejectedValue(new Error("no components in tests")),
  loadModel,
  removeModel,
  unloadModel,
}));
vi.mock("../../lib/api/catalog", () => ({
  startCatalogDownload: vi.fn(),
  fetchCatalogDetail: vi.fn().mockRejectedValue(new Error("no detail in tests")),
}));
const openExternal = vi.fn();
vi.mock("../../lib/openExternal", () => ({
  openExternal: (...a: unknown[]) => openExternal(...a),
}));

import InstalledTab from "./InstalledTab.vue";
import { formatGB } from "../../lib/format";
import { isSeparator, useContextMenuStore } from "../../stores/contextMenu";
import { useHostStatusStore } from "../../stores/hostStatus";
import { useModelStore } from "../../stores/models";

function model(part: Partial<ModelEntry> = {}): ModelEntry {
  return {
    name: "flux-dev:q4",
    family: "flux",
    size_gb: 6.8,
    is_loaded: false,
    hf_repo: "black-forest-labs/FLUX.1-dev",
    default_steps: 20,
    default_guidance: 3.5,
    default_width: 1024,
    default_height: 1024,
    description: "FLUX.1 Dev Q4 — smaller/faster, good quality",
    downloaded: true,
    disk_usage_bytes: 6_800_000_000,
    ...part,
  };
}

function status(part: Partial<ServerStatus> = {}): ServerStatus {
  return { version: "test", models_loaded: [], uptime_secs: 0, ...part };
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  document.body.innerHTML = "";
});

describe("InstalledTab shelf", () => {
  it("stacks the request id and the plain note under a row's name", async () => {
    useModelStore().all = [model({ display_name: "Photoreal — best quality" })];
    const wrapper = mount(InstalledTab);
    await flushPromises();
    expect(wrapper.get("[data-test='row-title']").text()).toBe("Photoreal — best quality");
    expect(wrapper.get("[data-test='row-id']").text()).toBe("flux-dev:q4");
    expect(wrapper.get("[data-test='row-note']").text()).toBe(
      "FLUX.1 Dev Q4 — smaller/faster, good quality",
    );
  });

  it("meters the disk used by styles once this machine reports its models disk", async () => {
    useModelStore().all = [
      model(),
      model({ name: "sdxl-base:fp16", family: "sdxl", size_gb: 6.9 }),
    ];
    const wrapper = mount(InstalledTab);
    await flushPromises();
    expect(wrapper.find("[data-test='styles-disk-meter']").exists()).toBe(false);

    useHostStatusStore().status = status({
      models_disk: { total_bytes: 512_000_000_000, free_bytes: 400_000_000_000 },
    });
    await flushPromises();
    const meter = wrapper.get("[data-test='styles-disk-meter']");
    // Each style's own weights, one segment per family, against the whole disk.
    expect(meter.text()).toContain(`${formatGB(13_700_000_000)} of ${formatGB(512_000_000_000)}`);
    expect(meter.get("[role='meter']").attributes("aria-valuenow")).toBe("3");
    expect(meter.findAll("[role='meter'] > span")).toHaveLength(2);
  });

  it("puts the page link and Remove behind the row's ⋯, and removes only after the confirm", async () => {
    useModelStore().all = [model()];
    removeModel.mockResolvedValue({ freed_bytes: 6_800_000_000, kept: [] });
    const wrapper = mount(InstalledTab);
    await flushPromises();

    await wrapper.get("[data-test='row-menu']").trigger("click");
    const menu = useContextMenuStore();
    const labels = menu.entries.flatMap((e) => (isSeparator(e) ? [] : [e.label]));
    expect(labels).toEqual(["Open the style's page", "Remove from disk…"]);

    const remove = menu.entries.find((e) => !isSeparator(e) && e.label === "Remove from disk…");
    if (!remove || isSeparator(remove)) throw new Error("no remove entry");
    expect(remove.danger).toBe(true);
    remove.action?.();
    await flushPromises();
    expect(removeModel).not.toHaveBeenCalled();

    const dialog = wrapper.get("[data-test='confirm-dialog']");
    expect(dialog.text()).toContain("Remove flux-dev:q4 from disk?");
    expect(dialog.text()).toContain("Frees 6.8 GB weights");
    await wrapper.get("[data-test='confirm-accept']").trigger("click");
    await flushPromises();
    expect(removeModel).toHaveBeenCalledWith("flux-dev:q4", undefined);
    expect(wrapper.find("[data-test='confirm-dialog']").exists()).toBe(false);
    wrapper.unmount();
  });

  it("speaks the lexicon on the empty shelf and routes to Browse more", async () => {
    useModelStore().all = [];
    const wrapper = mount(InstalledTab);
    await flushPromises();
    expect(wrapper.text()).toContain("No styles ready yet.");
    const cta = wrapper.findAll("button").find((b) => b.text() === "Browse more styles");
    await cta!.trigger("click");
    expect(wrapper.emitted("browse-catalog")).toHaveLength(1);
  });
});
