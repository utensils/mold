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
import { useGalleryStore } from "../../stores/gallery";

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

  it("heads the table once, on the grid every row shares", async () => {
    useModelStore().all = [model(), model({ name: "sdxl-base:fp16", family: "sdxl" })];
    const wrapper = mount(InstalledTab);
    await flushPromises();

    const headers = wrapper.findAll("[data-test='styles-columns']");
    expect(headers).toHaveLength(1);
    // SPEED has no field on ModelEntry; it is read from the prints already
    // made with the style (`typicalGenerationTimes`).
    expect(headers[0]!.findAll("span").map((c) => c.text())).toEqual([
      "Name",
      "Good for",
      "Size",
      "Speed",
      "Machine",
      "",
    ]);
    // One axis for the whole table: the header and every row read the same
    // custom property off the shared container.
    expect(wrapper.find(".model-table").exists()).toBe(true);
    expect(headers[0]!.classes()).toContain("model-table__header");
    expect(wrapper.findAll(".model-table-row--has-machines")).toHaveLength(2);
  });

  /**
   * The header pins five tracks. A row whose style has no description used to
   * emit four cells into them, so every column after NAME slid one track left
   * — size under "Good for", the machine dot under "Size", and the action
   * cluster crushed into a 128px track it overflowed.
   */
  it("emits one cell per pinned track whether or not a style has a description", async () => {
    useModelStore().all = [
      model(),
      model({ name: "sdxl-base:fp16", family: "sdxl", description: "" }),
    ];
    const wrapper = mount(InstalledTab);
    await flushPromises();

    const rows = wrapper.findAll("[data-test='model-table-row']");
    expect(rows).toHaveLength(2);
    const header = wrapper.get("[data-test='styles-columns']");
    for (const row of rows) {
      expect(row.find("[data-test='row-note']").exists()).toBe(true);
      expect(row.element.children.length).toBe(header.element.children.length);
    }
  });

  /**
   * The mock's SPEED column: `~20s` from the prints already made with the
   * style, the median of the newest few timed ones, and an empty cell — never
   * a guess — for a style nobody has timed. The cell is emitted either way so
   * the pinned axis holds.
   */
  it("reads each style's typical time off its recent prints", async () => {
    useModelStore().all = [model(), model({ name: "sdxl-base:fp16", family: "sdxl" })];
    const gallery = useGalleryStore();
    const print = (model: string, ms: number | undefined, n: number) =>
      ({
        filename: `p-${n}.png`,
        timestamp: 1_000 - n,
        metadata: {
          prompt: "x",
          model,
          seed: n,
          ...(ms == null ? {} : { generation_time_ms: ms }),
        },
      }) as unknown as import("../../lib/api/types").GalleryImage;
    gallery.buckets.local = {
      items: [
        print("flux-dev:q4", 4_000, 1),
        print("flux-dev:q4", 40_000, 2),
        print("flux-dev:q4", 5_000, 3),
        print("sdxl-base:fp16", undefined, 4),
      ],
      loading: false,
      error: null,
      loaded: true,
    };
    const wrapper = mount(InstalledTab);
    await flushPromises();

    const speeds = wrapper.findAll("[data-test='row-speed']").map((c) => c.text());
    expect(speeds).toEqual(["~5s", ""]);
    const header = wrapper.get("[data-test='styles-columns']");
    for (const row of wrapper.findAll("[data-test='model-table-row']")) {
      expect(row.element.children.length).toBe(header.element.children.length);
    }
  });

  it("names a family group once, in words, never a second wire slug per row", async () => {
    useModelStore().all = [model({ name: "wan22-t2v-a14b:q5", family: "wan" })];
    const wrapper = mount(InstalledTab);
    await flushPromises();
    // The heading is the family's name, not the raw slug CSS used to uppercase.
    expect(wrapper.text()).toContain("Wan Video");
    // And the row no longer repeats the heading it already sits beneath.
    expect(wrapper.find("[data-test='row-family']").exists()).toBe(false);
  });

  it("keeps the shelf row to one size line, and one machine line", async () => {
    useModelStore().all = [model()];
    const wrapper = mount(InstalledTab);
    await flushPromises();
    const sizes = wrapper.get("[data-test='row-sizes']");
    // The runtime footprint (weights + shared encoders) belongs to the drawer;
    // stacking it here is one of the three things that doubled the row height.
    expect(sizes.findAll("span")).toHaveLength(1);
  });

  it("rounds the meter's segment widths instead of writing raw floats", async () => {
    useModelStore().all = [model({ size_gb: 6.8 })];
    useHostStatusStore().status = status({
      models_disk: { total_bytes: 11_507_000_000, free_bytes: 1 },
    });
    const wrapper = mount(InstalledTab);
    await flushPromises();
    const bar = wrapper.get("[data-test='styles-disk-meter']");
    for (const segment of bar.findAll("[role='meter'] > span")) {
      expect(segment.attributes("style")).toMatch(/^width: \d+%;?$/);
    }
  });

  it("caps the meter at four families plus one 'Other'", async () => {
    useModelStore().all = [
      model({ name: "a", family: "flux", size_gb: 10 }),
      model({ name: "b", family: "sdxl", size_gb: 9 }),
      model({ name: "c", family: "wan", size_gb: 8 }),
      model({ name: "d", family: "ltx", size_gb: 7 }),
      model({ name: "e", family: "qwen-image", size_gb: 6 }),
      model({ name: "f", family: "sd35", size_gb: 5 }),
    ];
    useHostStatusStore().status = status({
      models_disk: { total_bytes: 512_000_000_000, free_bytes: 1 },
    });
    const wrapper = mount(InstalledTab);
    await flushPromises();
    const segments = wrapper
      .get("[data-test='styles-disk-meter']")
      .findAll("[role='meter'] > span");
    expect(segments).toHaveLength(5);
    expect(segments.at(-1)!.attributes("title")).toContain("Other");
    // The fold names what it swallowed rather than hiding it.
    expect(segments.at(-1)!.attributes("title")).toContain("Qwen Image");
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
