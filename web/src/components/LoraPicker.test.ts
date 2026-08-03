import { mount, flushPromises } from "@vue/test-utils";
import { describe, expect, it, vi, beforeEach } from "vitest";
import LoraPicker from "./LoraPicker.vue";
import type { CatalogEntryWire, LoraSelection } from "../types";

// Two synthetic FLUX LoRAs the picker will see as "installed".
function fakeEntry(
  id: string,
  name: string,
  primaryPath: string,
  trainedWords: string[] = [],
): CatalogEntryWire {
  return {
    id,
    source: "civitai",
    source_id: id.replace("cv:", ""),
    name,
    author: null,
    family: "flux",
    family_role: "finetune",
    sub_family: null,
    modality: "image",
    kind: "lora",
    file_format: "safetensors",
    bundling: "single-file",
    size_bytes: 100_000_000,
    download_count: 0,
    rating: null,
    likes: 0,
    nsfw: false,
    thumbnail_url: null,
    description: null,
    license: null,
    license_flags: null,
    tags: [],
    companions: [],
    download_recipe: { files: [], needs_token: null },
    supported: true,
    installed: true,
    primary_path: primaryPath,
    created_at: null,
    updated_at: null,
    added_at: 0,
    trained_words: trainedWords,
  };
}

const installed = [
  fakeEntry("cv:1", "Cinematic", "/loras/cinematic.safetensors", [
    "cinematic style",
    "moody lighting",
  ]),
  fakeEntry("cv:2", "Pixel Art", "/loras/pixel.safetensors", ["pixel art"]),
];

vi.mock("../api", () => ({
  fetchCatalogInstalled: vi.fn(async () => ({
    entries: installed,
    page: 1,
    page_size: installed.length,
    total: installed.length,
  })),
}));

function mountPicker(modelValue: LoraSelection[] = []) {
  return mount(LoraPicker, {
    props: { family: "flux", modelValue },
    global: { stubs: { RouterLink: { template: "<a><slot /></a>" } } },
  });
}

describe("LoraPicker — multi-LoRA stack", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows a helpful empty state when no LoRAs are installed and the stack is empty", async () => {
    const { fetchCatalogInstalled } = await import("../api");
    (
      fetchCatalogInstalled as unknown as ReturnType<typeof vi.fn>
    ).mockResolvedValueOnce({
      entries: [],
      page: 1,
      page_size: 0,
      total: 0,
    });
    const w = mountPicker([]);
    await flushPromises();
    // Old behaviour rendered nothing (a blank drawer section). The picker now
    // explains the empty state and points at Models rather than going dark.
    const empty = w.find("[data-test='lora-hint-empty']");
    expect(empty.exists()).toBe(true);
    expect(empty.text().toLowerCase()).toContain("no loras installed");
    expect(w.findAll("[data-test='lora-row']")).toHaveLength(0);
  });

  it("clicking + Add LoRA appends a row keyed to the first available LoRA", async () => {
    const w = mountPicker([]);
    await flushPromises();
    const add = w.findAll("button").find((b) => b.text().includes("Add LoRA"));
    expect(add).toBeDefined();
    await add!.trigger("click");
    const last = w.emitted("update:modelValue")?.at(-1)?.[0] as LoraSelection[];
    expect(last).toHaveLength(1);
    expect(last[0].path).toBe("/loras/cinematic.safetensors");
    expect(last[0].scale).toBe(1.0);
    expect(last[0].trainedWords).toEqual(["cinematic style", "moody lighting"]);
  });

  it("Add LoRA picks a LoRA not already in the stack", async () => {
    const w = mountPicker([
      {
        path: "/loras/cinematic.safetensors",
        scale: 1.0,
      },
    ]);
    await flushPromises();
    const add = w.findAll("button").find((b) => b.text().includes("Add LoRA"));
    await add!.trigger("click");
    const last = w.emitted("update:modelValue")?.at(-1)?.[0] as LoraSelection[];
    expect(last).toHaveLength(2);
    expect(last[1].path).toBe("/loras/pixel.safetensors");
  });

  it("rendering existing rows shows trigger-word chips", async () => {
    const w = mountPicker([
      {
        path: "/loras/cinematic.safetensors",
        scale: 1.0,
        trainedWords: ["cinematic style", "moody lighting"],
      },
    ]);
    await flushPromises();
    expect(w.text()).toContain("cinematic style");
    expect(w.text()).toContain("moody lighting");
  });

  it("renders an auto-download camera adapter before it is installed", async () => {
    const w = mountPicker([{ path: "camera-control:dolly-in", scale: 0.5 }]);
    await flushPromises();
    const row = w.get("[data-test='lora-row']");
    expect(row.get("select").text()).toContain("Dolly in camera control");
    expect(
      (row.get("input[type='range']").element as HTMLInputElement).value,
    ).toBe("0.5");
  });

  it("clicking a trigger-word chip emits append-prompt with the phrase", async () => {
    const w = mountPicker([
      {
        path: "/loras/cinematic.safetensors",
        scale: 1.0,
        trainedWords: ["cinematic style"],
      },
    ]);
    await flushPromises();
    // The chip reads "+ cinematic style" (an add affordance) but the event
    // carries the bare phrase so the prompt gets clean text.
    const chip = w
      .findAll("button")
      .find((b) => b.text().includes("cinematic style"));
    expect(chip).toBeDefined();
    await chip!.trigger("click");
    expect(w.emitted("append-prompt")?.at(-1)?.[0]).toBe("cinematic style");
  });

  it("clicking the remove button on a row removes it from the stack", async () => {
    const w = mountPicker([
      { path: "/loras/cinematic.safetensors", scale: 1.0 },
      { path: "/loras/pixel.safetensors", scale: 0.6 },
    ]);
    await flushPromises();
    // The control is a kit Icon with no text, so it is addressed by its
    // accessible name — the spec voice bans emoji glyphs as buttons.
    const removeBtns = w.findAll("[aria-label='Remove this LoRA']");
    expect(removeBtns).toHaveLength(2);
    await removeBtns[0].trigger("click");
    const last = w.emitted("update:modelValue")?.at(-1)?.[0] as LoraSelection[];
    expect(last).toHaveLength(1);
    expect(last[0].path).toBe("/loras/pixel.safetensors");
  });

  it("renders row controls as kit icons, never emoji glyphs", async () => {
    const w = mountPicker([{ path: "/loras/cinematic.safetensors", scale: 1 }]);
    await flushPromises();
    expect(w.html()).not.toMatch(
      /[\u{1F000}-\u{1FAFF}\u{2190}-\u{21FF}\u{2300}-\u{27BF}\u{2B00}-\u{2BFF}\u{FE0F}]/u,
    );
  });

  it("uses only explicit up/down buttons for row movement", async () => {
    const rows: LoraSelection[] = [
      { path: "/loras/cinematic.safetensors", scale: 0.8 },
      { path: "/loras/pixel.safetensors", scale: 1.15 },
    ];
    const w = mountPicker(rows);
    await flushPromises();

    const renderedRows = w.findAll("[data-test='lora-row']");
    expect(renderedRows[0].attributes("draggable")).not.toBe("true");
    expect(w.find("[data-test='lora-drag-handle']").exists()).toBe(false);

    const scale = renderedRows[0].get("input[type='range']");
    await scale.setValue("1.35");

    const emitted = w.emitted("update:modelValue") ?? [];
    expect(emitted).toHaveLength(1);
    expect(emitted[0][0]).toEqual([
      { path: "/loras/cinematic.safetensors", scale: 1.35 },
      rows[1],
    ]);

    await w.setProps({ modelValue: emitted[0][0] as LoraSelection[] });
    await w.find("[aria-label='Move LoRA down']").trigger("click");
    const moved = w
      .emitted("update:modelValue")
      ?.at(-1)?.[0] as LoraSelection[];
    expect(moved).toEqual([rows[1], { ...rows[0], scale: 1.35 }]);
  });

  it("exposes button fallback controls for moving LoRA rows", async () => {
    const rows: LoraSelection[] = [
      { path: "/loras/cinematic.safetensors", scale: 0.8 },
      { path: "/loras/pixel.safetensors", scale: 1.15 },
    ];
    const w = mountPicker(rows);
    await flushPromises();

    const moveDown = w.find("[aria-label='Move LoRA down']");
    expect(moveDown.exists()).toBe(true);
    await moveDown.trigger("click");

    const last = w.emitted("update:modelValue")?.at(-1)?.[0] as LoraSelection[];
    expect(last).toEqual([rows[1], rows[0]]);
  });

  it("trigger-words backfill from the catalog when the form-state row lacks them", async () => {
    // Mimics localStorage-restored state: row was saved before trigger
    // words were captured. The picker must still surface the chips by
    // matching the row path against the catalog entry.
    const w = mountPicker([
      { path: "/loras/cinematic.safetensors", scale: 1.0 },
    ]);
    await flushPromises();
    expect(w.text()).toContain("cinematic style");
  });
});
