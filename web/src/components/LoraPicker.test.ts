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
    engine_phase: 1,
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
  });
}

describe("LoraPicker — multi-LoRA stack", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders nothing when no LoRAs are installed and the stack is empty", async () => {
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
    expect(w.find("section").exists()).toBe(false);
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

  it("clicking a trigger-word chip emits append-prompt with the phrase", async () => {
    const w = mountPicker([
      {
        path: "/loras/cinematic.safetensors",
        scale: 1.0,
        trainedWords: ["cinematic style"],
      },
    ]);
    await flushPromises();
    const chip = w
      .findAll("button")
      .find((b) => b.text() === "cinematic style");
    expect(chip).toBeDefined();
    await chip!.trigger("click");
    expect(w.emitted("append-prompt")?.at(-1)?.[0]).toBe("cinematic style");
  });

  it("clicking the ✕ button on a row removes it from the stack", async () => {
    const w = mountPicker([
      { path: "/loras/cinematic.safetensors", scale: 1.0 },
      { path: "/loras/pixel.safetensors", scale: 0.6 },
    ]);
    await flushPromises();
    const removeBtns = w.findAll("button").filter((b) => b.text() === "✕");
    expect(removeBtns).toHaveLength(2);
    await removeBtns[0].trigger("click");
    const last = w.emitted("update:modelValue")?.at(-1)?.[0] as LoraSelection[];
    expect(last).toHaveLength(1);
    expect(last[0].path).toBe("/loras/pixel.safetensors");
  });

  it("dragging a row reorders the visible LoRA stack without dropping row metadata", async () => {
    const rows: LoraSelection[] = [
      {
        path: "/loras/cinematic.safetensors",
        scale: 0.8,
        trainedWords: ["cinematic style"],
      },
      {
        path: "/loras/pixel.safetensors",
        scale: 1.15,
        trainedWords: ["pixel art"],
      },
      {
        path: "/loras/third.safetensors",
        scale: 0.45,
        trainedWords: ["third style"],
      },
    ];
    const w = mountPicker(rows);
    await flushPromises();

    const data = new Map<string, string>();
    const dataTransfer = {
      effectAllowed: "",
      dropEffect: "",
      setData: vi.fn((key: string, value: string) => data.set(key, value)),
      getData: vi.fn((key: string) => data.get(key) ?? ""),
    };
    const renderedRows = w.findAll("[data-test='lora-row']");

    await renderedRows[0].trigger("dragstart", { dataTransfer });
    await renderedRows[2].trigger("drop", { dataTransfer });

    const last = w.emitted("update:modelValue")?.at(-1)?.[0] as LoraSelection[];
    expect(last).toEqual([rows[1], rows[2], rows[0]]);
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
