import { mount } from "@vue/test-utils";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ref } from "vue";
import CatalogTopbar from "./CatalogTopbar.vue";

const mockSetFilter = vi.fn();
const mockFilter = ref<Record<string, unknown>>({});

vi.mock("../composables/useCatalog", () => {
  return {
    useCatalog: () => ({
      filter: mockFilter,
      setFilter: mockSetFilter,
      loading: ref(false),
    }),
  };
});

describe("CatalogTopbar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFilter.value = {};
  });

  it("renders modality chips", () => {
    const w = mount(CatalogTopbar);
    expect(w.text()).toMatch(/image/i);
    expect(w.text()).toMatch(/video/i);
  });

  it("clicking image chip calls setFilter with modality=image", async () => {
    const w = mount(CatalogTopbar);
    const imageBtn = w
      .findAll("button")
      .find((b) => b.text().toLowerCase().includes("image"));
    expect(imageBtn).toBeDefined();
    await imageBtn!.trigger("click");
    expect(mockSetFilter).toHaveBeenCalledWith(
      expect.objectContaining({ modality: "image" }),
    );
  });

  it("clicking video chip calls setFilter with modality=video", async () => {
    const w = mount(CatalogTopbar);
    const videoBtn = w
      .findAll("button")
      .find((b) => b.text().toLowerCase().includes("video"));
    expect(videoBtn).toBeDefined();
    await videoBtn!.trigger("click");
    expect(mockSetFilter).toHaveBeenCalledWith(
      expect.objectContaining({ modality: "video" }),
    );
  });

  it("search input is present", () => {
    const w = mount(CatalogTopbar);
    const input = w.find("input[type=search]");
    expect(input.exists()).toBe(true);
  });

  it("renders kind chips for models, LoRAs, and components", () => {
    const w = mount(CatalogTopbar);
    expect(w.text()).toMatch(/Models/);
    expect(w.text()).toMatch(/LoRAs/);
    expect(w.text()).toMatch(/CLIP/);
    expect(w.text()).toMatch(/Tokenizers/);
  });

  it("clicking LoRAs chip filters catalog by kind=lora", async () => {
    const w = mount(CatalogTopbar);
    const loraBtn = w.findAll("button").find((b) => b.text() === "LoRAs");
    expect(loraBtn).toBeDefined();
    await loraBtn!.trigger("click");
    expect(mockSetFilter).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "lora" }),
    );
  });

  it("clicking Models chip filters catalog by kind=checkpoint", async () => {
    const w = mount(CatalogTopbar);
    const modelsBtn = w.findAll("button").find((b) => b.text() === "Models");
    expect(modelsBtn).toBeDefined();
    await modelsBtn!.trigger("click");
    expect(mockSetFilter).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "checkpoint" }),
    );
  });

  it("clicking CLIP chip filters catalog by kind=clip", async () => {
    const w = mount(CatalogTopbar);
    const clipBtn = w.findAll("button").find((b) => b.text() === "CLIP");
    expect(clipBtn).toBeDefined();
    await clipBtn!.trigger("click");
    expect(mockSetFilter).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "clip" }),
    );
  });

  it("clicking Tokenizers chip filters catalog by kind=tokenizer", async () => {
    const w = mount(CatalogTopbar);
    const tokenizerBtn = w
      .findAll("button")
      .find((b) => b.text() === "Tokenizers");
    expect(tokenizerBtn).toBeDefined();
    await tokenizerBtn!.trigger("click");
    expect(mockSetFilter).toHaveBeenCalledWith(
      expect.objectContaining({ kind: "tokenizer" }),
    );
  });

  it("clicking the kind 'All' chip clears the kind filter", async () => {
    mockFilter.value = { kind: "lora" };
    const w = mount(CatalogTopbar);
    // Two "All" buttons exist (modality + kind); the kind one is inside the
    // nav with aria-label "Kind filter".
    const kindNav = w.find('nav[aria-label="Kind filter"]');
    expect(kindNav.exists()).toBe(true);
    const allBtn = kindNav.findAll("button").find((b) => b.text() === "All");
    expect(allBtn).toBeDefined();
    await allBtn!.trigger("click");
    expect(mockSetFilter).toHaveBeenCalledWith(
      expect.objectContaining({ kind: undefined }),
    );
  });
});
