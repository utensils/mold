import { mount } from "@vue/test-utils";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { ref } from "vue";
import CatalogTopbar from "./CatalogTopbar.vue";

const mockSetFilter = vi.fn();
const mockSetLayout = vi.fn();
const mockFilter = ref<Record<string, unknown>>({});
const mockLayout = ref<"grid" | "list">("grid");

vi.mock("../composables/useCatalog", () => {
  return {
    useCatalog: () => ({
      filter: mockFilter,
      setFilter: mockSetFilter,
      layout: mockLayout,
      setLayout: mockSetLayout,
      loading: ref(false),
    }),
  };
});

describe("CatalogTopbar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFilter.value = {};
    mockLayout.value = "grid";
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

  it("offers the shared List and Grid layout control", async () => {
    const w = mount(CatalogTopbar);
    const group = w.find('[aria-label="Catalog layout"]');
    expect(group.exists()).toBe(true);
    expect(group.findAll("[role=radio]").map((radio) => radio.text())).toEqual([
      "List",
      "Grid",
    ]);
    await group.get("[data-test='layout-list']").trigger("click");
    expect(mockSetLayout).toHaveBeenCalledWith("list");
  });

  it("uses theme text tokens for inactive chip hover states", () => {
    const w = mount(CatalogTopbar);
    const buttons = w
      .findAll("nav button")
      .filter((button) => !button.classes().includes("bg-safelight"));
    expect(buttons.length).toBeGreaterThan(0);
    for (const button of buttons) {
      expect(button.classes()).toContain("hover:text-rebate");
      expect(button.classes()).not.toContain("hover:text-white");
    }
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

  it("checking Include NSFW sets include_nsfw filter", async () => {
    const w = mount(CatalogTopbar);
    const box = w.find("input[type=checkbox]");
    expect(box.exists()).toBe(true);
    await box.setValue(true);
    expect(mockSetFilter).toHaveBeenCalledWith(
      expect.objectContaining({ include_nsfw: true }),
    );
  });

  it("unchecking Include NSFW sends include_nsfw=false rather than omitting it", async () => {
    // The server's `include_nsfw` defaults to *true* when the parameter is
    // absent, so dropping it on uncheck left NSFW rows in the grid while the
    // box read as off. The off state has to be stated explicitly.
    mockFilter.value = { include_nsfw: true };
    const w = mount(CatalogTopbar);
    const box = w.find("input[type=checkbox]");
    await box.setValue(false);
    expect(mockSetFilter).toHaveBeenCalledWith(
      expect.objectContaining({ include_nsfw: false }),
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

describe("CatalogTopbar sort vocabulary", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFilter.value = {};
  });

  it("offers only sorts an upstream can actually serve", () => {
    // "name" was never implemented anywhere — the server 422s it, so it must
    // not be offerable. downloads/recent/rating are the honest vocabulary.
    const w = mount(CatalogTopbar);
    const values = w
      .findAll("option")
      .map((o) => o.attributes("value"))
      .filter((v): v is string => typeof v === "string");
    expect(values).toEqual(
      expect.arrayContaining(["downloads", "recent", "rating"]),
    );
    expect(values).not.toContain("name");
  });
});
