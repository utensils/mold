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
});
