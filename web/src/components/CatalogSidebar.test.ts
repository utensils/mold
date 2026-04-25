import { mount } from "@vue/test-utils";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { ref } from "vue";
import CatalogSidebar from "./CatalogSidebar.vue";

// Mock useCatalog so we control the reactive state
vi.mock("../composables/useCatalog", () => {
  const setFilter = vi.fn();
  const families = ref([
    { family: "flux", foundation: 2, finetune: 5 },
    { family: "sdxl", foundation: 1, finetune: 3 },
  ]);
  const filter = ref<{ family?: string }>({});
  return {
    useCatalog: () => ({ families, filter, setFilter }),
  };
});

describe("CatalogSidebar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders family rows from composable", () => {
    const w = mount(CatalogSidebar);
    expect(w.text()).toContain("flux");
    expect(w.text()).toContain("sdxl");
    expect(w.text()).toContain("2");
    expect(w.text()).toContain("5");
  });

  it("clicking a family row calls setFilter({ family })", async () => {
    const { useCatalog } = await import("../composables/useCatalog");
    const cat = useCatalog();
    const w = mount(CatalogSidebar);
    const rows = w.findAll("li");
    expect(rows.length).toBeGreaterThan(0);
    await rows[0].trigger("click");
    expect(cat.setFilter).toHaveBeenCalledWith({ family: "flux" });
  });
});
