import { mount } from "@vue/test-utils";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { ref } from "vue";
import CatalogSidebar from "./CatalogSidebar.vue";

// Mock useCatalog so we control the reactive state
vi.mock("../composables/useCatalog", () => {
  const setFilter = vi.fn();
  const families = ref([{ family: "flux" }, { family: "sdxl" }]);
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
    expect(w.text()).not.toContain("foundation");
    expect(w.text()).not.toContain("fine-tunes");
  });

  it("renders each family as a keyboard-operable button", async () => {
    const { useCatalog } = await import("../composables/useCatalog");
    const cat = useCatalog();
    const w = mount(CatalogSidebar);
    const rows = w.findAll("button[data-family]");
    expect(rows.length).toBeGreaterThan(0);
    expect(rows[0].element.tagName).toBe("BUTTON");
    expect(rows[0].attributes("data-family")).toBe("flux");
    await rows[0].trigger("click");
    expect(cat.setFilter).toHaveBeenCalledWith({ family: "flux" });
  });
});
