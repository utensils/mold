import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import LibraryNavigation from "./LibraryNavigation.vue";
const { push, fetchCollections } = vi.hoisted(() => ({ push: vi.fn(), fetchCollections: vi.fn() }));
vi.mock("vue-router", () => ({
  useRoute: () => ({ path: "/library", query: { scope: "prints" } }),
  useRouter: () => ({ push }),
}));
vi.mock("../../stores/contextMenu", () => ({ useContextMenuStore: () => ({ open: vi.fn() }) }));
vi.mock("../../stores/gallery", () => ({
  useGalleryStore: () => ({
    mergedCollections: [],
    filter: "all",
    anyOrganizeCapable: true,
    anyTrashCapable: true,
    basePrintCount: 2,
    trashCount: 0,
    fetchCollections,
  }),
}));
beforeEach(() => {
  localStorage.clear();
  vi.clearAllMocks();
});
describe("library disclosure controls", () => {
  it("uses visible shared icons and labels the current action", async () => {
    const view = mount(LibraryNavigation);
    const toggle = view.get('button[aria-label="Collapse collections"]');
    expect(toggle.get("svg").attributes("width")).toBe("18");
    await toggle.trigger("click");
    expect(view.find(".library-collections").exists()).toBe(false);
    expect(localStorage.getItem("mold.library.collections")).toBe("closed");
    await view.get('button[aria-label="Expand collections"]').trigger("click");
    expect(view.find(".library-collections").exists()).toBe(true);
    expect(fetchCollections).toHaveBeenCalledOnce();
    expect(push).not.toHaveBeenCalled();
    view.unmount();
  });
  it("collapses Library independently and restores collection expansion", async () => {
    const view = mount(LibraryNavigation);
    await view.get('button[aria-label="Collapse collections"]').trigger("click");
    await view.get('button[aria-label="Collapse library"]').trigger("click");
    expect(view.find(".library-children").exists()).toBe(false);
    await view.get('button[aria-label="Expand library"]').trigger("click");
    expect(view.get('button[aria-label="Expand collections"]').attributes("aria-expanded")).toBe(
      "false",
    );
    view.unmount();
  });
});
