import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import LibraryHeader from "./LibraryHeader.vue";
import type { GalleryKindFilter, LibraryScope } from "../../stores/gallery";

const kindOptions: { value: GalleryKindFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "image", label: "Images" },
];

function mountHeader(props: Partial<InstanceType<typeof LibraryHeader>["$props"]> = {}) {
  return mount(LibraryHeader, {
    props: {
      scope: "prints" as LibraryScope,
      scopes: ["prints", "collections", "trash"] as LibraryScope[],
      counts: { prints: 24, collections: 4, trash: 3 },
      countLabel: "24 prints · 3.1 GB",
      thumbnailSize: 220,
      mediaKind: "all" as GalleryKindFilter,
      kindOptions,
      search: "",
      selectMode: false,
      ...props,
    },
  });
}

describe("LibraryHeader", () => {
  it("renders the scope control with mono counts and the per-scope count label", () => {
    const wrapper = mountHeader();
    const scope = wrapper.get("[data-test='library-scope']");
    const labels = scope
      .findAll("button")
      .map((b) => `${b.find(".ms-seg__label").text()} ${b.find(".ms-seg__sub").text()}`);
    expect(labels).toEqual(["Everything 24", "Albums 4", "Trash 3"]);
    expect(scope.findAll("button")[0]!.attributes("aria-checked")).toBe("true");
    expect(wrapper.get("[data-test='library-count']").text()).toBe("24 prints · 3.1 GB");
    expect(wrapper.find("input[aria-label='Thumbnail size']").exists()).toBe(true);
    expect(wrapper.find("[aria-label='Media kind']").exists()).toBe(true);
    expect(wrapper.find("[aria-label='Refresh my images']").exists()).toBe(true);
    expect(wrapper.find("[data-test='empty-trash']").exists()).toBe(false);
  });

  it("emits scope changes from the control (click and keyboard)", async () => {
    const wrapper = mountHeader();
    const buttons = wrapper.get("[data-test='library-scope']").findAll("button");
    await buttons[1]!.trigger("click");
    expect(wrapper.emitted("update:scope")).toEqual([["collections"]]);
    await buttons[0]!.trigger("keydown", { key: "ArrowRight" });
    expect(wrapper.emitted("update:scope")?.at(-1)).toEqual(["collections"]);
  });

  it("hides the slider and kind control in Albums", () => {
    const wrapper = mountHeader({ scope: "collections", countLabel: "4 collections" });
    expect(wrapper.find("input[aria-label='Thumbnail size']").exists()).toBe(false);
    expect(wrapper.find("[aria-label='Media kind']").exists()).toBe(false);
    expect(wrapper.get("input[type='search']").attributes("placeholder")).toBe("Search albums…");
    expect(wrapper.find("[aria-label='Toggle select mode']").exists()).toBe(true);
  });

  it("keeps slider + Select in Trash and swaps Refresh for a danger Empty trash", async () => {
    const wrapper = mountHeader({
      scope: "trash",
      countLabel: "3 prints in trash · 41.6 MB",
      trashCount: 3,
    });
    expect(wrapper.find("input[aria-label='Thumbnail size']").exists()).toBe(true);
    expect(wrapper.find("[aria-label='Media kind']").exists()).toBe(false);
    expect(wrapper.find("[aria-label='Refresh my images']").exists()).toBe(false);
    const empty = wrapper.get("[data-test='empty-trash']");
    expect(empty.text()).toBe("Empty trash");
    expect(empty.classes()).toContain("ms-toolbar-button--danger");
    await empty.trigger("click");
    expect(wrapper.emitted("emptyTrash")).toHaveLength(1);
  });

  it("disables Empty trash at zero", () => {
    const wrapper = mountHeader({ scope: "trash", trashCount: 0 });
    expect(wrapper.get("[data-test='empty-trash']").attributes("disabled")).toBeDefined();
  });

  it("renders no scope control when the hosts only offer Prints", () => {
    const wrapper = mountHeader({ scopes: ["prints"] });
    expect(wrapper.find("[data-test='library-scope']").exists()).toBe(false);
    // The shell's title already names the view; the header keeps the count.
    expect(wrapper.find("[data-test='library-count']").exists()).toBe(true);
  });

  it("forwards search, select, history, and refresh", async () => {
    const wrapper = mountHeader();
    await wrapper.get("input[type='search']").setValue("cat");
    expect(wrapper.emitted("update:search")).toEqual([["cat"]]);
    await wrapper.get("[aria-label='Toggle select mode']").trigger("click");
    expect(wrapper.emitted("toggleSelect")).toHaveLength(1);
    await wrapper.get("[aria-label='Open history']").trigger("click");
    expect(wrapper.emitted("openHistory")).toHaveLength(1);
    await wrapper.get("[aria-label='Refresh my images']").trigger("click");
    expect(wrapper.emitted("refresh")).toHaveLength(1);
  });
});
