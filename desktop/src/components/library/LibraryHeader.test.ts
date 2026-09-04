import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import LibraryHeader from "./LibraryHeader.vue";
import type { GalleryKindFilter, LibraryScope } from "../../stores/gallery";

const kindOptions: { value: GalleryKindFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "image", label: "Pictures" },
];

function mountHeader(props: Partial<InstanceType<typeof LibraryHeader>["$props"]> = {}) {
  return mount(LibraryHeader, {
    props: {
      scope: "prints" as LibraryScope,
      scopes: ["prints", "favorites", "collections", "trash"] as LibraryScope[],
      counts: { prints: 24, favorites: 6, collections: 4, trash: 3 },
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
  it("renders the four scopes with mono counts and no second count label", () => {
    const wrapper = mountHeader();
    const scope = wrapper.get("[data-test='library-scope']");
    const labels = scope
      .findAll("button")
      .map((b) => `${b.find(".ms-seg__label").text()} ${b.find(".ms-seg__sub").text()}`);
    expect(labels).toEqual(["Everything 24", "Favourites 6", "Albums 4", "Trash 3"]);
    expect(scope.findAll("button")[0]!.attributes("aria-checked")).toBe("true");
    // The shell's title bar already says "24 pictures · 4 albums".
    expect(wrapper.find("[data-test='library-count']").exists()).toBe(false);
    expect(wrapper.find("input[aria-label='Thumbnail size']").exists()).toBe(true);
    expect(wrapper.find("[aria-label='Media kind']").exists()).toBe(true);
    expect(wrapper.find("[aria-label='Refresh my images']").exists()).toBe(true);
  });

  it("orders the toolbar scope → search → kind → spacer → size → Select → History", () => {
    const wrapper = mountHeader();
    const order = wrapper
      .get("[data-test='library-header']")
      .findAll(
        "[data-test='library-scope'], input[type='search'], [aria-label='Media kind'], input[aria-label='Thumbnail size'], [aria-label='Toggle select mode'], [aria-label='Open history']",
      )
      .map((el) => el.attributes("data-test") ?? el.attributes("aria-label"));
    expect(order).toEqual([
      "library-scope",
      "Search pictures",
      "Media kind",
      "Thumbnail size",
      "Toggle select mode",
      "Open history",
    ]);
  });

  it("emits scope changes from the control (click and keyboard)", async () => {
    const wrapper = mountHeader();
    const buttons = wrapper.get("[data-test='library-scope']").findAll("button");
    await buttons[1]!.trigger("click");
    expect(wrapper.emitted("update:scope")).toEqual([["favorites"]]);
    await buttons[0]!.trigger("keydown", { key: "ArrowRight" });
    expect(wrapper.emitted("update:scope")?.at(-1)).toEqual(["favorites"]);
  });

  it("keeps the slider and the kind control in every scope", () => {
    for (const scope of ["favorites", "collections", "trash"] as LibraryScope[]) {
      const wrapper = mountHeader({ scope });
      expect(wrapper.find("input[aria-label='Thumbnail size']").exists()).toBe(true);
      expect(wrapper.find("[aria-label='Media kind']").exists()).toBe(true);
      expect(wrapper.find("[aria-label='Toggle select mode']").exists()).toBe(true);
      wrapper.unmount();
    }
    expect(
      mountHeader({ scope: "collections" }).get("input[type='search']").attributes("placeholder"),
    ).toBe("Search albums…");
  });

  it("never carries Empty trash — that lives in the trash banner now", () => {
    const wrapper = mountHeader({ scope: "trash" });
    expect(wrapper.find("[data-test='empty-trash']").exists()).toBe(false);
    expect(wrapper.find("[aria-label='Refresh my images']").exists()).toBe(true);
  });

  it("renders no scope control when the hosts only offer Prints", () => {
    const wrapper = mountHeader({ scopes: ["prints"] });
    expect(wrapper.find("[data-test='library-scope']").exists()).toBe(false);
  });

  it("forwards search, select, history, and refresh, and marks History open", async () => {
    const wrapper = mountHeader();
    await wrapper.get("input[type='search']").setValue("cat");
    expect(wrapper.emitted("update:search")).toEqual([["cat"]]);
    await wrapper.get("[aria-label='Toggle select mode']").trigger("click");
    expect(wrapper.emitted("toggleSelect")).toHaveLength(1);
    await wrapper.get("[aria-label='Open history']").trigger("click");
    expect(wrapper.emitted("toggleHistory")).toHaveLength(1);
    await wrapper.get("[aria-label='Refresh my images']").trigger("click");
    expect(wrapper.emitted("refresh")).toHaveLength(1);

    await wrapper.setProps({ historyOpen: true });
    const history = wrapper.get("[aria-label='Close history']");
    expect(history.classes()).toContain("ms-toolbar-button--on");
    await history.trigger("click");
    expect(wrapper.emitted("toggleHistory")).toHaveLength(2);
  });

  /**
   * This row is one 40px line that never wraps, and `.ms-seg__btn` is
   * `flex: 1` + `white-space: nowrap`, so the segmented groups cannot give.
   * Measured at the app's own 1080px minimum the row asked for ~1105px and
   * `<main>`'s `overflow-hidden` ate Refresh, then History, then Select.
   */
  it("keeps the toolbar able to shrink: the search gives, and the kind control is flat chips", () => {
    const wrapper = mountHeader();
    const search = wrapper.get("input[type='search']").element.parentElement!;
    const classes = [...search.classList];
    // A fixed width in a nowrap row is a floor nothing can get under.
    expect(classes).not.toContain("w-[170px]");
    expect(classes).toContain("min-w-0");
    expect(classes).toContain("shrink");
    expect(classes).toContain("basis-[170px]");

    // The mock's media-kind control is unbordered chips, not a second
    // segmented group with its own border, padding and 7px insets.
    const kind = wrapper.get("[aria-label='Media kind']");
    expect(kind.classes()).not.toContain("ms-seg");
    expect(kind.findAll("[role='radio']")).toHaveLength(kindOptions.length);
  });

  it("marks the chosen media kind and emits the others", async () => {
    const wrapper = mountHeader();
    const chips = wrapper.get("[aria-label='Media kind']").findAll("[role='radio']");
    expect(chips.map((c) => c.text())).toEqual(["All", "Pictures"]);
    expect(chips[0]!.attributes("aria-checked")).toBe("true");
    expect(chips[1]!.attributes("aria-checked")).toBe("false");
    await chips[1]!.trigger("click");
    expect(wrapper.emitted("update:mediaKind")).toEqual([["image"]]);
  });
});
