import { afterEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import LibraryChipRow from "./LibraryChipRow.vue";

const tags = Array.from({ length: 11 }, (_, i) => ({ name: `tag${i + 1}`, count: 20 - i }));

function mountRow(props: Record<string, unknown> = {}) {
  return mount(LibraryChipRow, {
    props: {
      organize: true,
      favoritesOnly: false,
      favoritesCount: 6,
      tags,
      activeTags: [],
      hostChips: [
        { key: "local", label: "This Mac", count: 15 },
        { key: "plato", label: "plato", count: 12 },
      ],
      hostFilter: "all",
      allCount: 24,
      ...props,
    },
  });
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("LibraryChipRow", () => {
  it("shows ♥ Favorites, the top 8 tags with mono counts, More tags… (+N), and host chips", () => {
    const wrapper = mountRow();
    const fav = wrapper.get("[data-test='favorites-chip']");
    expect(fav.text()).toContain("Favorites");
    expect(fav.text()).toContain("6");
    expect(fav.attributes("aria-pressed")).toBe("false");
    const chips = wrapper.findAll("[data-test='tag-chip']");
    expect(chips.map((c) => c.attributes("data-tag"))).toEqual(tags.slice(0, 8).map((t) => t.name));
    expect(chips[0]!.find(".ms-lib-chip__n").text()).toBe("20");
    expect(wrapper.get("[data-test='more-tags']").text()).toContain("+3");
    expect(wrapper.find("[role='tablist']").exists()).toBe(true);
    expect(wrapper.find("[data-test='clear-filters']").exists()).toBe(false);
  });

  it("toggles favorites and tags through emits, and keeps an active hidden tag inline", async () => {
    const wrapper = mountRow({ activeTags: ["tag11"] });
    await wrapper.get("[data-test='favorites-chip']").trigger("click");
    expect(wrapper.emitted("update:favoritesOnly")).toEqual([[true]]);
    const chips = wrapper.findAll("[data-test='tag-chip']");
    expect(chips.map((c) => c.attributes("data-tag"))).toContain("tag11");
    expect(chips.at(-1)!.attributes("aria-pressed")).toBe("true");
    await chips[0]!.trigger("click");
    expect(wrapper.emitted("toggleTag")).toEqual([["tag1"]]);
    await wrapper.get("[data-test='clear-filters']").trigger("click");
    expect(wrapper.emitted("clearFilters")).toHaveLength(1);
  });

  it("opens More tags… as a searchable checkable list", async () => {
    const wrapper = mountRow({ activeTags: ["tag10"] });
    await wrapper.get("[data-test='more-tags']").trigger("click");
    const panel = document.body.querySelector("[data-test='more-tags-panel']");
    expect(panel).not.toBeNull();
    const rows = panel!.querySelectorAll("[data-test='more-tag-row']");
    expect(rows).toHaveLength(11);
    const ten = panel!.querySelector("[data-tag='tag10']")!;
    expect(ten.getAttribute("aria-checked")).toBe("true");
    const input = panel!.querySelector("input") as HTMLInputElement;
    input.value = "tag1";
    input.dispatchEvent(new Event("input"));
    await wrapper.vm.$nextTick();
    expect(panel!.querySelectorAll("[data-test='more-tag-row']")).toHaveLength(3); // tag1, tag10, tag11
    (panel!.querySelector("[data-tag='tag11']") as HTMLButtonElement).click();
    expect(wrapper.emitted("toggleTag")).toEqual([["tag11"]]);
  });

  it("renders the open collection as a removable chip", async () => {
    const wrapper = mountRow({ collectionName: "Smurfs" });
    const chip = wrapper.get("[data-test='collection-chip']");
    expect(chip.text()).toContain("Smurfs");
    await chip.trigger("click");
    expect(wrapper.emitted("exitCollection")).toHaveLength(1);
    expect(wrapper.find("[data-test='clear-filters']").exists()).toBe(true);
  });

  it("hides ♥ and tags when no host can organize but keeps the host chips", () => {
    const wrapper = mountRow({ organize: false });
    expect(wrapper.find("[data-test='favorites-chip']").exists()).toBe(false);
    expect(wrapper.findAll("[data-test='tag-chip']")).toHaveLength(0);
    expect(wrapper.find("[role='tablist']").exists()).toBe(true);
  });

  it("says so quietly when there are no tags yet", () => {
    const wrapper = mountRow({ tags: [] });
    expect(wrapper.get("[data-test='no-tags']").text()).toBe("No tags yet");
    expect(wrapper.find("[data-test='more-tags']").exists()).toBe(false);
  });
});
