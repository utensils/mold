import { afterEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import BulkBar from "./BulkBar.vue";
import type { MergedCollection } from "@studio/lib/libraryOrganization";

const collections: MergedCollection[] = [
  { slug: "smurfs", name: "Smurfs", count: 9, hosts: [], cover: null },
  { slug: "river", name: "River studies", count: 6, hosts: [], cover: null },
];

function mountBar(props: Record<string, unknown> = {}) {
  return mount(BulkBar, {
    props: {
      selectedCount: 5,
      total: 24,
      scope: "prints",
      organize: true,
      trash: true,
      collections,
      collectionSelected: ["smurfs"],
      collectionMixed: ["river"],
      tags: ["blue"],
      tagSuggestions: [{ name: "outdoor", count: 3 }],
      hostNote: "This Mac · plato",
      ...props,
    },
  });
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("BulkBar (Prints)", () => {
  it("reads N selected, then a spacer, then ★ Favourite · Add tag · Add to album · Export… · Delete", async () => {
    const wrapper = mountBar();
    expect(wrapper.get("[data-test='bulk-count']").text()).toBe("5 selected");
    const cluster = wrapper
      .findAll("button[data-test^='bulk-']")
      .map((el) => el.attributes("data-test"));
    expect(cluster).toEqual([
      "bulk-select-all",
      "bulk-clear",
      "bulk-favorite",
      "bulk-tags",
      "bulk-collections",
      "bulk-export",
      "bulk-delete",
    ]);
    expect(wrapper.find("[data-test='bulk-spacer']").exists()).toBe(true);
    wrapper.unmount();
  });

  it("emits export for the selection", async () => {
    const wrapper = mountBar();
    const button = wrapper.get("[data-test='bulk-export']");
    expect(button.text()).toContain("Export…");
    await button.trigger("click");
    expect(wrapper.emitted("export")).toHaveLength(1);
    wrapper.unmount();
  });

  it("offers Add to album · Add tag · ★ Favourite · Move to trash", async () => {
    const wrapper = mountBar();
    expect(wrapper.text()).toContain("5 selected");
    expect(wrapper.get("[data-test='bulk-collections']").text()).toContain("Add to album");
    expect(wrapper.get("[data-test='bulk-tags']").text()).toContain("Add tag");
    expect(wrapper.get("[data-test='bulk-favorite']").text()).toContain("Favourite");
    const trash = wrapper.get("[data-test='bulk-delete']");
    expect(trash.text()).toContain("Move 5 pictures to trash");
    await trash.trigger("click");
    expect(wrapper.emitted("trash")).toHaveLength(1);
    expect(wrapper.emitted("delete")).toBeUndefined();
  });

  it("toggles favorite: any unfavorited ⇒ favorite all, else unfavorite all", async () => {
    const wrapper = mountBar({ allFavorite: false });
    await wrapper.get("[data-test='bulk-favorite']").trigger("click");
    expect(wrapper.emitted("favorite")).toEqual([[true]]);
    await wrapper.setProps({ allFavorite: true });
    expect(wrapper.get("[data-test='bulk-favorite']").text()).toContain("Unfavourite");
    await wrapper.get("[data-test='bulk-favorite']").trigger("click");
    expect(wrapper.emitted("favorite")?.at(-1)).toEqual([false]);
  });

  it("opens the collection picker with mixed state and relays toggle / create", async () => {
    const wrapper = mountBar();
    await wrapper.get("[data-test='bulk-collections']").trigger("click");
    const panel = document.body.querySelector("[data-test='bulk-collections-panel']")!;
    expect(panel).not.toBeNull();
    expect(panel.textContent).toContain("Add 5 pictures to");
    expect(panel.textContent).toContain("fans out to This Mac · plato");
    const rows = panel.querySelectorAll("[data-test='collection-row']");
    expect(rows[0]!.getAttribute("aria-checked")).toBe("true");
    expect(rows[1]!.getAttribute("aria-checked")).toBe("mixed");
    (rows[1] as HTMLButtonElement).click();
    expect(wrapper.emitted("toggleCollection")).toEqual([["river", true]]);
    (panel.querySelector("[data-test='collection-new']") as HTMLButtonElement).click();
    await wrapper.vm.$nextTick();
    const input = panel.querySelector("[data-test='collection-new-input']") as HTMLInputElement;
    input.value = "Halcyon";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    expect(wrapper.emitted("createCollection")).toEqual([["Halcyon"]]);
  });

  it("opens the tag editor over the intersection and relays add / remove", async () => {
    const wrapper = mountBar();
    await wrapper.get("[data-test='bulk-tags']").trigger("click");
    const panel = document.body.querySelector("[data-test='bulk-tags-panel']")!;
    expect(panel.querySelectorAll("[data-test='tag-chip']")).toHaveLength(1);
    (panel.querySelector("[data-test='tag-remove']") as HTMLButtonElement).click();
    expect(wrapper.emitted("removeTags")).toEqual([[["blue"]]]);
    const input = panel.querySelector("[data-test='tag-input']") as HTMLInputElement;
    input.value = "outdoor";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    expect(wrapper.emitted("addTags")).toEqual([[["outdoor"]]]);
  });

  it("keeps the two-press hard-delete arming on hosts without a trash", async () => {
    const wrapper = mountBar({ trash: false, selectedCount: 2 });
    const button = wrapper.get("[data-test='bulk-delete']");
    expect(button.text()).toBe("Delete selected");
    await button.trigger("click");
    expect(wrapper.emitted("update:confirming")).toEqual([[true]]);
    expect(wrapper.emitted("delete")).toBeUndefined();
    await wrapper.setProps({ confirming: true });
    expect(wrapper.get("[data-test='bulk-delete']").text()).toBe(
      "Delete 2 pictures? This can't be undone.",
    );
    await wrapper.get("[data-test='bulk-delete']").trigger("click");
    expect(wrapper.emitted("delete")).toHaveLength(1);
  });

  it("hides organization actions when no host can organize", () => {
    const wrapper = mountBar({ organize: false });
    expect(wrapper.find("[data-test='bulk-collections']").exists()).toBe(false);
    expect(wrapper.find("[data-test='bulk-tags']").exists()).toBe(false);
    expect(wrapper.find("[data-test='bulk-favorite']").exists()).toBe(false);
    expect(wrapper.find("[data-test='bulk-delete']").exists()).toBe(true);
  });

  it("offers Remove from album inside a drill-in", async () => {
    const wrapper = mountBar({ collectionName: "Smurfs" });
    await wrapper.get("[data-test='bulk-remove-from-collection']").trigger("click");
    expect(wrapper.emitted("removeFromCollection")).toHaveLength(1);
  });
});

describe("BulkBar (Trash)", () => {
  it("offers Restore and Delete forever only", async () => {
    const wrapper = mountBar({ scope: "trash", selectedCount: 2 });
    expect(wrapper.find("[data-test='bulk-collections']").exists()).toBe(false);
    expect(wrapper.find("[data-test='bulk-delete']").exists()).toBe(false);
    await wrapper.get("[data-test='bulk-restore']").trigger("click");
    expect(wrapper.emitted("restore")).toHaveLength(1);
    await wrapper.get("[data-test='bulk-delete-forever']").trigger("click");
    expect(wrapper.emitted("deleteForever")).toHaveLength(1);
  });
});

describe("BulkBar export progress", () => {
  it("says Exporting… while the batch runs, and never borrows the delete label", () => {
    const wrapper = mountBar({ exporting: true });
    const button = wrapper.get("[data-test='bulk-export']");
    expect(button.text()).toBe("Exporting…");
    expect(button.attributes("disabled")).toBeDefined();
    // The delete button keeps its own wording — exporting is not deleting.
    expect(wrapper.get("[data-test='bulk-delete']").text()).not.toContain("Deleting");
  });

  it("reads Export… when no batch is running", () => {
    const wrapper = mountBar();
    expect(wrapper.get("[data-test='bulk-export']").text()).toBe("Export…");
  });
});
