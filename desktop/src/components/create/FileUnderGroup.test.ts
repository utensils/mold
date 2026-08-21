import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import FileUnderGroup from "./FileUnderGroup.vue";
import {
  addTag,
  emptyFileUnderState,
  pickCollection,
  type FileUnderState,
} from "@studio/lib/fileUnder";

const COLLECTIONS = [
  { id: "c1", name: "Smurf Village", slug: "smurf-village", count: 12 },
  { id: "c2", name: "River studies", slug: "river-studies", count: 8 },
];

const TAGS = [
  { name: "blue", count: 9 },
  { name: "#grain", count: 4 },
  { name: "bluebell", count: 2 },
];

function mountGroup(
  overrides: Partial<{
    title: string;
    state: FileUnderState;
    autoTagTitle: boolean;
    tags: typeof TAGS;
    collections: typeof COLLECTIONS;
    model: string;
    extension: string;
    batchSize: number;
    outputKind: "print" | "sequence";
  }> = {},
) {
  return mount(FileUnderGroup, {
    props: {
      title: "Smurf Village",
      state: emptyFileUnderState(),
      autoTagTitle: true,
      tags: TAGS,
      collections: COLLECTIONS,
      model: "z-image-turbo:bf16",
      extension: "png",
      ...overrides,
    },
  });
}

/** The last state the component emitted. */
function emitted(wrapper: ReturnType<typeof mountGroup>): FileUnderState {
  const events = wrapper.emitted("update:state");
  expect(events).toBeTruthy();
  return events!.at(-1)![0] as FileUnderState;
}

describe("FileUnderGroup — ghost chip", () => {
  it("derives the dashed chip from the title", () => {
    const wrapper = mountGroup();
    const ghost = wrapper.get("[data-test='file-under-ghost-tag']");
    expect(ghost.text()).toContain("smurf-village");
    expect(ghost.text()).toContain("from title");
  });

  it("shows no chip while auto-tagging is off", () => {
    const wrapper = mountGroup({ autoTagTitle: false });
    expect(wrapper.find("[data-test='file-under-ghost-tag']").exists()).toBe(false);
  });

  it("shows no chip for an untitled print", () => {
    const wrapper = mountGroup({ title: "" });
    expect(wrapper.find("[data-test='file-under-ghost-tag']").exists()).toBe(false);
  });

  it("shows no chip for a title that slugs to nothing", () => {
    const wrapper = mountGroup({ title: "日本語" });
    expect(wrapper.find("[data-test='file-under-ghost-tag']").exists()).toBe(false);
  });

  it("removing it records the opt-out rather than editing the title", async () => {
    const wrapper = mountGroup();
    await wrapper.get("[data-test='file-under-ghost-remove']").trigger("click");
    expect(emitted(wrapper).ghostRemoved).toBe(true);
  });

  it("stays gone once removed while the title keeps deriving the same slug", () => {
    const wrapper = mountGroup({ state: { ...emptyFileUnderState(), ghostRemoved: true } });
    expect(wrapper.find("[data-test='file-under-ghost-tag']").exists()).toBe(false);
  });
});

describe("FileUnderGroup — tags", () => {
  it("adds a typed tag on Enter", async () => {
    const wrapper = mountGroup();
    const input = wrapper.get("[data-test='file-under-tag-input']");
    await input.setValue("kodak gold");
    await input.trigger("keydown.enter");
    expect(emitted(wrapper).manualTags).toEqual(["kodak gold"]);
  });

  // The one rule `addTag` deliberately does NOT enforce: the `#` split lives
  // at the entry point, so the same text takes two different paths.
  it("strips a leading # from TYPED input", async () => {
    const wrapper = mountGroup();
    const input = wrapper.get("[data-test='file-under-tag-input']");
    await input.setValue("#kodak");
    await input.trigger("keydown.enter");
    expect(emitted(wrapper).manualTags).toEqual(["kodak"]);
  });

  it("keeps the SAME leading # when the host reported it as a suggestion", async () => {
    const wrapper = mountGroup({ tags: [{ name: "#kodak", count: 3 }] });
    await wrapper.get("[data-test='file-under-tag-input']").setValue("kodak");
    const rows = wrapper.findAll("[data-test='file-under-tag-suggestion']");
    expect(rows.map((row) => row.text())).toContain("#kodak3");
    // mousedown, not click: the input keeps focus so several tags can be
    // picked in a row.
    await rows[0]!.trigger("mousedown");
    expect(emitted(wrapper).manualTags).toEqual(["#kodak"]);
  });

  it("adds a host-reported #tag verbatim from the suggestion list", async () => {
    const wrapper = mountGroup();
    await wrapper.get("[data-test='file-under-tag-input']").setValue("grain");
    const rows = wrapper.findAll("[data-test='file-under-tag-suggestion']");
    expect(rows.map((row) => row.text())).toContain("#grain4");
    await rows[0]!.trigger("mousedown");
    expect(emitted(wrapper).manualTags).toEqual(["#grain"]);
  });

  it("orders suggestions prefix-first with counts and hides tags already on the print", async () => {
    const wrapper = mountGroup({ state: addTag(emptyFileUnderState(), "bluebell") });
    await wrapper.get("[data-test='file-under-tag-input']").setValue("blue");
    const rows = wrapper.findAll("[data-test='file-under-tag-suggestion']");
    expect(rows).toHaveLength(1);
    expect(rows[0]!.text()).toContain("blue");
    expect(rows[0]!.text()).toContain("9");
  });

  it("names the popover's Enter contract", async () => {
    const wrapper = mountGroup();
    await wrapper.get("[data-test='file-under-tag-input']").setValue("blue");
    expect(wrapper.get("[data-test='file-under-tag-suggestions']").text()).toContain(
      "↵ adds · new names are created on develop",
    );
  });

  it("refuses a duplicate at the input instead of at submit", async () => {
    const wrapper = mountGroup({ state: addTag(emptyFileUnderState(), "blue") });
    const input = wrapper.get("[data-test='file-under-tag-input']");
    await input.setValue("BLUE");
    await input.trigger("keydown.enter");
    expect(wrapper.get("[data-test='file-under-tag-error']").text()).toContain("already on this");
    expect(wrapper.emitted("update:state")).toBeUndefined();
  });

  it("refuses a duplicate of the ghost chip", async () => {
    const wrapper = mountGroup();
    const input = wrapper.get("[data-test='file-under-tag-input']");
    await input.setValue("smurf-village");
    await input.trigger("keydown.enter");
    expect(wrapper.find("[data-test='file-under-tag-error']").exists()).toBe(true);
    expect(wrapper.emitted("update:state")).toBeUndefined();
  });

  it("refuses an over-long tag", async () => {
    const wrapper = mountGroup();
    const input = wrapper.get("[data-test='file-under-tag-input']");
    await input.setValue("x".repeat(65));
    await input.trigger("keydown.enter");
    expect(wrapper.get("[data-test='file-under-tag-error']").text()).toContain("64 characters");
  });

  it("ignores an empty Enter", async () => {
    const wrapper = mountGroup();
    await wrapper.get("[data-test='file-under-tag-input']").trigger("keydown.enter");
    expect(wrapper.emitted("update:state")).toBeUndefined();
    expect(wrapper.find("[data-test='file-under-tag-error']").exists()).toBe(false);
  });

  it("removes a manual chip", async () => {
    const wrapper = mountGroup({ state: addTag(emptyFileUnderState(), "blue") });
    await wrapper.get("[data-test='file-under-tag-remove']").trigger("click");
    expect(emitted(wrapper).manualTags).toEqual([]);
  });
});

describe("FileUnderGroup — collection", () => {
  it("pre-selects the collection whose slug matches the title, and says so", () => {
    const wrapper = mountGroup();
    expect(wrapper.get("[data-test='file-under-collection']").text()).toContain("Smurf Village");
    expect(wrapper.get("[data-test='file-under-collection-match']").text()).toContain(
      "matched to title",
    );
  });

  it("reads None when the title matches nothing", () => {
    const wrapper = mountGroup({ title: "Something else" });
    expect(wrapper.get("[data-test='file-under-collection']").text()).toContain("None");
    expect(wrapper.find("[data-test='file-under-collection-match']").exists()).toBe(false);
  });

  it("clearing the match sticks for that title slug", async () => {
    const wrapper = mountGroup();
    await wrapper.get("[data-test='file-under-collection-clear']").trigger("click");
    const state = emitted(wrapper);
    expect(state.clearedMatchSlug).toBe("smurf-village");
    await wrapper.setProps({ state });
    expect(wrapper.get("[data-test='file-under-collection']").text()).toContain("None");
  });

  it("lists None, every collection with its count, and New collection…", async () => {
    const wrapper = mountGroup();
    await wrapper.get("[data-test='file-under-collection']").trigger("click");
    const menu = wrapper.get("[data-test='file-under-collection-menu']");
    expect(menu.find("[data-test='file-under-collection-none']").exists()).toBe(true);
    const options = menu.findAll("[data-test='file-under-collection-option']");
    expect(options.map((option) => option.text())).toEqual(["✓Smurf Village12", "River studies8"]);
    expect(menu.find("[data-test='file-under-new-collection']").exists()).toBe(true);
  });

  it("picking a row outranks the title match", async () => {
    const wrapper = mountGroup();
    await wrapper.get("[data-test='file-under-collection']").trigger("click");
    await wrapper.findAll("[data-test='file-under-collection-option']")[1]!.trigger("click");
    const state = emitted(wrapper);
    expect(state.pickedExplicitly).toBe(true);
    expect(state.picked).toEqual({ id: "c2", name: "River studies" });
    expect(wrapper.find("[data-test='file-under-collection-menu']").exists()).toBe(false);
  });

  it("creates nothing up front — a new name is only a pick", async () => {
    const wrapper = mountGroup({ title: "Untitled thing" });
    await wrapper.get("[data-test='file-under-collection']").trigger("click");
    await wrapper.get("[data-test='file-under-new-collection']").trigger("click");
    const input = wrapper.get("[data-test='file-under-new-collection-input']");
    await input.setValue("Client · Halcyon");
    await input.trigger("keydown.enter");
    expect(emitted(wrapper).picked).toEqual({ name: "Client · Halcyon" });
  });

  it("a new name that slugs onto an existing collection picks that one", async () => {
    const wrapper = mountGroup({ title: "Untitled thing" });
    await wrapper.get("[data-test='file-under-collection']").trigger("click");
    await wrapper.get("[data-test='file-under-new-collection']").trigger("click");
    const input = wrapper.get("[data-test='file-under-new-collection-input']");
    await input.setValue("river studies");
    await input.trigger("keydown.enter");
    expect(emitted(wrapper).picked).toEqual({ id: "c2", name: "River studies" });
  });

  it("None clears an explicit pick", async () => {
    const wrapper = mountGroup({
      state: pickCollection(emptyFileUnderState(), { name: "River studies" }),
    });
    await wrapper.get("[data-test='file-under-collection']").trigger("click");
    await wrapper.get("[data-test='file-under-collection-none']").trigger("click");
    expect(emitted(wrapper).pickedExplicitly).toBe(false);
  });

  it("Escape closes the popovers", async () => {
    const wrapper = mountGroup();
    await wrapper.get("[data-test='file-under-collection']").trigger("click");
    expect(wrapper.find("[data-test='file-under-collection-menu']").exists()).toBe(true);
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await wrapper.vm.$nextTick();
    expect(wrapper.find("[data-test='file-under-collection-menu']").exists()).toBe(false);
    wrapper.unmount();
  });
});

describe("FileUnderGroup — filename preview", () => {
  it("previews the gallery grammar with the title slug", () => {
    const wrapper = mountGroup();
    const line = wrapper.get("[data-test='file-under-filename']").text();
    expect(line).toContain("files as");
    expect(line).toMatch(/mold-z-image-turbo-bf16-\d+~smurf-village\.png/);
  });

  it("omits the ~slug for an untitled print", () => {
    const wrapper = mountGroup({ title: "" });
    const line = wrapper.get("[data-test='file-under-filename']").text();
    expect(line).not.toContain("~");
    expect(line).toMatch(/mold-z-image-turbo-bf16-\d+\.png/);
  });

  it("follows the live extension", () => {
    const wrapper = mountGroup({ title: "", model: "ltx2", extension: "mp4" });
    expect(wrapper.get("[data-test='file-under-filename']").text()).toMatch(/mold-ltx2-\d+\.mp4/);
  });

  it("switches to the chain grammar for a sequence's stitched print", () => {
    const wrapper = mountGroup({ model: "ltx2", extension: "mp4", outputKind: "sequence" });
    expect(wrapper.get("[data-test='file-under-filename']").text()).toContain(
      "mold-chain-…-take-0~smurf-village.mp4",
    );
  });
});
