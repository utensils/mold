import { mount, type VueWrapper } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import {
  emptyFileUnderState,
  type FileUnderState,
} from "@studio/lib/fileUnder";
import FileUnderGroup from "./FileUnderGroup.vue";
import type { TagCount } from "../../types";

const collections = [
  { slug: "smurfs", name: "Smurfs", count: 12 },
  { slug: "river-studies", name: "River studies", count: 8 },
];

const suggestions: TagCount[] = [
  { name: "blue", count: 9 },
  { name: "#kodak", count: 4 },
  { name: "dusk", count: 2 },
];

function mountGroup(props: Partial<Record<string, unknown>> = {}) {
  return mount(FileUnderGroup, {
    props: {
      state: emptyFileUnderState(),
      title: "Smurfs",
      autoTag: true,
      suggestions,
      collections,
      model: "z-image-turbo:bf16",
      ext: "png",
      timestamp: 1787320481000,
      ...props,
    },
  });
}

/** Last `update:state` payload the group emitted. */
function lastState(wrapper: VueWrapper): FileUnderState {
  const events = wrapper.emitted("update:state");
  expect(events).toBeTruthy();
  return events![events!.length - 1]![0] as FileUnderState;
}

describe("FileUnderGroup tags row", () => {
  it("renders the ghost chip derived from the title, marked as auto", () => {
    const wrapper = mountGroup();
    const ghost = wrapper.get("[data-test='file-under-ghost']");
    expect(ghost.text()).toContain("smurfs");
    expect(ghost.text()).toContain("from title");
  });

  it("renders no ghost chip for an untitled print", () => {
    const wrapper = mountGroup({ title: "" });
    expect(wrapper.find("[data-test='file-under-ghost']").exists()).toBe(false);
  });

  it("renders no ghost chip when auto-tagging is off", () => {
    const wrapper = mountGroup({ autoTag: false });
    expect(wrapper.find("[data-test='file-under-ghost']").exists()).toBe(false);
  });

  it("retires the ghost chip when it is removed and offers a restore", async () => {
    const wrapper = mountGroup();
    await wrapper.get("[data-test='file-under-ghost-remove']").trigger("click");
    expect(lastState(wrapper).ghostRemoved).toBe(true);

    await wrapper.setProps({ state: lastState(wrapper) });
    expect(wrapper.find("[data-test='file-under-ghost']").exists()).toBe(false);
    await wrapper
      .get("[data-test='file-under-ghost-restore']")
      .trigger("click");
    expect(lastState(wrapper).ghostRemoved).toBe(false);
  });

  it("adds a typed tag on Enter, stripping a leading hash", async () => {
    const wrapper = mountGroup();
    const input = wrapper.get("[data-test='file-under-tag-input']");
    await input.setValue("#kodak");
    await input.trigger("keydown", { key: "Enter" });
    expect(lastState(wrapper).manualTags).toEqual(["kodak"]);
  });

  it("adds a suggested tag verbatim, hash included", async () => {
    const wrapper = mountGroup();
    await wrapper.get("[data-test='file-under-tag-input']").trigger("focus");
    // The suggestion popover teleports to <body> so the rail cannot clip it.
    const hashed = [
      ...document.querySelectorAll<HTMLElement>(
        "[data-test='file-under-suggestion']",
      ),
    ].find((button) => button.textContent?.includes("#kodak"));
    expect(hashed).toBeTruthy();
    hashed!.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
    await wrapper.vm.$nextTick();
    expect(lastState(wrapper).manualTags).toEqual(["#kodak"]);
  });

  it("shows the suggestion footer so a new name reads as deferred", async () => {
    const wrapper = mountGroup();
    await wrapper.get("[data-test='file-under-tag-input']").trigger("focus");
    const foot = document.querySelector<HTMLElement>(
      "[data-test='file-under-suggest-foot']",
    );
    expect(foot?.textContent?.trim()).toBe(
      "↵ adds · new names are created on develop",
    );
  });

  it("keeps the footer up for a brand-new name with no matches", async () => {
    const wrapper = mountGroup();
    const input = wrapper.get("[data-test='file-under-tag-input']");
    await input.trigger("focus");
    await input.setValue("nothing-like-this");
    expect(
      document.querySelector("[data-test='file-under-suggest-empty']")
        ?.textContent,
    ).toContain("No matching tag yet.");
    expect(
      document.querySelector("[data-test='file-under-suggest-foot']"),
    ).toBeTruthy();
  });

  it("refuses a duplicate of the ghost chip with an inline message", async () => {
    const wrapper = mountGroup();
    const input = wrapper.get("[data-test='file-under-tag-input']");
    await input.setValue("Smurfs");
    await input.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("update:state")).toBeUndefined();
    expect(wrapper.get("[data-test='file-under-tag-error']").text()).toContain(
      "already on this print",
    );
  });

  it("removes a real chip without retiring the ghost", async () => {
    const state: FileUnderState = {
      ...emptyFileUnderState(),
      manualTags: ["blue"],
    };
    const wrapper = mountGroup({ state });
    const remove = wrapper
      .findAll("[data-test='file-under-tag-remove']")
      .at(-1)!;
    await remove.trigger("click");
    expect(lastState(wrapper)).toMatchObject({
      ghostRemoved: false,
      manualTags: [],
    });
  });
});

describe("FileUnderGroup collection row", () => {
  it("pre-selects the collection whose slug matches the title", () => {
    const wrapper = mountGroup();
    const field = wrapper.get("[data-test='file-under-collection']");
    expect(field.text()).toContain("Smurfs");
    expect(field.text()).toContain("matched to title");
  });

  it("shows None when nothing matches and nothing was picked", () => {
    const wrapper = mountGroup({ title: "Riverbank at dusk" });
    expect(wrapper.get("[data-test='file-under-collection']").text()).toContain(
      "None",
    );
  });

  it("keeps a cleared match cleared while the title still slugs the same", async () => {
    const wrapper = mountGroup();
    await wrapper
      .get("[data-test='file-under-collection-clear']")
      .trigger("click");
    const cleared = lastState(wrapper);
    expect(cleared.clearedMatchSlug).toBe("smurfs");
    await wrapper.setProps({ state: cleared, title: "smurfs " });
    expect(wrapper.get("[data-test='file-under-collection']").text()).toContain(
      "None",
    );
  });

  it("re-offers the match once the title slugs to something else", async () => {
    const wrapper = mountGroup();
    await wrapper
      .get("[data-test='file-under-collection-clear']")
      .trigger("click");
    await wrapper.setProps({
      state: lastState(wrapper),
      title: "River studies",
    });
    expect(wrapper.get("[data-test='file-under-collection']").text()).toContain(
      "River studies",
    );
  });

  it("picks an existing collection by name from the popover", async () => {
    const wrapper = mountGroup();
    await wrapper
      .get("[data-test='file-under-collection-open']")
      .trigger("click");
    const rows = document.querySelectorAll<HTMLElement>(
      "[data-test='file-under-collection-option']",
    );
    const river = [...rows].find((row) => row.textContent?.includes("River"))!;
    river.click();
    await wrapper.vm.$nextTick();
    expect(lastState(wrapper)).toMatchObject({
      pickedExplicitly: true,
      picked: { name: "River studies" },
    });
  });

  it("creates a pick by name from the inline New collection input", async () => {
    const wrapper = mountGroup();
    await wrapper
      .get("[data-test='file-under-collection-open']")
      .trigger("click");
    document
      .querySelector<HTMLElement>("[data-test='file-under-collection-new']")!
      .click();
    await wrapper.vm.$nextTick();
    const input = document.querySelector<HTMLInputElement>(
      "[data-test='file-under-collection-new-input']",
    )!;
    input.value = "Client · Halcyon";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" }));
    await wrapper.vm.$nextTick();
    expect(lastState(wrapper).picked).toEqual({ name: "Client · Halcyon" });
  });
});

describe("FileUnderGroup filename preview", () => {
  it("previews the creation-time name with the title slug", () => {
    const wrapper = mountGroup();
    expect(wrapper.get("[data-test='file-under-preview']").text()).toContain(
      "mold-z-image-turbo-bf16-1787320481000~smurfs.png",
    );
  });

  it("drops the ~slug for an untitled print", () => {
    const wrapper = mountGroup({ title: "" });
    expect(wrapper.get("[data-test='file-under-preview']").text()).toContain(
      "mold-z-image-turbo-bf16-1787320481000.png",
    );
  });
});
