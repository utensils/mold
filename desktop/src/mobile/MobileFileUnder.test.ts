import { readFileSync } from "node:fs";
import { mount, type VueWrapper } from "@vue/test-utils";
import { afterEach, describe, expect, it } from "vitest";
import { emptyFileUnderState, type FileUnderState } from "@studio/lib/fileUnder";
import MobileFileUnder from "./MobileFileUnder.vue";

const component = readFileSync("src/mobile/MobileFileUnder.vue", "utf8");

let wrapper: VueWrapper | null = null;
/** The owner's copy of the draft: every reducer emits a NEW state, so this is
 * what `MobileApp` would be holding in `form.fileUnder`. */
let current: FileUnderState = emptyFileUnderState();

afterEach(() => {
  wrapper?.unmount();
  wrapper = null;
});

const collections = [
  { name: "Smurfs", slug: "smurfs", count: 12 },
  { name: "River studies", slug: "river-studies", count: 8 },
];

const tags = [
  { name: "blue", count: 9 },
  { name: "#kodak", count: 4 },
];

/**
 * The group is prop-driven — every reducer emits a NEW state — so the harness
 * plays the owner and feeds each emission straight back, exactly as
 * `MobileApp` does with `form.fileUnder`.
 */
function mountGroup(
  props: Partial<{
    title: string;
    state: FileUnderState;
    autoTagTitle: boolean;
    tags: typeof tags;
    collections: typeof collections;
    model: string;
    extension: string;
    batchSize: number;
    outputKind: "print" | "sequence";
  }> = {},
): VueWrapper {
  current = props.state ?? emptyFileUnderState();
  wrapper = mount(MobileFileUnder, {
    attachTo: document.body,
    props: {
      title: "Smurfs",
      autoTagTitle: true,
      tags,
      collections,
      model: "z-image-turbo:bf16",
      extension: "png",
      ...props,
      state: current,
      "onUpdate:state": (next: FileUnderState) => {
        current = next;
        void wrapper?.setProps({ state: next });
      },
    },
  });
  return wrapper;
}

function state(): FileUnderState {
  return current;
}

async function openTagSheet(): Promise<void> {
  await wrapper!.get("[data-test='mobile-file-under-add-tag']").trigger("click");
}

async function typeTag(text: string): Promise<void> {
  await openTagSheet();
  await wrapper!.get("[data-test='mobile-file-under-tag-input']").setValue(text);
  await wrapper!.get("[data-test='mobile-file-under-tag-add']").trigger("click");
}

describe("MobileFileUnder tags row", () => {
  it("renders the ghost chip derived from the title, marked as its source", () => {
    mountGroup();

    const ghost = wrapper!.get("[data-test='mobile-file-under-ghost']");
    expect(ghost.text()).toContain("smurfs");
    expect(ghost.text()).toContain("from title");
  });

  it("renders no ghost chip for an untitled print or with auto-tagging off", () => {
    mountGroup({ title: "" });
    expect(wrapper!.find("[data-test='mobile-file-under-ghost']").exists()).toBe(false);

    wrapper!.unmount();
    mountGroup({ autoTagTitle: false });
    expect(wrapper!.find("[data-test='mobile-file-under-ghost']").exists()).toBe(false);
  });

  it("retires the ghost chip when it is removed so the live title cannot re-offer it", async () => {
    mountGroup();

    await wrapper!.get("[data-test='mobile-file-under-ghost-remove']").trigger("click");

    expect(state().ghostRemoved).toBe(true);
    expect(wrapper!.find("[data-test='mobile-file-under-ghost']").exists()).toBe(false);
    // The title still slugs to "smurfs"; the opt-out has to survive that.
    await wrapper!.setProps({ title: "Smurfs " });
    expect(wrapper!.find("[data-test='mobile-file-under-ghost']").exists()).toBe(false);
  });

  it("strips a leading hash from a tag typed into the sheet", async () => {
    mountGroup();

    await typeTag("#kodak");

    expect(state().manualTags).toEqual(["kodak"]);
    expect(wrapper!.get("[data-test='mobile-file-under-tag']").text()).toContain("kodak");
  });

  it("adds a suggestion picked in the same sheet verbatim, hash included", async () => {
    mountGroup();
    await openTagSheet();

    const suggestion = wrapper!
      .findAll("[data-test='mobile-file-under-tag-suggestion']")
      .find((candidate) => candidate.text().includes("#kodak"));
    await suggestion!.trigger("click");

    // The host really does hold a tag called "#kodak"; stripping it here
    // would file a DIFFERENT tag than the one that was picked.
    expect(state().manualTags).toEqual(["#kodak"]);
  });

  it("refuses a duplicate of the ghost chip with an inline message", async () => {
    mountGroup();

    await typeTag("Smurfs");

    expect(state().manualTags).toEqual([]);
    expect(wrapper!.get("[data-test='mobile-file-under-tag-error']").text()).toBe(
      "That tag is already on this print.",
    );
  });

  it("removes a real chip from the row without retiring the ghost", async () => {
    mountGroup();
    await typeTag("blue");

    await wrapper!.get("[data-test='mobile-file-under-tag-remove']").trigger("click");

    expect(state().manualTags).toEqual([]);
    expect(state().ghostRemoved).toBe(false);
    expect(wrapper!.find("[data-test='mobile-file-under-ghost']").exists()).toBe(true);
  });
});

describe("MobileFileUnder collection row", () => {
  it("pre-selects the collection whose slug matches the title and says so", () => {
    mountGroup();

    const row = wrapper!.get("[data-test='mobile-file-under-collection']");
    expect(row.text()).toContain("Smurfs");
    expect(wrapper!.get("[data-test='mobile-file-under-collection-match']").text()).toContain(
      "matched to title",
    );
  });

  it("shows None when nothing matches the title", () => {
    mountGroup({ title: "Harbour" });

    expect(wrapper!.get("[data-test='mobile-file-under-collection']").text()).toContain("None");
    expect(wrapper!.find("[data-test='mobile-file-under-collection-clear']").exists()).toBe(false);
  });

  it("keeps a cleared match cleared while the title still slugs the same", async () => {
    mountGroup();

    await wrapper!.get("[data-test='mobile-file-under-collection-clear']").trigger("click");

    expect(state().clearedMatchSlug).toBe("smurfs");
    expect(wrapper!.get("[data-test='mobile-file-under-collection']").text()).toContain("None");
    await wrapper!.setProps({ title: "  Smurfs  " });
    expect(wrapper!.get("[data-test='mobile-file-under-collection']").text()).toContain("None");
    // A genuinely different slug is a new offer.
    await wrapper!.setProps({ title: "River studies" });
    expect(wrapper!.get("[data-test='mobile-file-under-collection']").text()).toContain(
      "River studies",
    );
  });

  it("picks an existing collection from the sheet, outranking the title match", async () => {
    mountGroup();
    await wrapper!.get("[data-test='mobile-file-under-collection']").trigger("click");

    const option = wrapper!
      .findAll("[data-test='mobile-file-under-collection-option']")
      .find((candidate) => candidate.text().includes("River studies"));
    await option!.trigger("click");

    expect(state().picked).toEqual({ name: "River studies" });
    expect(state().pickedExplicitly).toBe(true);
    expect(wrapper!.find("[data-test='mobile-file-under-collection-match']").exists()).toBe(false);
  });

  it("records a brand-new collection by name without creating anything", async () => {
    mountGroup({ title: "Harbour" });
    await wrapper!.get("[data-test='mobile-file-under-collection']").trigger("click");
    await wrapper!.get("[data-test='mobile-file-under-new-collection']").trigger("click");
    await wrapper!
      .get("[data-test='mobile-file-under-new-collection-input']")
      .setValue("Night ferries");
    await wrapper!.get("[data-test='mobile-file-under-new-collection-create']").trigger("click");

    expect(state().picked).toEqual({ name: "Night ferries" });
    expect(wrapper!.get("[data-test='mobile-file-under-collection']").text()).toContain(
      "Night ferries",
    );
  });

  it("chooses None from the sheet", async () => {
    mountGroup();
    await wrapper!.get("[data-test='mobile-file-under-collection']").trigger("click");
    await wrapper!.get("[data-test='mobile-file-under-collection-none']").trigger("click");

    expect(wrapper!.get("[data-test='mobile-file-under-collection']").text()).toContain("None");
  });
});

describe("MobileFileUnder filename preview", () => {
  it("previews the creation-time gallery name with the title slug", () => {
    mountGroup();

    const preview = wrapper!.get("[data-test='mobile-file-under-filename']").text();
    expect(preview).toContain("files as");
    expect(preview).toMatch(/mold-z-image-turbo-bf16-\d+~smurfs\.png/);
  });

  it("drops the slug segment for an untitled print", () => {
    mountGroup({ title: "" });

    expect(wrapper!.get("[data-test='mobile-file-under-filename']").text()).not.toContain("~");
  });

  it("previews the chain grammar for a sequence's stitched print", () => {
    mountGroup({ outputKind: "sequence" });

    expect(wrapper!.get("[data-test='mobile-file-under-filename']").text()).toContain(
      "mold-chain-…-take-0~smurfs.mp4",
    );
  });
});

describe("MobileFileUnder iPhone interaction invariants", () => {
  it("keeps every row and control at 44pt and every editable input at 16px", () => {
    // The rows and the sheet controls are the whole surface of this group;
    // a sub-44pt chip remove or a 15px input is an iPhone regression.
    for (const rule of [
      /\.mobile-file-under-row\s*\{[^}]*min-height:\s*44px/s,
      /\.mobile-file-under-add\s*\{[^}]*min-height:\s*44px/s,
      /\.mobile-file-under-clear\s*\{[^}]*min-height:\s*44px/s,
      /\.mobile-file-under-clear\s*\{[^}]*min-width:\s*44px/s,
      /\.mobile-file-under-chip\s+button\s*\{[^}]*min-height:\s*44px/s,
      /\.mobile-file-under-new\s*\{[^}]*min-height:\s*44px/s,
      /\.mobile-file-under\s+input\s*\{[^}]*font-size:\s*16px/s,
    ]) {
      expect(component).toMatch(rule);
    }
  });
});
