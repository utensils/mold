import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ReferenceImageStrip from "./ReferenceImageStrip.vue";

const PNG = "iVBORw0KGgo=";
const WEBP = "UklGRgAAAABXRUJQ";

function images(count: number) {
  return Array.from({ length: count }, (_, index) => ({
    data: index === 1 ? WEBP : PNG,
    filename: `ref-${index + 1}.png`,
  }));
}

function strip(props: Record<string, unknown> = {}) {
  return mount(ReferenceImageStrip, {
    props: { images: images(3), ...props },
    attachTo: document.body,
  });
}

describe("ReferenceImageStrip", () => {
  it("draws every reference as its own numbered thumbnail, in order", () => {
    const wrapper = strip();
    const tiles = wrapper.findAll("[data-test^='reference-tile-']");
    expect(tiles).toHaveLength(3);
    expect(
      [0, 1, 2].map((i) =>
        wrapper.get(`[data-test='reference-label-${i}']`).text(),
      ),
    ).toEqual(["Image 1", "Image 2", "Image 3"]);
    expect(wrapper.get("[data-test='reference-ordinal-2']").text()).toBe("3");
    // The filename rides under the ordinal so a user can tell which is which.
    expect(wrapper.get("[data-test='reference-tile-0']").text()).toContain(
      "ref-1.png",
    );
    // One picture each, never a "name +N more" summary.
    expect(wrapper.text()).not.toMatch(/\+\d+ more/);
  });

  it("puts every thumbnail on the alpha checkerboard, with its own container type", () => {
    const wrapper = strip();
    const thumbs = wrapper.findAll("[data-test^='reference-thumb-']");
    expect(thumbs).toHaveLength(3);
    for (const thumb of thumbs)
      expect(thumb.classes()).toContain("ms-alpha-bed");
    expect(thumbs[1]!.attributes("src")).toMatch(/^data:image\/webp;base64,/);
    expect(thumbs[0]!.attributes("src")).toMatch(/^data:image\/png;base64,/);
  });

  it("names the Target on a target-first recipe", () => {
    const wrapper = strip({ firstIsTarget: true });
    expect(wrapper.get("[data-test='reference-role-0']").text()).toBe("Target");
    expect(wrapper.get("[data-test='reference-role-1']").text()).toBe(
      "Reference",
    );
  });

  it("marks the last reference as the one that sets the canvas, with a sentence", () => {
    const wrapper = strip({ setsCanvas: true });
    const badges = wrapper.findAll("[data-test='reference-sets-canvas']");
    expect(badges).toHaveLength(1);
    expect(
      wrapper
        .get("[data-test='reference-tile-2']")
        .find("[data-test='reference-sets-canvas']")
        .exists(),
    ).toBe(true);
    expect(
      wrapper.get("[data-test='reference-tile-2']").attributes(),
    ).toHaveProperty("data-sets-canvas");
    expect(wrapper.get("[data-test='reference-canvas-note']").text()).toMatch(
      /last image sets the canvas/i,
    );
  });

  it("shows no canvas mark where the recipe has no canvas rule", () => {
    const wrapper = strip();
    expect(wrapper.find("[data-test='reference-sets-canvas']").exists()).toBe(
      false,
    );
    expect(wrapper.find("[data-test='reference-canvas-note']").exists()).toBe(
      false,
    );
  });

  it("offsets the numbering past a source that ships first", () => {
    const wrapper = strip({ images: images(1), ordinalBase: 1 });
    expect(wrapper.get("[data-test='reference-label-0']").text()).toBe(
      "Image 2",
    );
  });

  it("removes one picture by index", async () => {
    const wrapper = strip();
    await wrapper.get("[data-test='reference-remove-1']").trigger("click");
    expect(wrapper.emitted("remove")).toEqual([[1]]);
  });

  it("reorders with labelled earlier/later buttons that stop at the ends", async () => {
    const wrapper = strip();
    const earlier0 = wrapper.get("[data-test='reference-earlier-0']");
    const later2 = wrapper.get("[data-test='reference-later-2']");
    expect(earlier0.attributes("disabled")).toBeDefined();
    expect(later2.attributes("disabled")).toBeDefined();
    expect(
      wrapper.get("[data-test='reference-earlier-1']").attributes("aria-label"),
    ).toBe("Move image 2 earlier");
    await wrapper.get("[data-test='reference-earlier-1']").trigger("click");
    await wrapper.get("[data-test='reference-later-1']").trigger("click");
    expect(wrapper.emitted("move")).toEqual([
      [1, 0],
      [1, 2],
    ]);
  });

  it("keeps keyboard focus on the moved picture's button", async () => {
    const wrapper = strip();
    const button = wrapper.get("[data-test='reference-later-0']");
    (button.element as HTMLButtonElement).focus();
    await button.trigger("click");
    // The parent applies the move and hands back the new order.
    const [a, b, c] = images(3);
    await wrapper.setProps({ images: [b, a, c] });
    await wrapper.vm.$nextTick();
    expect(document.activeElement).toBe(
      wrapper.get("[data-test='reference-later-1']").element,
    );
    wrapper.unmount();
  });

  it("reorders by dragging one tile onto another", async () => {
    const wrapper = strip();
    const store = new Map<string, string>();
    const dataTransfer = {
      setData: (type: string, value: string) => store.set(type, value),
      getData: (type: string) => store.get(type) ?? "",
      types: [] as string[],
      files: [] as File[],
      effectAllowed: "",
    };
    await wrapper
      .get("[data-test='reference-tile-2']")
      .trigger("dragstart", { dataTransfer });
    dataTransfer.types = [...store.keys()];
    await wrapper
      .get("[data-test='reference-tile-0']")
      .trigger("drop", { dataTransfer });
    expect(wrapper.emitted("move")).toEqual([[2, 0]]);
    // An internal reorder is never mistaken for a file drop.
    expect(wrapper.emitted("files")).toBeUndefined();
  });

  it("is the references drop target and hands dropped files to the surface", async () => {
    const wrapper = strip();
    const root = wrapper.get("[data-test='reference-strip']");
    expect(root.attributes("data-drop-target")).toBe("references");
    const file = new File(["x"], "new.png", { type: "image/png" });
    const event = new Event("drop", { bubbles: true, cancelable: true });
    Object.defineProperty(event, "dataTransfer", {
      value: { files: [file], types: ["Files"], getData: () => "" },
    });
    root.element.dispatchEvent(event);
    // Handled, so a window-level drop listener leaves it alone.
    expect(event.defaultPrevented).toBe(true);
    expect(wrapper.emitted("files")).toEqual([[[file]]]);
  });

  it("offers an add tile until the advertised ceiling is reached", async () => {
    const wrapper = strip({ max: 4 });
    const add = wrapper.get("[data-test='reference-add']");
    expect(add.attributes("disabled")).toBeUndefined();
    await add.trigger("click");
    expect(wrapper.emitted("add")).toHaveLength(1);
    await wrapper.setProps({ images: images(4) });
    expect(
      wrapper.get("[data-test='reference-add']").attributes("disabled"),
    ).toBeDefined();
    expect(wrapper.get("[data-test='reference-add']").text()).toContain("Full");
  });

  it("becomes one full-width drop zone with the surface's wording while empty", () => {
    const wrapper = strip({
      images: [],
      emptyLabel: "Attach references",
      required: true,
    });
    const add = wrapper.get("[data-test='reference-add']");
    expect(add.text()).toContain("Attach references");
    expect(add.attributes("aria-required")).toBe("true");
    expect(wrapper.findAll("[data-test^='reference-tile-']")).toHaveLength(0);
    expect(
      wrapper
        .get("[data-test='reference-strip']")
        .attributes("data-drop-target"),
    ).toBe("references");
  });

  it("uses the surface's own strip test id and tile prefix", () => {
    const wrapper = strip({
      stripTestId: "attachment-strip",
      testIdPrefix: "m-",
    });
    expect(wrapper.find("[data-test='attachment-strip']").exists()).toBe(true);
    expect(wrapper.find("[data-test='m-reference-tile-0']").exists()).toBe(
      true,
    );
  });

  it("keeps every control 44px tall and drops dragging on touch", () => {
    const wrapper = strip({ touchFriendly: true });
    expect(wrapper.get("[data-test='reference-strip']").classes()).toContain(
      "ris--touch",
    );
    expect(
      wrapper.get("[data-test='reference-tile-0']").attributes("draggable"),
    ).toBe("false");
  });
});
