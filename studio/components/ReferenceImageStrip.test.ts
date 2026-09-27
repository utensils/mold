import { readFileSync } from "node:fs";
import { join } from "node:path";
import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ReferenceImageStrip from "./ReferenceImageStrip.vue";

/** The touch strip's control height is a layout constant the phone relies
 * on (44pt, the iPhone interaction invariant), so it is read from the SFC. */
describe("ReferenceImageStrip touch sizing", () => {
  it("gives every touch control at least 44px", () => {
    const source = readFileSync(
      join(import.meta.dirname, "ReferenceImageStrip.vue"),
      "utf8",
    );
    const touch = source.match(/\.ris--touch\s*\{([^}]*)\}/)?.[1] ?? "";
    const control = Number(touch.match(/--ris-control:\s*(\d+)px/)?.[1]);
    expect(control).toBeGreaterThanOrEqual(44);
    // Every button reads the variable, so the constant is the whole story.
    const action = source.match(/\.ris__action\s*\{([^}]*)\}/)?.[1] ?? "";
    expect(action).toMatch(/min-height:\s*var\(--ris-control\)/);
  });
});

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

/** Three 100x80 tiles in one row, 10px apart (a real browser's layout;
 * happy-dom lays nothing out). */
function layOut(wrapper: ReturnType<typeof strip>) {
  wrapper.findAll("[data-test^='reference-tile-']").forEach((tile, index) => {
    const left = index * 110;
    tile.element.getBoundingClientRect = () =>
      ({
        left,
        top: 0,
        right: left + 100,
        bottom: 80,
        width: 100,
        height: 80,
        x: left,
        y: 0,
        toJSON: () => ({}),
      }) as DOMRect;
  });
}

function pointer(
  target: Element,
  type: string,
  clientX: number,
  clientY: number,
  pointerType = "mouse",
) {
  target.dispatchEvent(
    new PointerEvent(type, {
      bubbles: true,
      cancelable: true,
      clientX,
      clientY,
      button: 0,
      pointerId: 1,
      pointerType,
    }),
  );
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
    await wrapper.setProps({ images: [b!, a!, c!] });
    await wrapper.vm.$nextTick();
    expect(document.activeElement).toBe(
      wrapper.get("[data-test='reference-later-1']").element,
    );
    wrapper.unmount();
  });

  it("moves keyboard focus to the next tile's remove button after Remove", async () => {
    const wrapper = strip();
    const button = wrapper.get("[data-test='reference-remove-1']");
    (button.element as HTMLButtonElement).focus();
    await button.trigger("click");
    // The parent applies the removal and hands back the new order — index 1
    // now holds what used to be image 3.
    const [a, , c] = images(3);
    await wrapper.setProps({ images: [a!, c!] });
    await wrapper.vm.$nextTick();
    expect(document.activeElement).toBe(
      wrapper.get("[data-test='reference-remove-1']").element,
    );
    wrapper.unmount();
  });

  it("moves keyboard focus to the previous tile's remove button when the last picture is removed", async () => {
    const wrapper = strip();
    const button = wrapper.get("[data-test='reference-remove-2']");
    (button.element as HTMLButtonElement).focus();
    await button.trigger("click");
    const [a, b] = images(3);
    await wrapper.setProps({ images: [a!, b!] });
    await wrapper.vm.$nextTick();
    expect(document.activeElement).toBe(
      wrapper.get("[data-test='reference-remove-1']").element,
    );
    wrapper.unmount();
  });

  it("moves keyboard focus to the add tile when the only picture is removed", async () => {
    const wrapper = strip({ images: images(1) });
    const button = wrapper.get("[data-test='reference-remove-0']");
    (button.element as HTMLButtonElement).focus();
    await button.trigger("click");
    await wrapper.setProps({ images: [] });
    await wrapper.vm.$nextTick();
    expect(document.activeElement).toBe(
      wrapper.get("[data-test='reference-add']").element,
    );
    wrapper.unmount();
  });

  it("reorders by dragging one tile onto another with the pointer", async () => {
    const wrapper = strip();
    layOut(wrapper);
    const from = wrapper.get("[data-test='reference-tile-2']").element;
    pointer(from, "pointerdown", 250, 40);
    pointer(from, "pointermove", 200, 40);
    pointer(from, "pointermove", 30, 40);
    await wrapper.vm.$nextTick();
    // The target shows where the picture will land; the source is lifted.
    const target = wrapper.get("[data-test='reference-tile-0']");
    expect(target.classes()).toContain("ris__tile--over");
    expect(target.classes()).toContain("ris__tile--over-before");
    expect(wrapper.get("[data-test='reference-tile-2']").classes()).toContain(
      "ris__tile--lifted",
    );
    pointer(from, "pointerup", 30, 40);
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("move")).toEqual([[2, 0]]);
    // An internal reorder is never mistaken for a file drop.
    expect(wrapper.emitted("files")).toBeUndefined();
    expect(target.classes()).not.toContain("ris__tile--over");
    // Announced by the numbers the prompt uses.
    expect(wrapper.get("[data-test='reference-announce']").text()).toBe(
      "Moved image 3 to position 1.",
    );
    wrapper.unmount();
  });

  it("never reorders on a click without movement, and a button click still works", async () => {
    const wrapper = strip();
    layOut(wrapper);
    const tile = wrapper.get("[data-test='reference-tile-1']").element;
    pointer(tile, "pointerdown", 150, 40);
    pointer(tile, "pointermove", 152, 41);
    pointer(tile, "pointerup", 152, 41);
    expect(wrapper.emitted("move")).toBeUndefined();

    const later = wrapper.get("[data-test='reference-later-1']");
    pointer(later.element, "pointerdown", 150, 70);
    pointer(later.element, "pointermove", 260, 70);
    pointer(later.element, "pointerup", 260, 70);
    // A press that starts on a button is that button's, never a drag.
    expect(wrapper.emitted("move")).toBeUndefined();
    await later.trigger("click");
    expect(wrapper.emitted("move")).toEqual([[1, 2]]);
    wrapper.unmount();
  });

  it("aborts a drag on Escape and on pointercancel", async () => {
    const wrapper = strip();
    layOut(wrapper);
    const from = wrapper.get("[data-test='reference-tile-0']").element;
    pointer(from, "pointerdown", 50, 40);
    pointer(from, "pointermove", 260, 40);
    await wrapper.vm.$nextTick();
    expect(wrapper.get("[data-test='reference-tile-2']").classes()).toContain(
      "ris__tile--over",
    );
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await wrapper.vm.$nextTick();
    expect(
      wrapper.get("[data-test='reference-tile-2']").classes(),
    ).not.toContain("ris__tile--over");
    pointer(from, "pointerup", 260, 40);

    pointer(from, "pointerdown", 50, 40);
    pointer(from, "pointermove", 260, 40);
    pointer(from, "pointercancel", 260, 40);
    pointer(from, "pointerup", 260, 40);
    expect(wrapper.emitted("move")).toBeUndefined();
    wrapper.unmount();
  });

  it("moves nothing when the pointer is released outside every tile", () => {
    const wrapper = strip();
    layOut(wrapper);
    const from = wrapper.get("[data-test='reference-tile-0']").element;
    pointer(from, "pointerdown", 50, 40);
    pointer(from, "pointermove", 900, 900);
    pointer(from, "pointerup", 900, 900);
    expect(wrapper.emitted("move")).toBeUndefined();
    wrapper.unmount();
  });

  it("does not pointer-drag while disabled, on the touch strip, or with a finger", () => {
    for (const [props, pointerType] of [
      [{ disabled: true }, "mouse"],
      [{ touchFriendly: true }, "mouse"],
      [{}, "touch"],
    ] as const) {
      const wrapper = strip(props);
      layOut(wrapper);
      const from = wrapper.get("[data-test='reference-tile-2']").element;
      pointer(from, "pointerdown", 250, 40, pointerType);
      pointer(from, "pointermove", 30, 40, pointerType);
      pointer(from, "pointerup", 30, 40, pointerType);
      expect(wrapper.emitted("move")).toBeUndefined();
      wrapper.unmount();
    }
  });

  it("uses no HTML5 drag for reordering, so the only drag the strip hears is a file", () => {
    const wrapper = strip();
    const tile = wrapper.get("[data-test='reference-tile-0']");
    expect(tile.attributes("draggable")).toBeUndefined();
    expect(tile.attributes("data-reorderable")).toBe("true");
    wrapper.unmount();
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

  it("hands a file dropped on a tile to the strip to append, never as a reorder", () => {
    const wrapper = strip();
    const file = new File(["x"], "new.png", { type: "image/png" });
    const event = new Event("drop", { bubbles: true, cancelable: true });
    Object.defineProperty(event, "dataTransfer", {
      value: { files: [file], types: ["Files"], getData: () => "" },
    });
    wrapper.get("[data-test='reference-tile-1']").element.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
    expect(wrapper.emitted("files")).toEqual([[[file]]]);
    expect(wrapper.emitted("move")).toBeUndefined();
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
      wrapper
        .get("[data-test='reference-tile-0']")
        .attributes("data-reorderable"),
    ).toBeUndefined();
  });
});
