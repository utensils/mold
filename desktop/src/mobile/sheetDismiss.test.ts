import { mount, type VueWrapper } from "@vue/test-utils";
import { afterEach, describe, expect, it } from "vitest";
import { resetOverlayStackForTests } from "@ui/lib/overlayStack";
import MobileLibrarySheet from "./MobileLibrarySheet.vue";
import MobileAdvancedSheet from "./MobileAdvancedSheet.vue";
import MobileStyleSheet from "./MobileStyleSheet.vue";
import MobileImagePickerSheet from "./MobileImagePickerSheet.vue";
import MobileReferenceCropSheet from "./MobileReferenceCropSheet.vue";

const wrappers: VueWrapper[] = [];
afterEach(() => {
  wrappers.splice(0).forEach((wrapper) => wrapper.unmount());
  resetOverlayStackForTests();
});
function touch(target: Element, type: string, x: number, y: number, count = 1) {
  const event = new Event(type, { bubbles: true, cancelable: true });
  const points = Array.from({ length: count }, (_, identifier) => ({
    identifier,
    clientX: x,
    clientY: y,
  }));
  Object.defineProperties(event, {
    touches: { value: type === "touchend" ? [] : points },
    changedTouches: { value: points },
  });
  target.dispatchEvent(event);
  return event;
}
function swipe(target: Element, dx = 0, dy = 160) {
  touch(target, "touchstart", 100, 100);
  const move = touch(target, "touchmove", 100 + dx, 100 + dy);
  touch(target, "touchend", 100 + dx, 100 + dy);
  return move;
}
function library(props = {}) {
  const wrapper = mount(MobileLibrarySheet, {
    props: { open: true, title: "All tags", ...props },
    slots: {
      default:
        '<div class="tags"><button>moon 62</button><input /><div class="nested"><p>More tags</p></div></div>',
    },
  });
  wrappers.push(wrapper);
  return wrapper;
}

describe("phone sheet dismissal", () => {
  it("dismisses All tags by default and follows the finger before closing", async () => {
    const wrapper = library();
    const grabber = wrapper.get(".mobile-library-sheet-grabber").element;
    touch(grabber, "touchstart", 100, 100);
    expect(touch(grabber, "touchmove", 100, 260).defaultPrevented).toBe(true);
    await wrapper.vm.$nextTick();
    expect(wrapper.get(".mobile-library-sheet-panel").attributes("style")).toContain("translateY");
    touch(grabber, "touchend", 100, 260);
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("scrolls content without dismissing, but keeps the grabber usable after scrolling", () => {
    const wrapper = library();
    wrapper.get(".mobile-library-sheet-body").element.scrollTop = 90;
    expect(swipe(wrapper.get(".tags").element).defaultPrevented).toBe(false);
    expect(wrapper.emitted("close")).toBeUndefined();
    swipe(wrapper.get(".mobile-library-sheet-grabber").element);
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("respects nested scrollers and interactive fields", () => {
    const wrapper = library();
    wrapper.get(".nested").element.scrollTop = 30;
    for (const selector of [".nested p", ".tags button", "input"]) {
      expect(swipe(wrapper.get(selector).element).defaultPrevented).toBe(false);
    }
    expect(wrapper.emitted("close")).toBeUndefined();
  });

  it.each([
    [0, 40],
    [140, 20],
    [0, -140],
  ])("does not dismiss a short, horizontal or upward drag (%s, %s)", (dx, dy) => {
    const wrapper = library();
    swipe(wrapper.get(".tags").element, dx, dy);
    expect(wrapper.emitted("close")).toBeUndefined();
  });

  it.each(["touchcancel", "multitouch", "direction"])(
    "abandons a %s gesture until a fresh start",
    (kind) => {
      const wrapper = library();
      const target = wrapper.get(".tags").element;
      touch(target, "touchstart", 100, 100);
      touch(target, "touchmove", kind === "direction" ? 220 : 100, 120);
      if (kind === "touchcancel") touch(target, "touchcancel", 100, 120);
      if (kind === "multitouch") touch(target, "touchmove", 100, 220, 2);
      touch(target, "touchmove", 100, 300);
      touch(target, "touchend", 100, 300);
      expect(wrapper.emitted("close")).toBeUndefined();
      swipe(target);
      expect(wrapper.emitted("close")).toHaveLength(1);
    },
  );

  it("does not dismiss disabled, closed or covered sheets", async () => {
    const lower = library();
    const upper = library({ swipeToDismiss: false });
    swipe(lower.get(".tags").element);
    swipe(upper.get(".tags").element);
    expect(lower.emitted("close")).toBeUndefined();
    expect(upper.emitted("close")).toBeUndefined();
    await upper.setProps({ open: false, swipeToDismiss: true });
    swipe(upper.get(".tags").element);
    expect(upper.emitted("close")).toBeUndefined();
  });

  it.each([
    [MobileAdvancedSheet, { count: 0 }, ".mobile-sheet-grabber"],
    [MobileStyleSheet, { models: [], selected: null }, ".mobile-sheet-grabber"],
    [MobileImagePickerSheet, { target: null }, "header strong"],
    [
      MobileReferenceCropSheet,
      {
        title: "Crop",
        image: { data: "", mimeType: "image/png", width: 10, height: 10 },
        crop: null,
      },
      ".mobile-crop-sheet-grabber",
    ],
  ] as const)("wires downward dismissal on %s", (component, props, selector) => {
    const wrapper = mount(component as typeof MobileAdvancedSheet, {
      props: { open: true, ...props } as never,
      global: {
        stubs: { ReferenceCropEditor: { template: '<div class="crop-canvas">Crop canvas</div>' } },
      },
    });
    wrappers.push(wrapper);
    if (wrapper.find(".crop-canvas").exists()) {
      expect(swipe(wrapper.get(".crop-canvas").element).defaultPrevented).toBe(false);
      expect(wrapper.emitted("close")).toBeUndefined();
    }
    swipe(wrapper.get(selector).element);
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("caps the fall and keeps the scrim legible however far the finger travels", async () => {
    // Ported from the lane's own composable test when main's implementation
    // won the merge: these two numbers are what stop a committed gesture
    // dragging the panel off the screen and blacking out the scrim behind it.
    const wrapper = library();
    const grabber = wrapper.get(".mobile-library-sheet-grabber").element;
    touch(grabber, "touchstart", 100, 100);
    touch(grabber, "touchmove", 100, 1_000);
    await wrapper.vm.$nextTick();

    expect(wrapper.get(".mobile-library-sheet-panel").attributes("style")).toContain(
      "translateY(280px)",
    );
    const scrim = wrapper.get(".mobile-library-sheet-backdrop").attributes("style") ?? "";
    expect(Number(/opacity:\s*([\d.]+)/.exec(scrim)?.[1])).toBeCloseTo(0.24, 5);
  });
});
