import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ReferenceCropEditor from "./ReferenceCropEditor.vue";

const IMAGE = {
  data: "SU1BR0U=",
  mimeType: "image/png",
  width: 1920,
  height: 1080,
};
// 8:5 sits outside every preset tolerance, so the box starts unlocked.
const CROP = { x: 400, y: 200, width: 800, height: 500 };

function pointer(type: string, x: number, y: number): MouseEvent {
  return new MouseEvent(type, { clientX: x, clientY: y, bubbles: true });
}

function mountEditor(crop: typeof CROP | null = CROP) {
  return mount(ReferenceCropEditor, {
    props: { image: IMAGE, crop },
    attachTo: document.body,
  });
}

describe("ReferenceCropEditor", () => {
  it("shows the crop, its pad cost against the uncropped source, and the matching preset", () => {
    const wrapper = mountEditor({ x: 420, y: 0, width: 1080, height: 1080 });
    const hint = wrapper.get('[data-test="crop-hint"]').text();
    expect(hint).toContain("1080×1080");
    expect(hint).toContain("4,096");
    expect(hint).toContain("7,296");
    expect(
      wrapper.get('[data-test="crop-aspect-1:1"]').attributes("aria-checked"),
    ).toBe("true");
    expect(wrapper.get('[data-test="crop-box"]').attributes("style")).toContain(
      "21.875%",
    );
    wrapper.unmount();
  });

  it("drags the box to move it and a corner handle to resize it, then applies a normalized rect", async () => {
    const wrapper = mountEditor();
    const box = wrapper.get('[data-test="crop-box"]');
    box.element.dispatchEvent(pointer("pointerdown", 100, 100));
    window.dispatchEvent(pointer("pointermove", 200.4, 150));
    window.dispatchEvent(pointer("pointerup", 200.4, 150));
    await wrapper.vm.$nextTick();

    const handle = wrapper.get('[data-test="crop-handle-se"]');
    handle.element.dispatchEvent(pointer("pointerdown", 1300, 750));
    window.dispatchEvent(pointer("pointermove", 1400, 850));
    window.dispatchEvent(pointer("pointerup", 1400, 850));
    await wrapper.vm.$nextTick();

    await wrapper.get('[data-test="crop-apply"]').trigger("click");
    expect(wrapper.emitted("apply")).toEqual([
      [{ x: 500, y: 250, width: 900, height: 600 }],
    ]);
    wrapper.unmount();
  });

  it("keeps a chosen aspect locked while a corner is dragged", async () => {
    const wrapper = mountEditor();
    await wrapper.get('[data-test="crop-aspect-1:1"]').trigger("click");
    const handle = wrapper.get('[data-test="crop-handle-se"]');
    handle.element.dispatchEvent(pointer("pointerdown", 1500, 1080));
    window.dispatchEvent(pointer("pointermove", 1700, 900));
    window.dispatchEvent(pointer("pointerup", 1700, 900));
    await wrapper.vm.$nextTick();
    await wrapper.get('[data-test="crop-apply"]').trigger("click");
    const [[applied]] = wrapper.emitted("apply") as [
      [{ width: number; height: number }],
    ];
    expect(applied.width).toBe(applied.height);
    wrapper.unmount();
  });

  it("centers a preset, nudges with the keyboard, and Reset applies as no crop", async () => {
    const wrapper = mountEditor();
    await wrapper.get('[data-test="crop-aspect-1:1"]').trigger("click");
    expect(wrapper.get('[data-test="crop-hint"]').text()).toContain("4,096");
    await wrapper
      .get('[data-test="crop-box"]')
      .trigger("keydown", { key: "ArrowRight" });
    await wrapper.get('[data-test="crop-box"]').trigger("keydown", {
      key: "ArrowDown",
      shiftKey: true,
    });
    await wrapper.get('[data-test="crop-apply"]').trigger("click");
    expect(wrapper.emitted("apply")).toEqual([
      [{ x: 428, y: 0, width: 1080, height: 1080 }],
    ]);

    await wrapper.get('[data-test="crop-reset"]').trigger("click");
    expect(
      wrapper.get('[data-test="crop-aspect-free"]').attributes("aria-checked"),
    ).toBe("true");
    await wrapper.get('[data-test="crop-apply"]').trigger("click");
    expect(wrapper.emitted("apply")?.[1]).toEqual([null]);
    wrapper.unmount();
  });

  it("cancels without applying and sizes every handle to a 44pt target", async () => {
    const wrapper = mountEditor();
    await wrapper.get('[data-test="crop-cancel"]').trigger("click");
    expect(wrapper.emitted("cancel")).toHaveLength(1);
    expect(wrapper.emitted("apply")).toBeUndefined();
    expect(wrapper.findAll('[data-test^="crop-handle-"]')).toHaveLength(4);
    expect(
      wrapper.get('[data-test="crop-handle-nw"]').attributes("aria-label"),
    ).toBe("Resize crop from the top-left corner");
    wrapper.unmount();
  });
});
