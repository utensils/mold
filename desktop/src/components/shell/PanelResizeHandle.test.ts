import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import PanelResizeHandle from "./PanelResizeHandle.vue";

function mountHandle() {
  return mount(PanelResizeHandle, { props: { label: "Resize sidebar" } });
}

describe("PanelResizeHandle a11y", () => {
  it("renders a labelled vertical separator with a col-resize cursor", () => {
    const wrapper = mountHandle();
    expect(wrapper.attributes("role")).toBe("separator");
    expect(wrapper.attributes("aria-orientation")).toBe("vertical");
    expect(wrapper.attributes("aria-label")).toBe("Resize sidebar");
    expect(wrapper.classes().join(" ")).toContain("cursor-col-resize");
  });
});

describe("PanelResizeHandle drag lifecycle", () => {
  it("emits resize deltas from the pointerdown origin while dragging", async () => {
    const wrapper = mountHandle();
    // Moves before a pointerdown must be inert.
    await wrapper.trigger("pointermove", { clientX: 50 });
    expect(wrapper.emitted("resize")).toBeUndefined();

    await wrapper.trigger("pointerdown", { button: 0, pointerId: 1, clientX: 100 });
    await wrapper.trigger("pointermove", { buttons: 1, clientX: 130 });
    await wrapper.trigger("pointermove", { buttons: 1, clientX: 90 });
    expect(wrapper.emitted("resize")).toEqual([[30], [-10]]);
  });

  it("emits commit on pointerup and stops tracking moves", async () => {
    const wrapper = mountHandle();
    await wrapper.trigger("pointerdown", { button: 0, pointerId: 1, clientX: 100 });
    await wrapper.trigger("pointermove", { buttons: 1, clientX: 140 });
    await wrapper.trigger("pointerup", { pointerId: 1 });
    expect(wrapper.emitted("commit")).toHaveLength(1);

    await wrapper.trigger("pointermove", { buttons: 1, clientX: 300 });
    expect(wrapper.emitted("resize")).toEqual([[40]]);
  });

  it("treats pointercancel like pointerup", async () => {
    const wrapper = mountHandle();
    await wrapper.trigger("pointerdown", { button: 0, pointerId: 7, clientX: 10 });
    await wrapper.trigger("pointercancel", { pointerId: 7 });
    expect(wrapper.emitted("commit")).toHaveLength(1);
  });

  it("ignores non-primary buttons", async () => {
    const wrapper = mountHandle();
    await wrapper.trigger("pointerdown", { button: 2, pointerId: 1, clientX: 100 });
    await wrapper.trigger("pointermove", { clientX: 130 });
    expect(wrapper.emitted("resize")).toBeUndefined();
  });

  it("self-heals a lost pointer capture (all buttons already up)", async () => {
    const wrapper = mountHandle();
    await wrapper.trigger("pointerdown", { button: 0, pointerId: 1, clientX: 100 });
    await wrapper.trigger("pointermove", { clientX: 130, buttons: 0 });
    // No phantom resize; the drag ends and commits once.
    expect(wrapper.emitted("resize")).toBeUndefined();
    expect(wrapper.emitted("commit")).toHaveLength(1);
    await wrapper.trigger("pointermove", { clientX: 150, buttons: 0 });
    expect(wrapper.emitted("resize")).toBeUndefined();
  });

  it("emits reset on double-click", async () => {
    const wrapper = mountHandle();
    await wrapper.trigger("dblclick");
    expect(wrapper.emitted("reset")).toHaveLength(1);
  });
});
