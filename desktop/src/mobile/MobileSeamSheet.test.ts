import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import SeamEditor from "@ui/components/SeamEditor.vue";
import MobileSeamSheet from "./MobileSeamSheet.vue";

function mountSheet(props: Partial<InstanceType<typeof MobileSeamSheet>["$props"]> = {}) {
  return mount(MobileSeamSheet, {
    props: {
      open: true,
      transition: "smooth" as const,
      fadeFrames: 8,
      motionTail: 17,
      fps: 24,
      fadeFramesMax: 32,
      fromLabel: "opening",
      toLabel: "clip 2",
      ...props,
    },
  });
}

describe("MobileSeamSheet", () => {
  it("names the seam it edits in its body head row", () => {
    // SheetPanel's `full` variant drops #header, so this bespoke sheet renders
    // its own head row inside the scrolling body.
    const wrapper = mountSheet();
    const head = wrapper.get("[data-test='mobile-seam-sheet-head']");
    expect(head.text()).toBe("Join · opening → clip 2");
    expect(wrapper.get(".mobile-seam-sheet-body").element.contains(head.element)).toBe(true);
  });

  it("labels a zero motion tail as joined clips instead of Smooth", () => {
    const wrapper = mountSheet({ motionTail: 0, fromLabel: "clip 2", toLabel: "clip 3" });
    expect(wrapper.get("[data-test='mobile-seam-sheet-head']").text()).toBe(
      "Join · clip 2 → clip 3",
    );
    expect(wrapper.text()).toContain("Join");
    expect(wrapper.text()).not.toContain("Join clips");
    expect(wrapper.text()).not.toContain("Smooth");
  });

  it("hosts the shared SeamEditor at iPhone touch size and forwards its updates", async () => {
    const wrapper = mountSheet({ transition: "fade", fadeFrames: 12 });
    const editor = wrapper.getComponent(SeamEditor);
    expect(editor.props("large")).toBe(true);
    expect(editor.props("fadeFramesMax")).toBe(32);
    // ⌥-to-all is a pointer affordance mobile has no way to express.
    expect(editor.props("showApplyAllHint")).toBe(false);

    await editor.vm.$emit("update:transition", "cut");
    await editor.vm.$emit("update:fadeFrames", 14);

    expect(wrapper.emitted("update:transition")).toEqual([["cut"]]);
    expect(wrapper.emitted("update:fadeFrames")).toEqual([[14]]);
  });

  it("clamps the fade stepper to the server cap, falling back to 32", () => {
    expect(mountSheet({ fadeFramesMax: 16 }).getComponent(SeamEditor).props("fadeFramesMax")).toBe(
      16,
    );
    const noCap = mount(MobileSeamSheet, {
      props: {
        open: true,
        transition: "fade" as const,
        fadeFrames: 8,
        motionTail: 17,
        fps: 24,
        fromLabel: "opening",
        toLabel: "clip 2",
      },
    });
    expect(noCap.getComponent(SeamEditor).props("fadeFramesMax")).toBe(32);
  });

  it("closes from Done and from the backdrop, and hides while closed", async () => {
    const wrapper = mountSheet();
    await wrapper.get("[data-test='mobile-seam-sheet-done']").trigger("click");
    await wrapper.get("[data-test='mobile-seam-sheet-backdrop']").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(2);

    expect(wrapper.get("[data-test='mobile-seam-sheet']").classes()).toContain("is-open");
    await wrapper.setProps({ open: false });
    const closed = wrapper.get("[data-test='mobile-seam-sheet']");
    expect(closed.classes()).not.toContain("is-open");
    expect(closed.attributes("aria-hidden")).toBe("true");
  });
});
