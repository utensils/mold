import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ReferenceCropModal from "./ReferenceCropModal.vue";
import ReferenceCropEditor from "@studio/components/ReferenceCropEditor.vue";

const image = {
  data: "SU1BR0U=",
  mimeType: "image/png",
  width: 1024,
  height: 768,
};

describe("ReferenceCropModal", () => {
  it("renders nothing while closed", () => {
    const wrapper = mount(ReferenceCropModal, {
      props: { open: false, image, crop: null, title: "Crop reference 1" },
    });
    expect(wrapper.findComponent(ReferenceCropEditor).exists()).toBe(false);
  });

  it("hosts the shared editor in a dialog and forwards apply and cancel", async () => {
    const wrapper = mount(ReferenceCropModal, {
      props: { open: true, image, crop: null, title: "Crop reference 1" },
      attachTo: document.body,
    });
    expect(wrapper.get("[role='dialog']").attributes("aria-label")).toBe(
      "Crop reference 1",
    );
    const editor = wrapper.getComponent(ReferenceCropEditor);
    editor.vm.$emit("apply", { x: 0, y: 0, width: 512, height: 768 });
    expect(wrapper.emitted("apply")).toEqual([
      [{ x: 0, y: 0, width: 512, height: 768 }],
    ]);
    editor.vm.$emit("cancel");
    expect(wrapper.emitted("close")).toHaveLength(1);
    wrapper.unmount();
  });
});
