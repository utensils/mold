import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import { ref, type Ref } from "vue";
import ReferenceCropEditor from "./ReferenceCropEditor.vue";
import ReferenceCropModal from "./ReferenceCropModal.vue";

const image = {
  data: "SU1BR0U=",
  mimeType: "image/png",
  width: 1024,
  height: 768,
};

const PROPS = { open: true, image, crop: null, title: "Crop reference 1" };

describe("ReferenceCropModal", () => {
  it("renders nothing while closed", () => {
    const wrapper = mount(ReferenceCropModal, {
      props: { ...PROPS, open: false },
    });
    expect(wrapper.findComponent(ReferenceCropEditor).exists()).toBe(false);
  });

  it("renders nothing without an image, however open it is asked to be", () => {
    const wrapper = mount(ReferenceCropModal, {
      props: { ...PROPS, image: null },
    });
    expect(wrapper.findComponent(ReferenceCropEditor).exists()).toBe(false);
  });

  it("hosts the shared editor in a dialog and forwards apply and cancel", async () => {
    const wrapper = mount(ReferenceCropModal, {
      props: PROPS,
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

  /*
   * The host element is the one thing the two surfaces could not share. The
   * browser puts Create in a long scrolling column, so the overlay needs a
   * viewport host of its own; the desktop app opens the crop from inside the
   * inspector, whose own containing block would otherwise trap it. Both are
   * the same markup in a different place, so the place is a prop.
   */
  it("stays in place by default", () => {
    const wrapper = mount(ReferenceCropModal, {
      props: PROPS,
      attachTo: document.body,
    });
    expect(wrapper.find("[data-test='reference-crop-host']").exists()).toBe(
      true,
    );
    wrapper.unmount();
  });

  it("moves its host out to the document when asked to", () => {
    const wrapper = mount(ReferenceCropModal, {
      props: { ...PROPS, teleport: true },
      attachTo: document.body,
    });
    expect(wrapper.find("[data-test='reference-crop-host']").exists()).toBe(
      false,
    );
    expect(
      document.body.querySelector("[data-test='reference-crop-host']"),
    ).not.toBeNull();
    wrapper.unmount();
    expect(
      document.body.querySelector("[data-test='reference-crop-host']"),
    ).toBeNull();
  });

  /*
   * Tab trapping and returning focus to the opener need the host surface's own
   * rules, so the surface hands them in. Without one the shared ModalPanel's
   * Escape and Tab handling still stands on its own.
   */
  it("engages the focus contract its surface hands it", () => {
    const onKeydown = vi.fn();
    const useFocus = vi.fn(
      (open: Ref<boolean>, host: Ref<HTMLElement | null>) => {
        expect(open.value).toBe(true);
        expect(host).toBeDefined();
        return { onKeydown };
      },
    );
    const wrapper = mount(ReferenceCropModal, {
      props: { ...PROPS, useFocus },
      attachTo: document.body,
    });
    expect(useFocus).toHaveBeenCalledTimes(1);
    wrapper.get("[data-test='reference-crop-host']").trigger("keydown.tab");
    expect(onKeydown).toHaveBeenCalled();
    wrapper.unmount();
  });

  it("closes through the surface's own close callback", () => {
    const close = ref<(() => void) | null>(null);
    const useFocus = (
      _open: Ref<boolean>,
      _host: Ref<HTMLElement | null>,
      onClose?: () => void,
    ) => {
      close.value = onClose ?? null;
      return { onKeydown: () => {} };
    };
    const wrapper = mount(ReferenceCropModal, {
      props: { ...PROPS, useFocus },
      attachTo: document.body,
    });
    close.value?.();
    expect(wrapper.emitted("close")).toHaveLength(1);
    wrapper.unmount();
  });
});
