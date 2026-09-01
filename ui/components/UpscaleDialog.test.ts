import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";

import UpscaleDialog from "./UpscaleDialog.vue";

describe("UpscaleDialog", () => {
  it("dismisses an open dialog when Escape is pressed", async () => {
    const wrapper = mount(UpscaleDialog, {
      attachTo: document.body,
      props: {
        open: true,
        kind: "video",
        sourceName: "wide.mp4",
        modelValue: "real-esrgan-x4plus:fp16",
      },
    });

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await wrapper.vm.$nextTick();

    expect(wrapper.emitted("close")).toHaveLength(1);
    wrapper.unmount();
  });

  it("ignores Escape while the dialog is closed", async () => {
    const wrapper = mount(UpscaleDialog, {
      attachTo: document.body,
      props: {
        open: false,
        kind: "image",
        sourceName: "still.png",
        modelValue: "real-esrgan-x4plus:fp16",
      },
    });

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await wrapper.vm.$nextTick();

    expect(wrapper.emitted("close")).toBeUndefined();
    wrapper.unmount();
  });

  it("offers host selection only when multiple capable copies are available", async () => {
    const wrapper = mount(UpscaleDialog, {
      props: {
        open: true,
        kind: "video",
        sourceName: "clip.mp4",
        modelValue: "real-esrgan-x4plus:fp16",
        executionHosts: [
          { key: "local", label: "This Mac" },
          { key: "plato", label: "plato" },
        ],
        executionHostValue: "local",
      },
    });

    const host = document.querySelector(
      "[data-test='upscale-host']",
    ) as HTMLSelectElement;
    expect([...host.options].map((option) => option.text)).toEqual([
      "This Mac",
      "plato",
    ]);
    host.value = "plato";
    host.dispatchEvent(new Event("change"));
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("update:executionHostValue")).toEqual([["plato"]]);
    wrapper.unmount();

    const oneHost = mount(UpscaleDialog, {
      attachTo: document.body,
      props: {
        open: true,
        kind: "image",
        sourceName: "still.png",
        modelValue: "real-esrgan-x4plus:fp16",
        executionHosts: [{ key: "local", label: "This Mac" }],
        executionHostValue: "local",
      },
    });
    expect(document.querySelector("[data-test='upscale-host']")).toBeNull();
    oneHost.unmount();
  });
});
