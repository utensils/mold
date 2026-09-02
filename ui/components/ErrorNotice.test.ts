import { mount } from "@vue/test-utils";
import { describe, expect, it, vi } from "vitest";
import ErrorNotice from "./ErrorNotice.vue";

describe("ErrorNotice", () => {
  it("shows human copy and copies the optional diagnostic payload", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    const wrapper = mount(ErrorNotice, {
      props: {
        message: "plato couldn’t find that model or job.",
        copyMessage:
          "plato couldn’t find that model or job.\n\nTechnical details: SSE request failed with HTTP 404",
      },
    });

    expect(wrapper.get("[data-test='error-notice-message']").text()).toBe(
      "plato couldn’t find that model or job.",
    );
    await wrapper.get("[data-test='copy-error-notice']").trigger("click");
    expect(writeText).toHaveBeenCalledWith(
      "plato couldn’t find that model or job.\n\nTechnical details: SSE request failed with HTTP 404",
    );
    expect(
      wrapper.get("[data-test='copy-error-notice']").attributes("aria-label"),
    ).toBe("Error copied");
  });

  it("wraps an unbroken URL inside the notice instead of overflowing its container", () => {
    const wrapper = mount(ErrorNotice, {
      props: {
        message:
          "Couldn't reach the media host: error sending request for url (http://192.168.1.114:7680/api/gallery/export/mold%2Dhunyuan3d%2Dfp16%2D1788359192104.glb)",
      },
    });
    const message = wrapper.get("[data-test='error-notice-message']");
    expect(message.classes()).toContain("[overflow-wrap:anywhere]");
    expect(message.classes()).toContain("min-w-0");
  });

  it("renders compact spacing and emits from an accessible dismiss button", async () => {
    const wrapper = mount(ErrorNotice, {
      props: { message: "Expansion failed.", compact: true, dismissible: true },
    });

    expect(wrapper.get("[data-test='error-notice']").classes()).toContain(
      "py-1.5",
    );
    expect(
      wrapper.get("[data-test='error-notice-message']").classes(),
    ).toContain("leading-snug");

    const dismiss = wrapper.get("[data-test='dismiss-error-notice']");
    expect(dismiss.attributes("aria-label")).toBe("Dismiss error message");
    await dismiss.trigger("click");
    expect(wrapper.emitted("dismiss")).toEqual([[]]);
  });
});
