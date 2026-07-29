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
});
