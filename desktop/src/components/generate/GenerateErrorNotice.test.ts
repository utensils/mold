import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import GenerateErrorNotice from "./GenerateErrorNotice.vue";

describe("GenerateErrorNotice", () => {
  const writeText = vi.fn(async () => undefined);

  beforeEach(() => {
    writeText.mockClear();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
  });

  it("renders readable error copy and copies the complete message", async () => {
    const wrapper = mount(GenerateErrorNotice, {
      props: { message: "The generation route changed." },
    });

    expect(wrapper.get("[data-test='error-notice-message']").classes()).toContain("text-body");
    await wrapper.get("[data-test='copy-error-notice']").trigger("click");

    expect(writeText).toHaveBeenCalledWith("The generation route changed.");
    expect(wrapper.get("[data-test='copy-error-notice']").attributes("aria-label")).toBe(
      "Error copied",
    );
  });

  it("renders recovery actions supplied by the owning workflow", () => {
    const wrapper = mount(GenerateErrorNotice, {
      props: { message: "Resolve this first." },
      slots: { actions: "<button data-test='recovery-action'>Recover</button>" },
    });

    expect(wrapper.get("[data-test='recovery-action']").text()).toBe("Recover");
  });

  it("clears transient clipboard feedback when the error changes", async () => {
    const wrapper = mount(GenerateErrorNotice, {
      props: { message: "The first generation failed." },
    });

    await wrapper.get("[data-test='copy-error-notice']").trigger("click");
    expect(wrapper.get("[data-test='copy-error-notice']").attributes("aria-label")).toBe(
      "Error copied",
    );

    await wrapper.setProps({ message: "The retry failed differently." });

    expect(wrapper.get("[data-test='copy-error-notice']").attributes("aria-label")).toBe(
      "Copy error message",
    );
  });
});
