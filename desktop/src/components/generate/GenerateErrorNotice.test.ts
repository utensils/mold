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

    expect(wrapper.get("[data-test='generate-error-message']").classes()).toContain("text-body");
    await wrapper.get("[data-test='copy-generate-error']").trigger("click");

    expect(writeText).toHaveBeenCalledWith("The generation route changed.");
    expect(wrapper.get("[data-test='copy-generate-error']").attributes("aria-label")).toBe(
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
});
