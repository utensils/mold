import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ActionBlocker from "./ActionBlocker.vue";

describe("ActionBlocker", () => {
  it("names the blocked action and its correction accessibly", () => {
    const wrapper = mount(ActionBlocker, {
      props: { title: "Before you generate", reason: "Choose a model first." },
    });
    expect(wrapper.get("[role='status']").text()).toContain(
      "Before you generate",
    );
    expect(wrapper.text()).toContain("Choose a model first.");
    expect(
      wrapper.get("[data-test='action-blocker']").attributes("data-variant"),
    ).toBe("error");
  });

  it("renders a non-blocking advisory with its own default title", () => {
    const wrapper = mount(ActionBlocker, {
      props: { reason: "The server may reject this size.", variant: "warn" },
    });
    const el = wrapper.get("[data-test='action-blocker']");
    expect(el.attributes("data-variant")).toBe("warn");
    expect(el.classes()).toContain("ms-action-blocker--warn");
    expect(el.text()).toContain("Heads up");
  });
});
