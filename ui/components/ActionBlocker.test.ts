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
  });
});
