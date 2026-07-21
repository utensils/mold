import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import BadgePill from "./BadgePill.vue";

const TONES = [
  "accent",
  "info",
  "success",
  "warning",
  "danger",
  "neutral",
] as const;

function make(extra: Record<string, unknown> = {}, content = "3 on") {
  return mount(BadgePill, { props: extra, slots: { default: content } });
}

describe("BadgePill", () => {
  it("renders slot content in a non-interactive span", () => {
    const wrapper = make();
    expect(wrapper.element.tagName).toBe("SPAN");
    expect(wrapper.classes()).toContain("ms-badge");
    expect(wrapper.text()).toBe("3 on");
    expect(wrapper.attributes("role")).toBeUndefined();
  });

  it("defaults to the accent tone, solid", () => {
    const wrapper = make();
    expect(wrapper.attributes("data-tone")).toBe("accent");
    expect(wrapper.attributes("data-outline")).toBeUndefined();
  });

  it.each(TONES)("exposes tone %s via data-tone", (tone) => {
    const wrapper = make({ tone });
    expect(wrapper.attributes("data-tone")).toBe(tone);
  });

  it("marks the outline variant via data-outline", () => {
    const wrapper = make({ tone: "danger", outline: true });
    expect(wrapper.attributes("data-outline")).toBe("true");
    expect(wrapper.attributes("data-tone")).toBe("danger");
  });
});
