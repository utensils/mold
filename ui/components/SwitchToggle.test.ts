import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import SwitchToggle from "./SwitchToggle.vue";

function make(modelValue = false, extra: Record<string, unknown> = {}) {
  return mount(SwitchToggle, {
    props: { modelValue, label: "Upscale after generate", ...extra },
  });
}

describe("SwitchToggle", () => {
  it("renders a switch with aria-checked reflecting the model", () => {
    const off = make(false).find("[role=switch]");
    expect(off.attributes("aria-checked")).toBe("false");
    expect(off.attributes("data-on")).toBeUndefined();
    const on = make(true).find("[role=switch]");
    expect(on.attributes("aria-checked")).toBe("true");
    expect(on.attributes("data-on")).toBe("true");
  });

  it("uses the label prop as the accessible name", () => {
    const wrapper = make();
    expect(wrapper.find("[role=switch]").attributes("aria-label")).toBe(
      "Upscale after generate",
    );
  });

  it("emits the inverted value on click", async () => {
    const wrapper = make(false);
    await wrapper.find("[role=switch]").trigger("click");
    expect(wrapper.emitted("update:modelValue")).toEqual([[true]]);
  });

  it("toggles off from on", async () => {
    const wrapper = make(true);
    await wrapper.find("[role=switch]").trigger("click");
    expect(wrapper.emitted("update:modelValue")).toEqual([[false]]);
  });

  it("toggles with Space and Enter", async () => {
    const wrapper = make(false);
    const el = wrapper.find("[role=switch]");
    await el.trigger("keydown", { key: " " });
    await el.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("update:modelValue")).toEqual([[true], [true]]);
  });

  it("ignores unrelated keys", async () => {
    const wrapper = make(false);
    await wrapper.find("[role=switch]").trigger("keydown", { key: "a" });
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
  });

  it("ignores interaction when disabled and keeps the disabled attribute", async () => {
    const wrapper = make(false, { disabled: true });
    const el = wrapper.find("[role=switch]");
    expect(el.attributes("disabled")).toBeDefined();
    await el.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("update:modelValue")).toBeUndefined();
  });
});
