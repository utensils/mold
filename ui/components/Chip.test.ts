import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import Chip from "./Chip.vue";

function make(extra: Record<string, unknown> = {}, label = "Cinematic") {
  return mount(Chip, { props: extra, slots: { default: label } });
}

describe("Chip", () => {
  it("renders the slot label in a type=button pill", () => {
    const wrapper = make();
    const button = wrapper.find("button");
    expect(button.exists()).toBe(true);
    expect(button.attributes("type")).toBe("button");
    expect(button.text()).toBe("Cinematic");
  });

  it("reflects inactive state via aria-pressed=false and no data-on", () => {
    const wrapper = make();
    expect(wrapper.attributes("aria-pressed")).toBe("false");
    expect(wrapper.attributes("data-on")).toBeUndefined();
  });

  it("reflects active state via aria-pressed=true and data-on", () => {
    const wrapper = make({ active: true });
    expect(wrapper.attributes("aria-pressed")).toBe("true");
    expect(wrapper.attributes("data-on")).toBe("true");
  });

  it("emits click with the mouse event when clicked", async () => {
    const wrapper = make();
    await wrapper.trigger("click");
    const emitted = wrapper.emitted("click");
    expect(emitted).toHaveLength(1);
    expect(emitted![0]![0]).toBeInstanceOf(MouseEvent);
  });

  it("does not emit click when disabled", async () => {
    const wrapper = make({ disabled: true });
    expect(wrapper.attributes("disabled")).toBeDefined();
    await wrapper.trigger("click");
    expect(wrapper.emitted("click")).toBeUndefined();
  });
});
