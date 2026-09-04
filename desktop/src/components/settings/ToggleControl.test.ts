import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ToggleControl from "./ToggleControl.vue";

function make(modelValue: boolean) {
  return mount(ToggleControl, { props: { modelValue, ariaLabel: "Save every result" } });
}

describe("ToggleControl", () => {
  it("is a switch named by its aria-label", () => {
    const button = make(false).find("button");
    expect(button.attributes("role")).toBe("switch");
    expect(button.attributes("aria-checked")).toBe("false");
    expect(button.attributes("aria-label")).toBe("Save every result");
  });

  it("takes the theme's radii — the pill is banned in every theme", () => {
    const wrapper = make(false);
    const track = wrapper.find("button");
    const knob = wrapper.find("span");
    expect(track.classes()).toContain("rounded-control");
    expect(track.classes()).not.toContain("rounded-full");
    expect(knob.classes()).toContain("rounded-inner");
    expect(knob.classes()).not.toContain("rounded-full");
  });

  it("dims the knob while off and inks it against the accent while on", () => {
    expect(make(false).find("span").classes()).toContain("bg-fg-dim");
    const on = make(true);
    expect(on.find("button").classes()).toContain("bg-accent");
    expect(on.find("span").classes()).toContain("bg-on-accent");
  });

  it("commits the opposite value on click", async () => {
    const wrapper = make(false);
    await wrapper.find("button").trigger("click");
    expect(wrapper.emitted("commit")).toEqual([[true]]);
  });
});
