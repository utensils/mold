import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import NavItem from "./NavItem.vue";

function make(extra: Record<string, unknown> = {}) {
  return mount(NavItem, {
    props: { icon: "create", label: "Create", ...extra },
  });
}

describe("NavItem", () => {
  it("renders a button with the label and leading icon", () => {
    const wrapper = make();
    expect(wrapper.find("button").exists()).toBe(true);
    expect(wrapper.text()).toContain("Create");
    expect(wrapper.find("svg").exists()).toBe(true);
  });

  it("emits select on click", async () => {
    const wrapper = make();
    await wrapper.find("button").trigger("click");
    expect(wrapper.emitted("select")).toHaveLength(1);
  });

  it("sets aria-current=page only when active", async () => {
    const wrapper = make();
    expect(wrapper.find("button").attributes("aria-current")).toBeUndefined();
    expect(wrapper.find("button").attributes("data-on")).toBeUndefined();
    await wrapper.setProps({ active: true });
    expect(wrapper.find("button").attributes("aria-current")).toBe("page");
    expect(wrapper.find("button").attributes("data-on")).toBe("true");
  });

  it("defaults to the row variant", () => {
    const wrapper = make();
    expect(wrapper.find("button").classes()).toContain("ms-nav--row");
    expect(wrapper.find("button").classes()).not.toContain("ms-nav--tab");
  });

  it("hides the label when collapsed but keeps an accessible name", () => {
    const wrapper = make({ collapsed: true });
    expect(wrapper.find(".ms-nav__label").exists()).toBe(false);
    expect(wrapper.find("button").classes()).toContain("ms-nav--collapsed");
    expect(wrapper.find("button").attributes("aria-label")).toBe("Create");
  });

  it("tab variant renders a mono caption and ignores collapsed", () => {
    const wrapper = make({ variant: "tab", collapsed: true });
    expect(wrapper.find("button").classes()).toContain("ms-nav--tab");
    expect(wrapper.find("button").classes()).not.toContain("ms-nav--collapsed");
    expect(wrapper.find(".ms-nav__label").text()).toBe("Create");
  });

  it("tab variant uses the 22px icon size", () => {
    const wrapper = make({ variant: "tab" });
    expect(wrapper.find("svg").attributes("width")).toBe("22");
  });

  it("row variant uses the 18px icon size", () => {
    const wrapper = make();
    expect(wrapper.find("svg").attributes("width")).toBe("18");
  });

  it("renders a badge only when provided", async () => {
    const wrapper = make();
    expect(wrapper.find(".ms-nav__badge").exists()).toBe(false);
    await wrapper.setProps({ badge: 3 });
    expect(wrapper.find(".ms-nav__badge").text()).toBe("3");
  });
});
