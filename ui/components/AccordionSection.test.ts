import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import AccordionSection from "./AccordionSection.vue";

function make(
  extra: Record<string, unknown> = {},
  slots: Record<string, string> = {},
) {
  return mount(AccordionSection, {
    props: { open: false, title: "Scheduler & sampling", ...extra },
    slots: { default: "<p>Body content</p>", ...slots },
  });
}

describe("AccordionSection", () => {
  it("renders a button header with aria-expanded reflecting open", () => {
    const closed = make({ open: false });
    const head = closed.find("button.ms-acc__head");
    expect(head.exists()).toBe(true);
    expect(head.attributes("aria-expanded")).toBe("false");

    const opened = make({ open: true });
    expect(opened.find("button.ms-acc__head").attributes("aria-expanded")).toBe(
      "true",
    );
  });

  it("wires aria-controls to the body element id when open", () => {
    const wrapper = make({ open: true });
    const controls = wrapper
      .find("button.ms-acc__head")
      .attributes("aria-controls");
    expect(controls).toBeTruthy();
    const body = wrapper.find(".ms-acc__body");
    expect(body.attributes("id")).toBe(controls);
  });

  it("only renders the body slot when open", () => {
    const closed = make({ open: false });
    expect(closed.find(".ms-acc__body").exists()).toBe(false);
    expect(closed.text()).not.toContain("Body content");

    const opened = make({ open: true });
    expect(opened.text()).toContain("Body content");
  });

  it("emits toggle on header click", async () => {
    const wrapper = make();
    await wrapper.find("button.ms-acc__head").trigger("click");
    expect(wrapper.emitted("toggle")).toHaveLength(1);
  });

  it("renders title, summary, and icon plate", () => {
    const wrapper = make({
      summary: "Default (recommended)",
      icon: "scheduler",
    });
    expect(wrapper.find(".ms-acc__title").text()).toBe("Scheduler & sampling");
    expect(wrapper.find(".ms-acc__summary").text()).toBe(
      "Default (recommended)",
    );
    expect(wrapper.find(".ms-acc__plate svg").exists()).toBe(true);
  });

  it("omits summary and plate when not provided", () => {
    const wrapper = make();
    expect(wrapper.find(".ms-acc__summary").exists()).toBe(false);
    expect(wrapper.find(".ms-acc__plate").exists()).toBe(false);
  });

  it("shows chevron-down closed and chevron-up open", () => {
    // chevron-down: M6 9l6 6 6-6 · chevron-up: M6 15l6-6 6 6
    const closed = make({ open: false });
    expect(closed.find(".ms-acc__chevron").html()).toContain("M6 9l6 6 6-6");

    const opened = make({ open: true });
    expect(opened.find(".ms-acc__chevron").html()).toContain("M6 15l6-6 6 6");
  });

  it("renders a plain div header without chevron when headerInteractive is false", async () => {
    const wrapper = make({ headerInteractive: false });
    const head = wrapper.find(".ms-acc__head");
    expect(head.element.tagName).toBe("DIV");
    expect(head.attributes("aria-expanded")).toBeUndefined();
    expect(wrapper.find(".ms-acc__chevron").exists()).toBe(false);
    await head.trigger("click");
    expect(wrapper.emitted("toggle")).toBeUndefined();
  });

  it("renders the trailing action slot inside the header", () => {
    const wrapper = make(
      { headerInteractive: false },
      { action: '<span class="fake-switch">switch</span>' },
    );
    expect(wrapper.find(".ms-acc__head .fake-switch").exists()).toBe(true);
  });

  it("still renders the body when open with a non-interactive header", () => {
    const wrapper = make({ open: true, headerInteractive: false });
    expect(wrapper.text()).toContain("Body content");
  });
});
