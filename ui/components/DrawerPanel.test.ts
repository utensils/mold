import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import DrawerPanel from "./DrawerPanel.vue";

function make(
  props: Record<string, unknown> = {},
  slots: Record<string, string> = {},
) {
  return mount(DrawerPanel, {
    props: { open: true, title: "Model details", ...props },
    slots: { default: "<p>Body content</p>", ...slots },
  });
}

describe("DrawerPanel", () => {
  it("renders nothing when closed", () => {
    const wrapper = make({ open: false });
    expect(wrapper.find("[role=dialog]").exists()).toBe(false);
  });

  it("is a modal dialog named by its title", () => {
    const wrapper = make();
    const dialog = wrapper.find("[role=dialog]");
    expect(dialog.attributes("aria-modal")).toBe("true");
    expect(dialog.attributes("aria-label")).toBe("Model details");
    expect(dialog.attributes("tabindex")).toBe("-1");
  });

  it("renders the title and the default slot body", () => {
    const wrapper = make();
    expect(wrapper.find(".ms-drawer__title").text()).toBe("Model details");
    expect(wrapper.find(".ms-drawer__body").text()).toContain("Body content");
  });

  it("header slot replaces the title text", () => {
    const wrapper = make({}, { header: "<strong>Custom header</strong>" });
    expect(wrapper.find(".ms-drawer__title").exists()).toBe(false);
    expect(wrapper.find(".ms-drawer__header").text()).toContain(
      "Custom header",
    );
  });

  it("applies the width prop to the panel, defaulting to 452", () => {
    expect(make().find(".ms-drawer__panel").attributes("style")).toContain(
      "width: 452px",
    );
    expect(
      make({ width: 560 }).find(".ms-drawer__panel").attributes("style"),
    ).toContain("width: 560px");
  });

  it("emits close on backdrop click but not on panel click", async () => {
    const wrapper = make();
    await wrapper.find(".ms-drawer__panel").trigger("click");
    expect(wrapper.emitted("close")).toBeUndefined();
    await wrapper.find("[role=dialog]").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("emits close from the close button", async () => {
    const wrapper = make();
    await wrapper.find("button[aria-label=Close]").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("emits close on Escape", async () => {
    const wrapper = make();
    await wrapper.find("[role=dialog]").trigger("keydown", { key: "Escape" });
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("focuses the overlay root when opened", async () => {
    const wrapper = mount(DrawerPanel, {
      props: { open: false },
      attachTo: document.body,
    });
    await wrapper.setProps({ open: true });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(document.activeElement).toBe(wrapper.find("[role=dialog]").element);
    wrapper.unmount();
  });

  it("omits the footer wrapper unless the footer slot is used", () => {
    expect(make().find(".ms-drawer__footer").exists()).toBe(false);
    const wrapper = make({}, { footer: "<button>Pull model</button>" });
    expect(wrapper.find(".ms-drawer__footer").text()).toContain("Pull model");
  });
});
