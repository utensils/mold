import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import SheetPanel from "./SheetPanel.vue";

function make(
  props: Record<string, unknown> = {},
  slots: Record<string, string> = {},
) {
  return mount(SheetPanel, {
    props: { open: true, title: "Advanced", ...props },
    slots: { default: "<p>Sheet content</p>", ...slots },
  });
}

describe("SheetPanel", () => {
  it("renders nothing when closed", () => {
    const wrapper = make({ open: false });
    expect(wrapper.find("[role=dialog]").exists()).toBe(false);
  });

  it("is a modal dialog named by its title", () => {
    const wrapper = make();
    const dialog = wrapper.find("[role=dialog]");
    expect(dialog.attributes("aria-modal")).toBe("true");
    expect(dialog.attributes("aria-label")).toBe("Advanced");
    expect(dialog.attributes("tabindex")).toBe("-1");
  });

  it("defaults to the full variant with header, close button and body", () => {
    const wrapper = make();
    expect(wrapper.find("[role=dialog]").attributes("data-variant")).toBe(
      "full",
    );
    expect(wrapper.find(".ms-sheet__panel-full").exists()).toBe(true);
    expect(wrapper.find(".ms-sheet__title").text()).toBe("Advanced");
    expect(wrapper.find("button[aria-label=Close]").exists()).toBe(true);
    expect(wrapper.find(".ms-sheet__body").text()).toContain("Sheet content");
  });

  it("bottom variant renders grab handle and no close button", () => {
    const wrapper = make({ variant: "bottom", title: undefined });
    expect(wrapper.find("[role=dialog]").attributes("data-variant")).toBe(
      "bottom",
    );
    expect(wrapper.find(".ms-sheet__panel-bottom").exists()).toBe(true);
    expect(wrapper.find(".ms-sheet__handle").exists()).toBe(true);
    expect(wrapper.find("button[aria-label=Close]").exists()).toBe(false);
    expect(wrapper.text()).toContain("Sheet content");
  });

  it("bottom variant shows the title only when provided", () => {
    expect(
      make({ variant: "bottom" }).find(".ms-sheet__bottom-title").text(),
    ).toBe("Advanced");
    expect(
      make({ variant: "bottom", title: undefined })
        .find(".ms-sheet__bottom-title")
        .exists(),
    ).toBe(false);
  });

  it("emits close on backdrop click but not on panel click", async () => {
    const wrapper = make({ variant: "bottom" });
    await wrapper.find(".ms-sheet__panel-bottom").trigger("click");
    expect(wrapper.emitted("close")).toBeUndefined();
    await wrapper.find("[role=dialog]").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("emits close from the full-variant close button", async () => {
    const wrapper = make();
    await wrapper.find("button[aria-label=Close]").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
  });

  it("emits close on Escape in both variants", async () => {
    for (const variant of ["full", "bottom"] as const) {
      const wrapper = make({ variant });
      await wrapper.find("[role=dialog]").trigger("keydown", { key: "Escape" });
      expect(wrapper.emitted("close")).toHaveLength(1);
    }
  });

  it("focuses the overlay root when opened", async () => {
    const wrapper = mount(SheetPanel, {
      props: { open: false },
      attachTo: document.body,
    });
    await wrapper.setProps({ open: true });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(document.activeElement).toBe(wrapper.find("[role=dialog]").element);
    wrapper.unmount();
  });
});
