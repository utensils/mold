import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import RenameDialog from "./RenameDialog.vue";

function mountOpen(initial = "hal9000") {
  return mount(RenameDialog, {
    props: { open: true, title: "Rename host", initial },
    attachTo: document.body,
  });
}

describe("RenameDialog", () => {
  it("renders nothing while closed", () => {
    const wrapper = mount(RenameDialog, {
      props: { open: false, title: "Rename host", initial: "x" },
      attachTo: document.body,
    });
    expect(wrapper.find("[data-test='rename-dialog']").exists()).toBe(false);
    wrapper.unmount();
  });

  it("renders in its own frame at the dialog width", () => {
    const wrapper = mountOpen();
    const dialog = wrapper.find("[data-test='rename-dialog']");
    expect(dialog.classes()).toContain("ms-modal");
    expect(wrapper.find(".ms-modal__panel").attributes("style")).toContain("width: 480px");
    wrapper.unmount();
  });

  it("prefills the current name and saves the trimmed value", async () => {
    const wrapper = mountOpen();
    const input = wrapper.find<HTMLInputElement>("input");
    expect(input.element.value).toBe("hal9000");
    await input.setValue("  render box  ");
    await wrapper.find("[data-test='rename-save']").trigger("click");
    expect(wrapper.emitted("save")).toEqual([["render box"]]);
    wrapper.unmount();
  });

  it("cancels instead of saving an empty name", async () => {
    const wrapper = mountOpen();
    const input = wrapper.find<HTMLInputElement>("input");
    await input.setValue("   ");
    await input.trigger("keydown", { key: "Enter" });
    expect(wrapper.emitted("save")).toBeUndefined();
    expect(wrapper.emitted("cancel")).toHaveLength(1);
    wrapper.unmount();
  });

  it("cancels on Escape", async () => {
    const wrapper = mountOpen();
    await wrapper.find("input").trigger("keydown", { key: "Escape" });
    expect(wrapper.emitted("cancel")).toHaveLength(1);
    wrapper.unmount();
  });
});
