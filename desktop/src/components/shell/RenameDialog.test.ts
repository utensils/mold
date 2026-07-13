import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import RenameDialog from "./RenameDialog.vue";

function mountOpen(initial = "hal9000") {
  return mount(RenameDialog, {
    props: { open: true, title: "Rename host", initial },
  });
}

describe("RenameDialog", () => {
  it("renders nothing while closed", () => {
    const wrapper = mount(RenameDialog, {
      props: { open: false, title: "Rename host", initial: "x" },
    });
    expect(document.querySelector("[data-test='rename-dialog']")).toBeNull();
    wrapper.unmount();
  });

  it("prefills the current name and saves the trimmed value", async () => {
    const wrapper = mountOpen();
    const input = document.querySelector<HTMLInputElement>("[data-test='rename-dialog'] input")!;
    expect(input.value).toBe("hal9000");
    input.value = "  render box  ";
    input.dispatchEvent(new Event("input"));
    document
      .querySelector<HTMLButtonElement>("[data-test='rename-save']")!
      .dispatchEvent(new MouseEvent("click"));
    expect(wrapper.emitted("save")).toEqual([["render box"]]);
    wrapper.unmount();
  });

  it("cancels instead of saving an empty name", async () => {
    const wrapper = mountOpen();
    const input = document.querySelector<HTMLInputElement>("[data-test='rename-dialog'] input")!;
    input.value = "   ";
    input.dispatchEvent(new Event("input"));
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    expect(wrapper.emitted("save")).toBeUndefined();
    expect(wrapper.emitted("cancel")).toHaveLength(1);
    wrapper.unmount();
  });

  it("cancels on Escape", async () => {
    const wrapper = mountOpen();
    const input = document.querySelector<HTMLInputElement>("[data-test='rename-dialog'] input")!;
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    expect(wrapper.emitted("cancel")).toHaveLength(1);
    wrapper.unmount();
  });
});
