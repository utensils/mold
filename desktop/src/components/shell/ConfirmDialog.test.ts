import { afterEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ConfirmDialog from "./ConfirmDialog.vue";

// The dialog teleports to <body>; clean it between tests.
afterEach(() => {
  document.body.innerHTML = "";
});

describe("ConfirmDialog", () => {
  it("renders nothing while closed", () => {
    mount(ConfirmDialog, { props: { open: false, title: "Delete?" }, attachTo: document.body });
    expect(document.querySelector("[data-test='confirm-dialog']")).toBeNull();
  });

  it("shows the title, message, and labels when open", () => {
    mount(ConfirmDialog, {
      props: {
        open: true,
        title: "Launch GPU instance?",
        message: "Billing begins immediately.",
        confirmLabel: "Launch",
      },
      attachTo: document.body,
    });
    const dialog = document.querySelector("[data-test='confirm-dialog']");
    expect(dialog?.textContent).toContain("Launch GPU instance?");
    expect(dialog?.textContent).toContain("Billing begins immediately.");
    expect(document.querySelector("[data-test='confirm-accept']")?.textContent?.trim()).toBe(
      "Launch",
    );
  });

  it("lets a long confirmation label grow vertically instead of clipping", () => {
    mount(ConfirmDialog, {
      props: {
        open: true,
        title: "Launch GPU instance?",
        confirmLabel: "Launch — billing begins immediately",
      },
      attachTo: document.body,
    });

    const accept = document.querySelector("[data-test='confirm-accept']") as HTMLButtonElement;
    expect(accept.classList).toContain("min-h-8");
    expect(accept.classList).toContain("py-1.5");
    expect(accept.classList).not.toContain("h-8");
  });

  it("emits confirm / cancel", async () => {
    const wrapper = mount(ConfirmDialog, {
      props: { open: true, title: "Delete?" },
      attachTo: document.body,
    });
    (document.querySelector("[data-test='confirm-accept']") as HTMLButtonElement).click();
    (document.querySelector("[data-test='confirm-cancel']") as HTMLButtonElement).click();
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("confirm")).toHaveLength(1);
    expect(wrapper.emitted("cancel")).toHaveLength(1);
  });

  it("locks both buttons while busy so an in-flight action can't be re-fired", async () => {
    const wrapper = mount(ConfirmDialog, {
      props: { open: true, title: "Delete?", busy: true },
      attachTo: document.body,
    });
    (document.querySelector("[data-test='confirm-cancel']") as HTMLButtonElement).click();
    (document.querySelector("[data-test='confirm-accept']") as HTMLButtonElement).click();
    await wrapper.vm.$nextTick();
    // Disabled buttons don't dispatch clicks, so neither event fires.
    expect(wrapper.emitted("cancel")).toBeUndefined();
    expect(wrapper.emitted("confirm")).toBeUndefined();
  });
});
