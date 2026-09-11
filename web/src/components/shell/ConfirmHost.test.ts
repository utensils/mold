import { mount } from "@vue/test-utils";
import { afterEach, describe, expect, it } from "vitest";
import ConfirmHost from "./ConfirmHost.vue";
import {
  requestChoice,
  requestConfirm,
  requestText,
  resetNotifications,
} from "../../lib/toasts";

afterEach(() => {
  resetNotifications();
  document.body.innerHTML = "";
});

function el(test: string): HTMLElement | null {
  return document.querySelector(`[data-test='${test}']`);
}

async function settle() {
  await new Promise((resolve) => setTimeout(resolve, 0));
}

describe("ConfirmHost", () => {
  it("shows nothing while nothing is being asked", () => {
    mount(ConfirmHost, { attachTo: document.body });
    expect(el("confirm-dialog")).toBeNull();
  });

  it("answers a destructive question yes, with a danger button and no typing", async () => {
    const wrapper = mount(ConfirmHost, { attachTo: document.body });
    const answer = requestConfirm({
      title: "Empty trash?",
      body: "This can't be undone.",
      confirmLabel: "Delete forever",
      danger: true,
    });
    await wrapper.vm.$nextTick();
    expect(el("confirm-dialog")?.textContent).toContain("Empty trash?");
    expect(el("confirm-accept")?.dataset.danger).toBe("true");
    expect(el("confirm-typed")).toBeNull();
    (el("confirm-accept") as HTMLButtonElement).click();
    await expect(answer).resolves.toBe(true);
    wrapper.unmount();
  });

  it("answers no when it is cancelled", async () => {
    const wrapper = mount(ConfirmHost, { attachTo: document.body });
    const answer = requestConfirm({ title: "Forget this machine?", body: "" });
    await wrapper.vm.$nextTick();
    (el("confirm-cancel") as HTMLButtonElement).click();
    await expect(answer).resolves.toBe(false);
    wrapper.unmount();
  });

  it("carries a typed line back, starting from what it was given", async () => {
    const wrapper = mount(ConfirmHost, { attachTo: document.body });
    const answer = requestText({
      title: "Rename album",
      label: "Album name",
      initial: "Holiday",
    });
    await wrapper.vm.$nextTick();
    const input = el("dialog-text") as HTMLInputElement;
    expect(input.value).toBe("Holiday");
    input.value = "Holiday 2026";
    input.dispatchEvent(new Event("input"));
    await wrapper.vm.$nextTick();
    (el("confirm-accept") as HTMLButtonElement).click();
    await expect(answer).resolves.toBe("Holiday 2026");
    wrapper.unmount();
  });

  it("reports nothing at all when a typed line is cancelled", async () => {
    const wrapper = mount(ConfirmHost, { attachTo: document.body });
    const answer = requestText({ title: "Rename album", initial: "Holiday" });
    await wrapper.vm.$nextTick();
    (el("confirm-cancel") as HTMLButtonElement).click();
    await expect(answer).resolves.toBeNull();
    wrapper.unmount();
  });

  it("reports the chosen one of several answers", async () => {
    const wrapper = mount(ConfirmHost, { attachTo: document.body });
    const answer = requestChoice({
      title: "This print was made by a sequence",
      body: "Reuse gives you the first shot's prompt.",
      choices: [
        { id: "first", label: "Use the first shot" },
        { id: "cancel", label: "Leave it", danger: true },
      ],
    });
    await wrapper.vm.$nextTick();
    (el("dialog-choice-first") as HTMLButtonElement).click();
    await expect(answer).resolves.toBe("first");
    wrapper.unmount();
  });

  it("starts each question from its own initial value", async () => {
    const wrapper = mount(ConfirmHost, { attachTo: document.body });
    void requestText({ title: "Rename", initial: "First" });
    await wrapper.vm.$nextTick();
    void requestText({ title: "Rename again", initial: "Second" });
    await settle();
    await wrapper.vm.$nextTick();
    expect((el("dialog-text") as HTMLInputElement).value).toBe("Second");
    wrapper.unmount();
  });
});
