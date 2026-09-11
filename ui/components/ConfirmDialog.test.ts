import { afterEach, describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import ConfirmDialog from "./ConfirmDialog.vue";
import source from "./ConfirmDialog.vue?raw";

// The dialog renders in its own frame; the mounts attach to <body>, so clear it
// between tests.
afterEach(() => {
  document.body.innerHTML = "";
});

function el(test: string): HTMLElement | null {
  return document.querySelector(`[data-test='${test}']`);
}

describe("ConfirmDialog", () => {
  it("renders nothing while closed", () => {
    mount(ConfirmDialog, {
      props: { open: false, title: "Delete?" },
      attachTo: document.body,
    });
    expect(el("confirm-dialog")).toBeNull();
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
    expect(el("confirm-dialog")?.textContent).toContain("Launch GPU instance?");
    expect(el("confirm-dialog")?.textContent).toContain(
      "Billing begins immediately.",
    );
    expect(el("confirm-accept")?.textContent?.trim()).toBe("Launch");
    expect(el("confirm-cancel")?.textContent?.trim()).toBe("Cancel");
  });

  it("emits confirm / cancel", async () => {
    const wrapper = mount(ConfirmDialog, {
      props: { open: true, title: "Delete?" },
      attachTo: document.body,
    });
    (el("confirm-accept") as HTMLButtonElement).click();
    (el("confirm-cancel") as HTMLButtonElement).click();
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("confirm")).toHaveLength(1);
    expect(wrapper.emitted("cancel")).toHaveLength(1);
  });

  it("cancels on Escape from anywhere on the page — it is the top overlay", async () => {
    // ModalPanel listens on the document and asks the overlay stack whether
    // it is on top; a confirm that only closes while it holds focus is a trap.
    const wrapper = mount(ConfirmDialog, {
      props: { open: true, title: "Delete?" },
      attachTo: document.body,
    });
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    await wrapper.vm.$nextTick();
    expect(wrapper.emitted("cancel")).toHaveLength(1);
  });

  it("locks both buttons while busy so an in-flight action can't be re-fired", async () => {
    const wrapper = mount(ConfirmDialog, {
      props: { open: true, title: "Delete?", busy: true },
      attachTo: document.body,
    });
    (el("confirm-cancel") as HTMLButtonElement).click();
    (el("confirm-accept") as HTMLButtonElement).click();
    await wrapper.vm.$nextTick();
    // Disabled buttons don't dispatch clicks, so neither event fires.
    expect(wrapper.emitted("cancel")).toBeUndefined();
    expect(wrapper.emitted("confirm")).toBeUndefined();
  });

  it("renders a caller's own itemisation in the body", () => {
    mount(ConfirmDialog, {
      props: { open: true, title: "Rent this GPU?" },
      slots: { default: "<p>L40S · Secure cloud</p>" },
      attachTo: document.body,
    });
    expect(el("confirm-dialog")?.textContent).toContain("L40S · Secure cloud");
  });

  /*
   * A destructive question is a plain question with a danger button. The
   * browser's copy of this dialog could arm its confirm behind a typed phrase,
   * which nothing asks for any more and which we do not bring across.
   */
  it("never asks anyone to type a phrase", () => {
    expect(source).not.toContain("typedPhrase");
    expect(source).not.toContain("to continue");
  });

  it("marks the destructive confirm without relying on the label", () => {
    mount(ConfirmDialog, {
      props: {
        open: true,
        title: "Delete forever?",
        danger: true,
        confirmLabel: "Delete forever",
      },
      attachTo: document.body,
    });
    expect(el("confirm-accept")?.dataset.danger).toBe("true");
  });

  /*
   * A long confirmation label wraps rather than being clipped: the buttons
   * carry a minimum height and vertical padding, never a fixed height.
   */
  it("lets a long confirmation label grow vertically instead of clipping", () => {
    const rules = source.slice(source.indexOf("<style"));
    expect(rules).toContain("min-height:");
    expect(rules).not.toMatch(/\n\s+height:/);
  });

  describe("a one-line answer", () => {
    it("offers a text field and carries its value both ways", async () => {
      const wrapper = mount(ConfirmDialog, {
        props: {
          open: true,
          title: "Rename album",
          inputLabel: "Album name",
          modelValue: "Holiday",
        },
        attachTo: document.body,
      });
      const input = el("dialog-text") as HTMLInputElement;
      expect(input.value).toBe("Holiday");
      expect(el("confirm-dialog")?.textContent).toContain("Album name");
      input.value = "Holiday 2026";
      input.dispatchEvent(new Event("input"));
      await wrapper.vm.$nextTick();
      expect(wrapper.emitted("update:modelValue")).toEqual([["Holiday 2026"]]);
    });

    it("confirms on Enter so the field is not a dead end", async () => {
      const wrapper = mount(ConfirmDialog, {
        props: { open: true, title: "Rename album", modelValue: "Holiday" },
        attachTo: document.body,
      });
      (el("dialog-text") as HTMLInputElement).dispatchEvent(
        new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
      );
      await wrapper.vm.$nextTick();
      expect(wrapper.emitted("confirm")).toHaveLength(1);
    });

    it("has no field at all when no value was handed in", () => {
      mount(ConfirmDialog, {
        props: { open: true, title: "Delete?" },
        attachTo: document.body,
      });
      expect(el("dialog-text")).toBeNull();
    });
  });

  describe("a question with more than two answers", () => {
    it("stacks the choices and reports the one that was picked", async () => {
      const wrapper = mount(ConfirmDialog, {
        props: {
          open: true,
          title: "This print was made by a sequence",
          choices: [
            { id: "first", label: "Use the first shot's prompt" },
            { id: "discard", label: "Start over", danger: true },
          ],
        },
        attachTo: document.body,
      });
      expect(el("dialog-choice-first")?.textContent?.trim()).toBe(
        "Use the first shot's prompt",
      );
      expect(el("dialog-choice-discard")?.dataset.danger).toBe("true");
      (el("dialog-choice-first") as HTMLButtonElement).click();
      await wrapper.vm.$nextTick();
      expect(wrapper.emitted("choose")).toEqual([["first"]]);
    });

    it("drops the confirm button, because every answer is one of the choices", () => {
      mount(ConfirmDialog, {
        props: {
          open: true,
          title: "Pick one",
          choices: [{ id: "a", label: "A" }],
        },
        attachTo: document.body,
      });
      expect(el("confirm-accept")).toBeNull();
      expect(el("confirm-cancel")).not.toBeNull();
    });
  });
});
