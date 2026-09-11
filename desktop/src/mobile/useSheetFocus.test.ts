import { mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { defineComponent, h, nextTick, ref } from "vue";
import { createOverlayToken, popOverlay, pushOverlay } from "@ui/lib/overlayStack";
import { useSheetFocus } from "./useSheetFocus";

/** A minimal sheet: a focusable panel with two controls inside it. */
const Sheet = defineComponent({
  props: {
    open: { type: Boolean, required: true },
    isTop: { type: Function, default: () => true },
    focusFirstControl: { type: Boolean, default: false },
  },
  emits: ["close"],
  setup(props, { emit }) {
    const panel = ref<HTMLElement | null>(null);
    const { onKeydown } = useSheetFocus({
      panel,
      open: () => props.open,
      isTop: () => props.isTop() as boolean,
      onClose: () => emit("close"),
      firstControl: () =>
        props.focusFirstControl
          ? (panel.value?.querySelector<HTMLElement>("button, input") ?? null)
          : null,
    });
    return () =>
      h("div", { onKeydown }, [
        h("div", { ref: panel, tabindex: "-1", class: "panel" }, [
          h("button", { class: "first" }, "First"),
          h("button", { class: "last" }, "Last"),
        ]),
      ]);
  },
});

function mountSheet(props: Record<string, unknown> = {}) {
  return mount(Sheet, { attachTo: document.body, props: { open: false, ...props } });
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("useSheetFocus", () => {
  it("takes focus on open and gives it back to whatever had it", async () => {
    const trigger = document.createElement("button");
    document.body.append(trigger);
    trigger.focus();

    const wrapper = mountSheet();
    await wrapper.setProps({ open: true });
    await nextTick();
    await nextTick();
    expect(document.activeElement).toBe(wrapper.get(".panel").element);

    await wrapper.setProps({ open: false });
    expect(document.activeElement).toBe(trigger);
    wrapper.unmount();
    trigger.remove();
  });

  it("focuses the first control when the sheet is there to be edited", async () => {
    const wrapper = mountSheet({ focusFirstControl: true });
    await wrapper.setProps({ open: true });
    await nextTick();
    await nextTick();
    expect(document.activeElement).toBe(wrapper.get(".first").element);
    wrapper.unmount();
  });

  it("leaves focus alone while a sheet stands above it", async () => {
    const outside = document.createElement("button");
    document.body.append(outside);
    const above = createOverlayToken("above");
    pushOverlay(above);

    const wrapper = mountSheet({ isTop: () => false });
    outside.focus();
    await wrapper.setProps({ open: true });
    await nextTick();
    await nextTick();
    expect(document.activeElement).toBe(outside);

    popOverlay(above);
    wrapper.unmount();
    outside.remove();
  });

  it("closes on Escape, and only for the sheet on top", async () => {
    const wrapper = mountSheet({ open: true });
    await nextTick();

    await wrapper.trigger("keydown", { key: "Escape" });
    expect(wrapper.emitted("close")).toHaveLength(1);

    await wrapper.setProps({ isTop: () => false });
    await wrapper.trigger("keydown", { key: "Escape" });
    // Escape over a sheet above this one belongs to that sheet.
    expect(wrapper.emitted("close")).toHaveLength(1);
    wrapper.unmount();
  });

  it("keeps Tab inside the sheet in both directions", async () => {
    const wrapper = mountSheet({ open: true });
    await nextTick();
    await nextTick();

    // Forward from the panel itself — the state every sheet opens in.
    await wrapper.trigger("keydown", { key: "Tab" });
    expect(document.activeElement).toBe(wrapper.get(".first").element);

    (wrapper.get(".last").element as HTMLElement).focus();
    await wrapper.trigger("keydown", { key: "Tab" });
    expect(document.activeElement).toBe(wrapper.get(".first").element);

    (wrapper.get(".first").element as HTMLElement).focus();
    await wrapper.trigger("keydown", { key: "Tab", shiftKey: true });
    expect(document.activeElement).toBe(wrapper.get(".last").element);
    wrapper.unmount();
  });

  it("does nothing at all while closed", async () => {
    const outside = document.createElement("button");
    document.body.append(outside);
    outside.focus();

    const wrapper = mountSheet({ open: false });
    await wrapper.trigger("keydown", { key: "Escape" });
    await wrapper.trigger("keydown", { key: "Tab" });
    expect(wrapper.emitted("close")).toBeUndefined();
    expect(document.activeElement).toBe(outside);
    wrapper.unmount();
    outside.remove();
  });

  it("forgets the element to restore when the sheet unmounts while open", async () => {
    const trigger = document.createElement("button");
    document.body.append(trigger);
    trigger.focus();

    const wrapper = mountSheet({ open: true });
    await nextTick();
    const spy = vi.spyOn(trigger, "focus");
    wrapper.unmount();
    // A route change over an open sheet must not yank focus afterwards.
    expect(spy).not.toHaveBeenCalled();
    trigger.remove();
  });
});
