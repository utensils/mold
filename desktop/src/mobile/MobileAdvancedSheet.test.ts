import { flushPromises, mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import { nextTick } from "vue";
import MobileAdvancedSheet from "./MobileAdvancedSheet.vue";

describe("MobileAdvancedSheet", () => {
  it("keeps hosted advanced controls mounted whether the sheet is open or closed", async () => {
    const wrapper = mount(MobileAdvancedSheet, {
      props: { open: false, count: 0 },
      slots: { default: '<button data-test="hosted-control">Scheduler</button>' },
    });

    // Controls stay in the DOM while collapsed so the Generate flow can still
    // read every advanced field and validity binding.
    expect(wrapper.find("[data-test='hosted-control']").exists()).toBe(true);
    expect(wrapper.get("[data-test='mobile-advanced-sheet']").classes()).not.toContain("is-open");

    await wrapper.setProps({ open: true });
    expect(wrapper.get("[data-test='mobile-advanced-sheet']").classes()).toContain("is-open");
    expect(wrapper.find("[data-test='hosted-control']").exists()).toBe(true);
  });

  it("shows the active-count badge only when at least one advanced setting is set", async () => {
    const wrapper = mount(MobileAdvancedSheet, {
      props: { open: true, count: 0 },
    });

    expect(wrapper.find("[data-test='mobile-advanced-count']").exists()).toBe(false);

    await wrapper.setProps({ count: 3 });
    expect(wrapper.get("[data-test='mobile-advanced-count']").text()).toBe("3");
  });

  it("emits close and reset for the header control and the reset action", async () => {
    const wrapper = mount(MobileAdvancedSheet, {
      props: { open: true, count: 2 },
    });

    await wrapper.get("[data-test='mobile-advanced-close']").trigger("click");
    await wrapper.get("[data-test='mobile-advanced-reset']").trigger("click");

    expect(wrapper.emitted("close")).toHaveLength(1);
    expect(wrapper.emitted("reset")).toHaveLength(1);
  });
  it("takes focus on open and returns it to the trigger after Escape", async () => {
    const trigger = document.createElement("button");
    document.body.append(trigger);
    trigger.focus();
    const wrapper = mount(MobileAdvancedSheet, {
      attachTo: document.body,
      props: { open: false, count: 0 },
      slots: { default: '<input aria-label="Detail" />' },
    });
    expect(wrapper.get("[data-test=mobile-advanced-sheet]").attributes("inert")).toBeDefined();
    await wrapper.setProps({ open: true });
    await flushPromises();
    // The panel takes focus, not the overlay: the overlay is the scrim's host.
    expect(document.activeElement).toBe(wrapper.get(".mobile-sheet-panel").element);
    await wrapper.get("[data-test=mobile-advanced-sheet]").trigger("keydown", { key: "Escape" });
    expect(wrapper.emitted("close")).toHaveLength(1);
    await wrapper.setProps({ open: false });
    expect(document.activeElement).toBe(trigger);
    wrapper.unmount();
    trigger.remove();
  });

  it("is a bottom sheet: grabber, scrim, and a text Reset · title · Done header", async () => {
    const wrapper = mount(MobileAdvancedSheet, { props: { open: true, count: 0 } });
    await flushPromises();

    expect(wrapper.get("[data-test='mobile-advanced-sheet']").attributes("aria-modal")).toBe(
      "true",
    );
    expect(wrapper.find(".mobile-sheet-grabber").exists()).toBe(true);
    expect(wrapper.get(".mobile-sheet-title").text()).toBe("More settings");

    const head = wrapper.get(".mobile-advanced-sheet-head").element;
    const reset = wrapper.get("[data-test='mobile-advanced-reset']");
    const done = wrapper.get("[data-test='mobile-advanced-close']");
    // Both are text controls in the header — Reset leading, Done trailing —
    // never the 44px circular glyph the full-screen surface used to carry.
    expect(reset.element.closest(".mobile-advanced-sheet-head")).toBe(head);
    expect(done.element.closest(".mobile-advanced-sheet-head")).toBe(head);
    expect(reset.text()).toBe("Reset");
    expect(done.text()).toBe("Done");
    expect(done.classes()).toContain("mobile-sheet-action");

    await wrapper.get("[data-test='mobile-advanced-sheet-scrim']").trigger("click");
    expect(wrapper.emitted("close")).toHaveLength(1);
    wrapper.unmount();
  });

  it("wraps a forward Tab from the panel itself back to the first control", async () => {
    // The panel is the focus target on open, so the very first Tab arrives with
    // the panel focused. Without this the key fell through and iOS moved focus
    // to browser chrome behind the sheet.
    const wrapper = mount(MobileAdvancedSheet, {
      attachTo: document.body,
      props: { open: true, count: 0 },
      slots: { default: '<input aria-label="Detail" />' },
    });
    await flushPromises();
    expect(document.activeElement).toBe(wrapper.get(".mobile-sheet-panel").element);

    await wrapper.get("[data-test=mobile-advanced-sheet]").trigger("keydown", { key: "Tab" });
    expect(document.activeElement).toBe(wrapper.get("[data-test='mobile-advanced-reset']").element);
    wrapper.unmount();
  });

  it("leaves focus alone while another sheet stands above it", async () => {
    const { createOverlayToken, popOverlay, pushOverlay } = await import("@ui/lib/overlayStack");
    const outside = document.createElement("button");
    document.body.append(outside);
    const wrapper = mount(MobileAdvancedSheet, {
      attachTo: document.body,
      props: { open: false, count: 0 },
    });
    outside.focus();
    void wrapper.setProps({ open: true });
    await nextTick();
    // A sheet rising over this one while the focus watch waits its tick for
    // the panel to render — that tick is the whole window.
    const above = createOverlayToken("test-sheet-above");
    pushOverlay(above);
    await flushPromises();
    // A sheet opened underneath must not pull focus out of the one on top.
    expect(document.activeElement).toBe(outside);
    popOverlay(above);
    wrapper.unmount();
    outside.remove();
  });
});
