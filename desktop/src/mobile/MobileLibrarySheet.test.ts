import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { resetOverlayStackForTests } from "@ui/lib/overlayStack";
import MobileLibrarySheet from "./MobileLibrarySheet.vue";

afterEach(() => {
  vi.restoreAllMocks();
  resetOverlayStackForTests();
});

describe("mobile library sheet focus", () => {
  it("keeps read-first focus inside the sheet and restores its trigger", async () => {
    vi.spyOn(HTMLElement.prototype, "getClientRects").mockReturnValue([
      {},
    ] as unknown as DOMRectList);
    const trigger = document.createElement("button");
    document.body.append(trigger);
    trigger.focus();
    const wrapper = mount(MobileLibrarySheet, {
      attachTo: document.body,
      props: { open: false, title: "Work details", focusFirstControl: false },
      slots: { default: '<button data-test="action">Use settings</button>' },
    });
    expect(wrapper.attributes("inert")).toBeDefined();
    await wrapper.setProps({ open: true });
    await flushPromises();
    const panel = wrapper.get(".mobile-library-sheet-panel");
    expect(document.activeElement).toBe(panel.element);
    await panel.trigger("keydown", { key: "Tab", shiftKey: true });
    expect(document.activeElement?.hasAttribute("data-sheet-close")).toBe(true);
    await wrapper.get("[data-test=mobile-library-sheet-done]").trigger("keydown", { key: "Tab" });
    expect(document.activeElement).toBe(wrapper.get("[data-test=action]").element);
    await wrapper.setProps({ open: false });
    expect(document.activeElement).toBe(trigger);
    wrapper.unmount();
    trigger.remove();
  });

  it("lets only the top sheet consume Escape", async () => {
    const lower = mount(MobileLibrarySheet, { props: { open: true, title: "Image" } });
    const upper = mount(MobileLibrarySheet, { props: { open: true, title: "Tags" } });
    await flushPromises();
    await lower.trigger("keydown", { key: "Escape" });
    expect(lower.emitted("close")).toBeUndefined();
    await upper.trigger("keydown", { key: "Escape" });
    expect(upper.emitted("close")).toHaveLength(1);
    upper.unmount();
    lower.unmount();
  });
});
