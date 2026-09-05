import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { SEVERITY_MARKS } from "@ui/lib/notificationSeverity";
import Toasts from "./Toasts.vue";
import { useToastStore } from "../../stores/toasts";

beforeEach(() => {
  setActivePinia(createPinia());
  vi.useFakeTimers();
});

/**
 * The shelf teleports to <body> (see Toasts.vue). Stub the teleport so these
 * assertions keep reading the component's own tree; the teleport itself is
 * pinned by its own test below.
 */
function mountToasts() {
  return mount(Toasts, { global: { stubs: { teleport: true } } });
}

describe("Toasts a11y roles", () => {
  it("announces info toasts politely (role=status) and errors assertively (role=alert)", async () => {
    const wrapper = mountToasts();
    const toasts = useToastStore();
    toasts.push("Saved to Gallery");
    toasts.push("Something broke", "error");
    await wrapper.vm.$nextTick();

    const info = wrapper.get("[role='status']");
    expect(info.get("[data-test='toast-title']").text()).toBe("Saved to Gallery");
    expect(info.attributes("aria-live")).toBe("polite");

    const error = wrapper.get("[role='alert']");
    expect(error.get("[data-test='toast-title']").text()).toBe("Something broke");
    expect(error.attributes("aria-live")).toBe("assertive");
  });

  it("anchors the shelf bottom-right, above the status bar", async () => {
    const wrapper = mountToasts();
    const classes = wrapper.get("[aria-label='Notifications']").classes();

    expect(classes.some((c) => c.startsWith("bottom-[calc(var(--mold-shell-statusbar-h)"))).toBe(
      true,
    );
    expect(classes.some((c) => c.startsWith("top-"))).toBe(false);
  });

  it("renders the newest toast first so it slides in above the older ones", async () => {
    const wrapper = mountToasts();
    const toasts = useToastStore();
    toasts.push("Older");
    toasts.push("Newer");
    await wrapper.vm.$nextTick();

    const rendered = wrapper
      .findAll("[role='status']")
      .map((row) => row.get("[data-test='toast-title']").text());
    expect(rendered).toEqual(["Newer", "Older"]);
  });

  it("labels the toast region and dismisses on click", async () => {
    const wrapper = mountToasts();
    const toasts = useToastStore();
    toasts.push("Hi");
    await wrapper.vm.$nextTick();

    expect(wrapper.find("[aria-label='Notifications']").exists()).toBe(true);
    await wrapper.get("[role='status']").trigger("click");
    expect(toasts.items.length).toBe(0);
  });

  it("renders an error as a compact action card with hierarchy and an explicit close control", async () => {
    const wrapper = mountToasts();
    const toasts = useToastStore();
    const openMachines = vi.fn();
    toasts.push("Can't reach bender-7680", "error", {
      description: "It stays listed for reconnect.",
      action: { label: "Open Machines", run: openMachines },
      sticky: true,
    });
    await wrapper.vm.$nextTick();

    const toast = wrapper.get("[role='alert']");
    expect(toast.get("[data-test='toast-status-icon']").attributes("aria-hidden")).toBe("true");
    expect(toast.get("[data-test='toast-title']").text()).toBe("Can't reach bender-7680");
    expect(toast.get("[data-test='toast-description']").text()).toBe(
      "It stays listed for reconnect.",
    );
    expect(toast.classes()).toContain("w-80");

    await toast.get("[data-test='toast-action']").trigger("click");
    expect(openMachines).toHaveBeenCalledOnce();
    expect(toasts.items).toHaveLength(0);
  });

  it("dismisses from the explicit close control without running the body action", async () => {
    const wrapper = mountToasts();
    const toasts = useToastStore();
    const onClick = vi.fn();
    toasts.push("Generated", "info", { onClick });
    await wrapper.vm.$nextTick();

    await wrapper.get("[data-test='toast-dismiss']").trigger("click");

    expect(onClick).not.toHaveBeenCalled();
    expect(toasts.items).toHaveLength(0);
  });
});

describe("Toasts severity tones", () => {
  it("tints success green, warning yellow, and error red", async () => {
    const wrapper = mountToasts();
    const toasts = useToastStore();
    toasts.push("Reconnected to plato", "success");
    toasts.push("Can't reach plato", "warning");
    toasts.push("Generation failed", "error");
    await wrapper.vm.$nextTick();

    const chips = wrapper.findAll("[data-test='toast-status-icon']");
    const byKind = Object.fromEntries(
      chips.map((chip) => [chip.attributes("data-kind"), chip.attributes("style") ?? ""]),
    );
    // Every hue resolves through the shared table rather than a local class.
    expect(byKind.success).toContain(SEVERITY_MARKS.success.color);
    expect(byKind.warning).toContain(SEVERITY_MARKS.warning.color);
    expect(byKind.error).toContain(SEVERITY_MARKS.error.color);
    // An ordinary notice is green too — only warnings and errors stand out.
    toasts.push("Queued", "info");
    await wrapper.vm.$nextTick();
    const info = wrapper
      .findAll("[data-test='toast-status-icon']")
      .find((chip) => chip.attributes("data-kind") === "info");
    expect(info?.attributes("style")).toContain(SEVERITY_MARKS.info.color);
  });

  it("names the severity for screen readers, never color alone", async () => {
    const wrapper = mountToasts();
    const toasts = useToastStore();
    toasts.push("Can't reach plato", "warning");
    await wrapper.vm.$nextTick();

    expect(wrapper.get(".sr-only").text()).toBe("Warning");
  });

  it("marks warning and error with different glyphs, not just different hues", async () => {
    const wrapper = mountToasts();
    const toasts = useToastStore();
    toasts.push("Can't reach plato", "warning");
    toasts.push("Generation failed", "error");
    await wrapper.vm.$nextTick();

    const glyphs = Object.fromEntries(
      wrapper
        .findAll("[data-test='toast-status-icon']")
        .map((chip) => [chip.attributes("data-kind"), chip.text()]),
    );
    expect(glyphs.warning).toBe("!");
    expect(glyphs.error).toBe("✕");
  });

  /**
   * A `fixed` layer resolves against the nearest ancestor with a transform,
   * filter or container-type, not the viewport. The shelf lived inside the
   * app frame and was correct only because nothing there had grown one yet —
   * ContextMenu and Tooltip already teleport for exactly this reason.
   */
  it("teleports the shelf to <body> so no ancestor can ever capture it", async () => {
    document.body.innerHTML = "";
    const wrapper = mount(Toasts);
    useToastStore().push("Saved to My images");
    await wrapper.vm.$nextTick();

    const shelf = document.body.querySelector("[aria-label='Notifications']");
    expect(shelf).not.toBeNull();
    expect(shelf!.parentElement).toBe(document.body);
    // Nothing of the shelf is left behind in the component's own tree.
    expect(wrapper.find("[aria-label='Notifications']").exists()).toBe(false);
    wrapper.unmount();
    document.body.innerHTML = "";
  });
});
