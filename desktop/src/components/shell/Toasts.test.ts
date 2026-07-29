import { beforeEach, describe, expect, it, vi } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import Toasts from "./Toasts.vue";
import { useToastStore } from "../../stores/toasts";

beforeEach(() => {
  setActivePinia(createPinia());
  vi.useFakeTimers();
});

describe("Toasts a11y roles", () => {
  it("announces info toasts politely (role=status) and errors assertively (role=alert)", async () => {
    const wrapper = mount(Toasts);
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

  it("anchors the shelf to the top of the window, clear of the title bar", async () => {
    const wrapper = mount(Toasts);
    const classes = wrapper.get("[aria-label='Notifications']").classes();

    expect(classes.some((c) => c.startsWith("top-"))).toBe(true);
    expect(classes.some((c) => c.startsWith("bottom-"))).toBe(false);
  });

  it("renders the newest toast first so it slides in above the older ones", async () => {
    const wrapper = mount(Toasts);
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
    const wrapper = mount(Toasts);
    const toasts = useToastStore();
    toasts.push("Hi");
    await wrapper.vm.$nextTick();

    expect(wrapper.find("[aria-label='Notifications']").exists()).toBe(true);
    await wrapper.get("[role='status']").trigger("click");
    expect(toasts.items.length).toBe(0);
  });

  it("renders an error as a compact action card with hierarchy and an explicit close control", async () => {
    const wrapper = mount(Toasts);
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
    expect(toast.classes()).toContain("w-[min(25rem,calc(100vw-2rem))]");

    await toast.get("[data-test='toast-action']").trigger("click");
    expect(openMachines).toHaveBeenCalledOnce();
    expect(toasts.items).toHaveLength(0);
  });

  it("dismisses from the explicit close control without running the body action", async () => {
    const wrapper = mount(Toasts);
    const toasts = useToastStore();
    const onClick = vi.fn();
    toasts.push("Generated", "info", { onClick });
    await wrapper.vm.$nextTick();

    await wrapper.get("[data-test='toast-dismiss']").trigger("click");

    expect(onClick).not.toHaveBeenCalled();
    expect(toasts.items).toHaveLength(0);
  });
});
