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
    expect(info.text()).toBe("Saved to Gallery");
    expect(info.attributes("aria-live")).toBe("polite");

    const error = wrapper.get("[role='alert']");
    expect(error.text()).toBe("Something broke");
    expect(error.attributes("aria-live")).toBe("assertive");
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
});
