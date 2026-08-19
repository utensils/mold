import { beforeEach, describe, expect, it } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import NotificationsCenter from "./NotificationsCenter.vue";
import { useNotificationsStore } from "../stores/notifications";

beforeEach(() => {
  setActivePinia(createPinia());
  document.body.innerHTML = "";
});

function mountCenter() {
  return mount(NotificationsCenter, { attachTo: document.body });
}

describe("NotificationsCenter", () => {
  it("badges the bell with the unread count", async () => {
    const store = useNotificationsStore();
    store.record({ kind: "error", text: "boom", atMs: 1 });
    store.record({ kind: "info", text: "queued", atMs: 2 });

    const wrapper = mountCenter();
    expect(wrapper.get('[data-test="notifications-unread"]').text()).toBe("2");
    wrapper.unmount();
  });

  it("opens to the full untruncated message and marks everything read", async () => {
    const store = useNotificationsStore();
    const long = "Server error: ".padEnd(300, "x");
    store.record({
      kind: "error",
      text: "Generation failed",
      description: long,
      hostLabel: "plato",
      atMs: 1,
    });

    const wrapper = mountCenter();
    await wrapper.get('[data-test="notifications-bell"]').trigger("click");
    await flushPromises();

    const panel = document.body.querySelector(
      '[data-test="notifications-panel"]',
    );
    expect(panel?.textContent).toContain("Generation failed");
    expect(panel?.textContent).toContain(long);
    expect(panel?.textContent).toContain("plato");
    expect(store.unreadCount).toBe(0);
    expect(wrapper.find('[data-test="notifications-unread"]').exists()).toBe(
      false,
    );
    wrapper.unmount();
  });

  it("shows a repeat count for collapsed duplicates and clears on demand", async () => {
    const store = useNotificationsStore();
    store.record({ kind: "error", text: "Connection lost", atMs: 1 });
    store.record({ kind: "error", text: "Connection lost", atMs: 2 });

    const wrapper = mountCenter();
    await wrapper.get('[data-test="notifications-bell"]').trigger("click");
    await flushPromises();

    const panel = document.body.querySelector(
      '[data-test="notifications-panel"]',
    );
    expect(panel?.textContent).toContain("×2");

    const clearButton = document.body.querySelector<HTMLButtonElement>(
      '[data-test="notifications-clear"]',
    );
    clearButton?.click();
    await flushPromises();
    expect(store.entries).toHaveLength(0);
    expect(
      document.body.querySelector('[data-test="notifications-panel"]')
        ?.textContent,
    ).toContain("No notifications");
    wrapper.unmount();
  });
});

describe("NotificationsCenter severity colors", () => {
  it("dots each entry green / yellow / red by severity", async () => {
    const store = useNotificationsStore();
    store.record({ kind: "success", text: "Reconnected to plato", atMs: 3 });
    store.record({ kind: "warning", text: "Can't reach plato", atMs: 2 });
    store.record({ kind: "error", text: "Generation failed", atMs: 1 });

    const wrapper = mountCenter();
    await wrapper.get('[data-test="notifications-bell"]').trigger("click");
    await flushPromises();

    const dots = [
      ...document.body.querySelectorAll<HTMLElement>(
        '[data-test="notifications-dot"]',
      ),
    ].map((dot) => dot.style.background);
    // Newest first: error, warning, success.
    expect(dots).toEqual(["var(--stop)", "var(--warning)", "var(--success)"]);
    wrapper.unmount();
  });

  it("names the severity in assistive text, never color alone", async () => {
    const store = useNotificationsStore();
    store.record({ kind: "warning", text: "Can't reach plato", atMs: 1 });

    const wrapper = mountCenter();
    await wrapper.get('[data-test="notifications-bell"]').trigger("click");
    await flushPromises();

    expect(
      document.body.querySelector('[data-test="notifications-panel"]')
        ?.textContent,
    ).toContain("Warning");
    wrapper.unmount();
  });

  it("colors the unread badge with the worst unread severity", async () => {
    const store = useNotificationsStore();
    store.record({ kind: "success", text: "Reconnected to plato", atMs: 1 });

    const wrapper = mountCenter();
    expect(
      wrapper.get('[data-test="notifications-unread"]').attributes("style"),
    ).toContain("var(--success)");

    store.record({ kind: "error", text: "Generation failed", atMs: 2 });
    await flushPromises();
    expect(
      wrapper.get('[data-test="notifications-unread"]').attributes("style"),
    ).toContain("var(--stop)");
    wrapper.unmount();
  });
});
