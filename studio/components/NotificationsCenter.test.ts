import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import NotificationsCenter from "./NotificationsCenter.vue";
import { useNotificationsStore } from "../stores/notifications";

beforeEach(() => {
  setActivePinia(createPinia());
  document.body.innerHTML = "";
});

afterEach(() => {
  // The copy tests stub navigator; leaking that into a later file would make
  // failures depend on test order.
  vi.unstubAllGlobals();
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

  it("runs a recovery action from the message and reports its progress", async () => {
    let resolve!: () => void;
    const run = vi.fn(
      () =>
        new Promise<void>((done) => {
          resolve = done;
        }),
    );
    const store = useNotificationsStore();
    store.record({
      kind: "error",
      text: "flux-dev:q4 failed to download",
      atMs: 1,
      action: {
        label: "Retry",
        pendingLabel: "Retrying…",
        doneLabel: "Queued",
        run,
      },
    });

    const wrapper = mountCenter();
    await wrapper.get('[data-test="notifications-bell"]').trigger("click");
    await flushPromises();
    const action = document.body.querySelector<HTMLButtonElement>(
      '[data-test="notifications-action"]',
    )!;

    expect(
      document.body
        .querySelector('[data-test="notifications-panel"]')
        ?.textContent?.match(/—/g),
    ).toHaveLength(1);
    expect(action.textContent?.trim()).toBe("Retry");
    action.click();
    await flushPromises();
    expect(run).toHaveBeenCalledOnce();
    expect(action.textContent?.trim()).toBe("Retrying…");
    expect(action.disabled).toBe(true);

    resolve();
    await flushPromises();
    expect(action.textContent?.trim()).toBe("Queued");
    expect(action.disabled).toBe(true);
    wrapper.unmount();
  });

  it("leaves a failed recovery action available to try again", async () => {
    const run = vi.fn().mockRejectedValue(new Error("offline"));
    const store = useNotificationsStore();
    store.record({
      kind: "error",
      text: "Download failed",
      atMs: 1,
      action: { label: "Retry", run },
    });

    const wrapper = mountCenter();
    await wrapper.get('[data-test="notifications-bell"]').trigger("click");
    await flushPromises();
    const action = document.body.querySelector<HTMLButtonElement>(
      '[data-test="notifications-action"]',
    )!;
    action.click();
    await flushPromises();

    expect(action.textContent?.trim()).toBe("Retry failed");
    expect(action.disabled).toBe(false);
    action.click();
    await flushPromises();
    expect(run).toHaveBeenCalledTimes(2);
    wrapper.unmount();
  });

  it("lets separate failed downloads retry concurrently", async () => {
    const pending = () => new Promise<void>(() => {});
    const first = vi.fn(pending);
    const second = vi.fn(pending);
    const store = useNotificationsStore();
    store.record({
      kind: "error",
      text: "First download failed",
      atMs: 1,
      action: { label: "Retry", run: first },
    });
    store.record({
      kind: "error",
      text: "Second download failed",
      atMs: 2,
      action: { label: "Retry", run: second },
    });

    const wrapper = mountCenter();
    await wrapper.get('[data-test="notifications-bell"]').trigger("click");
    await flushPromises();
    const actions = [
      ...document.body.querySelectorAll<HTMLButtonElement>(
        '[data-test="notifications-action"]',
      ),
    ];

    actions[0]!.click();
    await flushPromises();
    expect(actions[0]!.disabled).toBe(true);
    expect(actions[1]!.disabled).toBe(false);

    actions[1]!.click();
    await flushPromises();
    expect(first).toHaveBeenCalledOnce();
    expect(second).toHaveBeenCalledOnce();
    expect(actions.every((action) => action.disabled)).toBe(true);
    wrapper.unmount();
  });

  it("enables a new retry when the same download fails again", async () => {
    const first = vi.fn().mockResolvedValue(undefined);
    const second = vi.fn().mockResolvedValue(undefined);
    const store = useNotificationsStore();
    store.record({
      kind: "error",
      text: "Download failed",
      atMs: 1,
      action: { label: "Retry", doneLabel: "Queued", run: first },
    });

    const wrapper = mountCenter();
    await wrapper.get('[data-test="notifications-bell"]').trigger("click");
    await flushPromises();
    const action = document.body.querySelector<HTMLButtonElement>(
      '[data-test="notifications-action"]',
    )!;
    action.click();
    await flushPromises();
    expect(action.textContent?.trim()).toBe("Queued");
    expect(action.disabled).toBe(true);

    store.record({
      kind: "error",
      text: "Download failed",
      atMs: 2,
      action: { label: "Retry", doneLabel: "Queued", run: second },
    });
    await flushPromises();

    expect(store.entries).toHaveLength(1);
    expect(store.entries[0]!.repeat).toBe(2);
    expect(action.textContent?.trim()).toBe("Retry");
    expect(action.disabled).toBe(false);
    action.click();
    await flushPromises();
    expect(first).toHaveBeenCalledOnce();
    expect(second).toHaveBeenCalledOnce();
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
    ];
    // Newest first: error, warning, success.
    expect(dots.map((dot) => dot.style.color)).toEqual([
      "var(--stop)",
      "var(--warning)",
      "var(--success)",
    ]);
    // Each severity also carries its own mark, so color is never the only cue.
    expect(dots.map((dot) => dot.textContent?.trim())).toEqual(["✕", "!", "✓"]);
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
    store.record({
      kind: "info",
      text: "Generated — saved to Library",
      atMs: 1,
    });

    const wrapper = mountCenter();
    // Only notices waiting: the bell reads green, not red.
    expect(
      wrapper.get('[data-test="notifications-unread"]').attributes("style"),
    ).toContain("var(--success)");

    store.record({ kind: "warning", text: "Can't reach plato", atMs: 2 });
    await flushPromises();
    expect(
      wrapper.get('[data-test="notifications-unread"]').attributes("style"),
    ).toContain("var(--warning)");

    store.record({ kind: "error", text: "Generation failed", atMs: 2 });
    await flushPromises();
    const badge = wrapper.get('[data-test="notifications-unread"]');
    expect(badge.attributes("style")).toContain("var(--stop)");
    // The count itself sits on that fill with the per-theme status ink.
    expect(badge.attributes("style")).toContain("var(--on-status)");
    wrapper.unmount();
  });
});

describe("NotificationsCenter copying", () => {
  it("copies the full message, body, and origin line to the clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText } });
    const store = useNotificationsStore();
    const long = "Server error: ".padEnd(400, "x");
    store.record({
      kind: "error",
      text: "Generation failed",
      description: long,
      hostLabel: "plato",
      atMs: Date.UTC(2026, 0, 2, 9, 19),
    });

    const wrapper = mountCenter();
    await wrapper.get('[data-test="notifications-bell"]').trigger("click");
    await flushPromises();

    const copy = document.body.querySelector<HTMLButtonElement>(
      '[data-test="notifications-copy"]',
    );
    expect(copy?.textContent?.trim()).toBe("Copy");
    copy?.click();
    await flushPromises();

    expect(writeText).toHaveBeenCalledTimes(1);
    const copied = writeText.mock.calls[0]![0] as string;
    expect(copied.startsWith("Generation failed\n")).toBe(true);
    expect(copied).toContain(long);
    expect(copied).toContain("plato · ");
    expect(
      document.body
        .querySelector('[data-test="notifications-copy"]')
        ?.textContent?.trim(),
    ).toBe("Copied");
    wrapper.unmount();
  });

  it("announces every copy, including a repeat of the same outcome", async () => {
    // A live region only speaks when its text changes, and pressing Copy twice
    // is exactly what someone unsure the first one worked does.
    vi.stubGlobal("navigator", {
      clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
    const store = useNotificationsStore();
    store.record({ kind: "info", text: "Queued", atMs: 1 });

    const wrapper = mountCenter();
    await wrapper.get('[data-test="notifications-bell"]').trigger("click");
    await flushPromises();
    const region = () =>
      document.body.querySelector('[data-test="notifications-copy-status"]');
    const copy = () =>
      document.body.querySelector<HTMLButtonElement>(
        '[data-test="notifications-copy"]',
      );

    copy()?.click();
    await flushPromises();
    expect(region()?.textContent?.trim()).toBe(
      "Notification copied to the clipboard",
    );

    // The region is cleared before the identical message is written back, so
    // the second press is a real DOM change rather than silence.
    const cleared: string[] = [];
    const observer = new MutationObserver(() =>
      cleared.push(region()?.textContent?.trim() ?? ""),
    );
    observer.observe(region()!, {
      childList: true,
      characterData: true,
      subtree: true,
    });
    copy()?.click();
    await flushPromises();
    observer.disconnect();
    expect(cleared).toContain("");
    expect(region()?.textContent?.trim()).toBe(
      "Notification copied to the clipboard",
    );
    wrapper.unmount();
  });

  it("says so when the copy could not happen instead of claiming success", async () => {
    vi.stubGlobal("navigator", {});
    Object.assign(document, { execCommand: vi.fn().mockReturnValue(false) });
    const store = useNotificationsStore();
    store.record({ kind: "warning", text: "Can't reach plato", atMs: 1 });

    const wrapper = mountCenter();
    await wrapper.get('[data-test="notifications-bell"]').trigger("click");
    await flushPromises();

    document.body
      .querySelector<HTMLButtonElement>('[data-test="notifications-copy"]')
      ?.click();
    await flushPromises();
    expect(
      document.body
        .querySelector('[data-test="notifications-copy"]')
        ?.textContent?.trim(),
    ).toBe("Copy failed");
    wrapper.unmount();
  });
});
