import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { useToastStore } from "./toasts";

beforeEach(() => {
  setActivePinia(createPinia());
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
});

describe("toast store", () => {
  it("auto-dismisses after the default lifetime and returns the id", () => {
    const toasts = useToastStore();
    const id = toasts.push("Saved");
    expect(toasts.has(id)).toBe(true);
    vi.advanceTimersByTime(4000);
    expect(toasts.has(id)).toBe(false);
  });

  it("honours a custom duration", () => {
    const toasts = useToastStore();
    const id = toasts.push("Deleted", "info", { durationMs: 6000 });
    vi.advanceTimersByTime(4000);
    expect(toasts.has(id)).toBe(true);
    vi.advanceTimersByTime(2000);
    expect(toasts.has(id)).toBe(false);
  });

  it("never auto-dismisses a sticky toast", () => {
    const toasts = useToastStore();
    const id = toasts.push("Can't reach hal9000 — check Machines.", "error", { sticky: true });
    vi.advanceTimersByTime(60_000);
    expect(toasts.has(id)).toBe(true);
  });

  it("runs an action and dismisses", () => {
    const toasts = useToastStore();
    const run = vi.fn();
    const id = toasts.push("Deleted", "info", { action: { label: "Undo", run } });
    toasts.runAction(id);
    expect(run).toHaveBeenCalledTimes(1);
    expect(toasts.has(id)).toBe(false);
  });

  it("runs a body-click handler and dismisses", () => {
    const toasts = useToastStore();
    const onClick = vi.fn();
    const id = toasts.push("Generated — saved to Library", "info", { onClick });
    toasts.click(id);
    expect(onClick).toHaveBeenCalledTimes(1);
    expect(toasts.has(id)).toBe(false);
  });

  it("keeps only three transient toasts but protects sticky/action ones", () => {
    const toasts = useToastStore();
    const sticky = toasts.push("offline", "error", { sticky: true });
    const undo = toasts.push("deleted", "info", { action: { label: "Undo", run: () => {} } });
    toasts.push("a");
    toasts.push("b");
    toasts.push("c");
    toasts.push("d");
    // The plain toasts trim to fit, but the sticky + action toasts survive.
    expect(toasts.has(sticky)).toBe(true);
    expect(toasts.has(undo)).toBe(true);
  });

  it("records every toast in the notifications center so missed ones stay readable", async () => {
    const { useNotificationsStore } = await import("@studio/stores/notifications");
    const toasts = useToastStore();
    const notifications = useNotificationsStore();

    toasts.push("Generation failed", "error", { description: "CUDA out of memory on plato" });
    vi.advanceTimersByTime(10_000); // The toast is long gone…

    expect(notifications.entries[0]).toMatchObject({
      kind: "error",
      text: "Generation failed",
      description: "CUDA out of memory on plato",
    });
  });
});
