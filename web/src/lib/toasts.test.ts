import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  dismissToast,
  requestConfirm,
  resetNotifications,
  runToastAction,
  settleConfirm,
  toast,
  undoableAction,
  useNotifications,
} from "./toasts";

beforeEach(() => {
  vi.useFakeTimers();
  resetNotifications();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("toast store", () => {
  it("survives a toast raised before pinia is installed", () => {
    // No active pinia yet in this file — recording is best-effort.
    expect(() => toast("info", "early")).not.toThrow();
  });

  it("records every toast in the notifications center so missed ones stay readable", async () => {
    const { createPinia, setActivePinia } = await import("pinia");
    setActivePinia(createPinia());
    const { useNotificationsStore } =
      await import("@studio/stores/notifications");

    toast("error", "Generation failed: CUDA out of memory");
    expect(useNotificationsStore().entries[0]).toMatchObject({
      kind: "error",
      text: "Generation failed: CUDA out of memory",
    });
  });

  it("adds and auto-dismisses non-error toasts", () => {
    const state = useNotifications();
    toast("success", "Generated — saved to Gallery");
    expect(state.toasts).toHaveLength(1);
    vi.advanceTimersByTime(5001);
    expect(state.toasts).toHaveLength(0);
  });

  it("keeps error toasts sticky until dismissed", () => {
    const state = useNotifications();
    const id = toast("error", "Can't reach studio.local:7680.");
    vi.advanceTimersByTime(60_000);
    expect(state.toasts).toHaveLength(1);
    dismissToast(id);
    expect(state.toasts).toHaveLength(0);
  });

  it("runs the action and dismisses", () => {
    const state = useNotifications();
    const onAction = vi.fn();
    const id = toast("info", "Print deleted", {
      actionLabel: "Undo",
      onAction,
    });
    runToastAction(id);
    expect(onAction).toHaveBeenCalledOnce();
    expect(state.toasts).toHaveLength(0);
  });
});

describe("undoableAction", () => {
  it("commits after the window when not undone", () => {
    const commit = vi.fn();
    const undo = vi.fn();
    undoableAction({ text: "Print deleted", undo, commit });
    vi.advanceTimersByTime(6001);
    expect(commit).toHaveBeenCalledOnce();
    expect(undo).not.toHaveBeenCalled();
  });

  it("never commits when undone inside the window", () => {
    const state = useNotifications();
    const commit = vi.fn();
    const undo = vi.fn();
    undoableAction({ text: "Print deleted", undo, commit });
    runToastAction(state.toasts[0]!.id);
    vi.advanceTimersByTime(10_000);
    expect(undo).toHaveBeenCalledOnce();
    expect(commit).not.toHaveBeenCalled();
  });

  it("commits exactly once when the toast is dismissed manually mid-window", () => {
    const state = useNotifications();
    const commit = vi.fn();
    const undo = vi.fn();
    undoableAction({ text: "Print deleted", undo, commit });
    // Closing the toast early via ✕ means "done with the undo offer" → commit.
    dismissToast(state.toasts[0]!.id);
    expect(commit).toHaveBeenCalledOnce();
    expect(undo).not.toHaveBeenCalled();
    // The window elapsing afterward must not commit a second time.
    vi.advanceTimersByTime(10_000);
    expect(commit).toHaveBeenCalledOnce();
  });

  it("does not commit after an undo, even if the toast is later dismissed", () => {
    const state = useNotifications();
    const commit = vi.fn();
    const undo = vi.fn();
    undoableAction({ text: "Print deleted", undo, commit });
    const id = state.toasts[0]!.id;
    runToastAction(id); // Undo
    dismissToast(id); // idempotent no-op — must not resurrect the commit
    vi.advanceTimersByTime(10_000);
    expect(undo).toHaveBeenCalledOnce();
    expect(commit).not.toHaveBeenCalled();
  });
});

describe("requestConfirm", () => {
  it("resolves true when settled affirmatively", async () => {
    const state = useNotifications();
    const pending = requestConfirm({
      title: "Delete print?",
      body: "This can't be undone.",
      danger: true,
    });
    expect(state.confirm?.title).toBe("Delete print?");
    settleConfirm(true);
    await expect(pending).resolves.toBe(true);
    expect(state.confirm).toBeNull();
  });

  it("cancels a pending confirm when a second one arrives", async () => {
    const first = requestConfirm({ title: "A", body: "a" });
    const second = requestConfirm({ title: "B", body: "b" });
    settleConfirm(true);
    await expect(first).resolves.toBe(false);
    await expect(second).resolves.toBe(true);
  });
});
