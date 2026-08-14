import { beforeEach, describe, expect, it } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { NOTIFICATION_CAP, useNotificationsStore } from "./notifications";

beforeEach(() => {
  setActivePinia(createPinia());
});

describe("notifications store", () => {
  it("records entries newest-first with unread tracking", () => {
    const store = useNotificationsStore();
    store.record({ kind: "error", text: "Generation failed", atMs: 1_000 });
    store.record({
      kind: "info",
      text: "Download queued",
      description: "flux-dev:q8 on plato",
      hostLabel: "plato",
      atMs: 2_000,
    });

    expect(store.entries.map((e) => e.text)).toEqual([
      "Download queued",
      "Generation failed",
    ]);
    expect(store.entries[0]).toMatchObject({
      kind: "info",
      description: "flux-dev:q8 on plato",
      hostLabel: "plato",
      atMs: 2_000,
      read: false,
    });
    expect(store.unreadCount).toBe(2);
  });

  it("marks everything read without dropping history, and clears on demand", () => {
    const store = useNotificationsStore();
    store.record({ kind: "error", text: "boom", atMs: 1 });
    store.markAllRead();
    expect(store.unreadCount).toBe(0);
    expect(store.entries).toHaveLength(1);

    store.record({ kind: "info", text: "later", atMs: 2 });
    expect(store.unreadCount).toBe(1);

    store.clear();
    expect(store.entries).toHaveLength(0);
    expect(store.unreadCount).toBe(0);
  });

  it("caps retained history at the newest NOTIFICATION_CAP entries", () => {
    const store = useNotificationsStore();
    for (let i = 0; i < NOTIFICATION_CAP + 5; i++) {
      store.record({ kind: "info", text: `n${i}`, atMs: i });
    }
    expect(store.entries).toHaveLength(NOTIFICATION_CAP);
    expect(store.entries[0]!.text).toBe(`n${NOTIFICATION_CAP + 4}`);
    expect(store.entries.at(-1)!.text).toBe("n5");
  });

  it("collapses an immediate duplicate into a repeat count instead of a new row", () => {
    const store = useNotificationsStore();
    store.record({ kind: "error", text: "Connection lost", atMs: 1_000 });
    store.record({ kind: "error", text: "Connection lost", atMs: 2_000 });

    expect(store.entries).toHaveLength(1);
    expect(store.entries[0]).toMatchObject({ repeat: 2, atMs: 2_000 });
    expect(store.unreadCount).toBe(1);
  });
});
