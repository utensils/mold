import { describe, expect, it } from "vitest";
import { NAV_ROUTES, resolveShellShortcut } from "./shortcuts";

const key = (k: string, mods: Partial<Parameters<typeof resolveShellShortcut>[0]> = {}) => ({
  key: k,
  metaKey: true,
  ctrlKey: false,
  altKey: false,
  shiftKey: false,
  ...mods,
});

describe("resolveShellShortcut", () => {
  it("maps ⌘1–⌘5 and ⌘, to the five screens plus settings", () => {
    expect(resolveShellShortcut(key("1"))).toEqual({ kind: "navigate", route: "/generate" });
    expect(resolveShellShortcut(key("2"))).toEqual({ kind: "navigate", route: "/gallery" });
    expect(resolveShellShortcut(key("3"))).toEqual({ kind: "navigate", route: "/chains" });
    expect(resolveShellShortcut(key("4"))).toEqual({ kind: "navigate", route: "/models" });
    expect(resolveShellShortcut(key("5"))).toEqual({ kind: "navigate", route: "/history" });
    expect(resolveShellShortcut(key(","))).toEqual({ kind: "navigate", route: "/settings" });
  });

  it("maps ⌘\\ to sidebar toggle and ⌘K to the command palette", () => {
    expect(resolveShellShortcut(key("\\"))).toEqual({ kind: "toggle-sidebar" });
    expect(resolveShellShortcut(key("k"))).toEqual({ kind: "command-palette" });
  });

  it("ignores keys without ⌘ or with extra modifiers", () => {
    expect(resolveShellShortcut(key("1", { metaKey: false }))).toBeNull();
    expect(resolveShellShortcut(key("1", { shiftKey: true }))).toBeNull();
    expect(resolveShellShortcut(key("1", { altKey: true }))).toBeNull();
    expect(resolveShellShortcut(key("1", { ctrlKey: true }))).toBeNull();
    expect(resolveShellShortcut(key("x"))).toBeNull();
  });

  it("covers every navigation route exactly once", () => {
    expect(new Set(Object.values(NAV_ROUTES)).size).toBe(6);
  });
});
