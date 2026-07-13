import { describe, expect, it } from "vitest";
import {
  NAV_ROUTES,
  allowsNativeSelectAll,
  isSelectAllChord,
  resolveShellShortcut,
} from "./shortcuts";

const key = (k: string, mods: Partial<Parameters<typeof resolveShellShortcut>[0]> = {}) => ({
  key: k,
  metaKey: true,
  ctrlKey: false,
  altKey: false,
  shiftKey: false,
  ...mods,
});

describe("resolveShellShortcut", () => {
  it("maps ⌘1–⌘6 and ⌘, to the six screens plus settings", () => {
    expect(resolveShellShortcut(key("1"))).toEqual({ kind: "navigate", route: "/generate" });
    expect(resolveShellShortcut(key("2"))).toEqual({ kind: "navigate", route: "/gallery" });
    expect(resolveShellShortcut(key("3"))).toEqual({ kind: "navigate", route: "/chains" });
    expect(resolveShellShortcut(key("4"))).toEqual({ kind: "navigate", route: "/models" });
    expect(resolveShellShortcut(key("5"))).toEqual({ kind: "navigate", route: "/history" });
    expect(resolveShellShortcut(key("6"))).toEqual({ kind: "navigate", route: "/jobs" });
    expect(resolveShellShortcut(key(","))).toEqual({ kind: "navigate", route: "/settings" });
  });

  it("maps ⌘\\ to sidebar toggle and ⌘K to the command palette", () => {
    expect(resolveShellShortcut(key("\\"))).toEqual({ kind: "toggle-sidebar" });
    expect(resolveShellShortcut(key("k"))).toEqual({ kind: "command-palette" });
  });

  it("maps ⌘. to cancel the focused job", () => {
    expect(resolveShellShortcut(key("."))).toEqual({ kind: "cancel-job" });
  });

  it("maps ⌘N and ⌘R to new generation and randomize seed", () => {
    expect(resolveShellShortcut(key("n"))).toEqual({ kind: "new-generation" });
    expect(resolveShellShortcut(key("r"))).toEqual({ kind: "randomize-seed" });
  });

  it("maps ⇧⌘C to copy seed", () => {
    expect(resolveShellShortcut(key("c", { shiftKey: true }))).toEqual({ kind: "copy-seed" });
    expect(resolveShellShortcut(key("C", { shiftKey: true }))).toEqual({ kind: "copy-seed" });
  });

  it("maps ⌘0 / ⌘+ / ⌘- to whole-app scaling", () => {
    expect(resolveShellShortcut(key("0"))).toEqual({ kind: "ui-scale", direction: "reset" });
    expect(resolveShellShortcut(key("="))).toEqual({ kind: "ui-scale", direction: "in" });
    expect(resolveShellShortcut(key("+"))).toEqual({ kind: "ui-scale", direction: "in" });
    expect(resolveShellShortcut(key("+", { shiftKey: true }))).toEqual({
      kind: "ui-scale",
      direction: "in",
    });
    expect(resolveShellShortcut(key("-"))).toEqual({ kind: "ui-scale", direction: "out" });
  });

  it("ignores keys without ⌘ or with disallowed modifiers", () => {
    expect(resolveShellShortcut(key("1", { metaKey: false }))).toBeNull();
    expect(resolveShellShortcut(key("1", { shiftKey: true }))).toBeNull();
    expect(resolveShellShortcut(key("1", { altKey: true }))).toBeNull();
    expect(resolveShellShortcut(key("1", { ctrlKey: true }))).toBeNull();
    expect(resolveShellShortcut(key("x"))).toBeNull();
  });

  it("covers every navigation route exactly once", () => {
    expect(new Set(Object.values(NAV_ROUTES)).size).toBe(7);
  });
});

describe("isSelectAllChord", () => {
  it("recognizes plain ⌘A in either case", () => {
    expect(isSelectAllChord(key("a"))).toBe(true);
    expect(isSelectAllChord(key("A"))).toBe(true);
  });

  it("rejects other keys and extra modifiers", () => {
    expect(isSelectAllChord(key("a", { metaKey: false }))).toBe(false);
    expect(isSelectAllChord(key("a", { shiftKey: true }))).toBe(false);
    expect(isSelectAllChord(key("a", { altKey: true }))).toBe(false);
    expect(isSelectAllChord(key("a", { ctrlKey: true }))).toBe(false);
    expect(isSelectAllChord(key("b"))).toBe(false);
  });
});

describe("allowsNativeSelectAll", () => {
  it("denies when nothing has focus or focus sits on chrome", () => {
    expect(allowsNativeSelectAll(null)).toBe(false);
    expect(allowsNativeSelectAll(document.body)).toBe(false);
    const div = document.createElement("div");
    document.body.appendChild(div);
    expect(allowsNativeSelectAll(div)).toBe(false);
  });

  it("allows inputs and textareas", () => {
    expect(allowsNativeSelectAll(document.createElement("input"))).toBe(true);
    expect(allowsNativeSelectAll(document.createElement("textarea"))).toBe(true);
  });

  it("allows contenteditable elements", () => {
    const div = document.createElement("div");
    div.setAttribute("contenteditable", "true");
    document.body.appendChild(div);
    expect(allowsNativeSelectAll(div)).toBe(true);
  });

  it("allows anything inside an opted-in [data-selectable] region", () => {
    const region = document.createElement("section");
    region.setAttribute("data-selectable", "");
    const child = document.createElement("span");
    region.appendChild(child);
    document.body.appendChild(region);
    expect(allowsNativeSelectAll(child)).toBe(true);
  });
});
