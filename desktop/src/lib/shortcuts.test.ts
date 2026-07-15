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
const resolveMacShortcut = (event: ReturnType<typeof key>) => resolveShellShortcut(event, "macos");

describe("resolveShellShortcut", () => {
  it("maps ⌘1–⌘6 and ⌘, to the six screens plus settings", () => {
    expect(resolveMacShortcut(key("1"))).toEqual({ kind: "navigate", route: "/generate" });
    expect(resolveMacShortcut(key("2"))).toEqual({ kind: "navigate", route: "/gallery" });
    expect(resolveMacShortcut(key("3"))).toEqual({ kind: "navigate", route: "/chains" });
    expect(resolveMacShortcut(key("4"))).toEqual({ kind: "navigate", route: "/models" });
    expect(resolveMacShortcut(key("5"))).toEqual({ kind: "navigate", route: "/history" });
    expect(resolveMacShortcut(key("6"))).toEqual({ kind: "navigate", route: "/jobs" });
    expect(resolveMacShortcut(key(","))).toEqual({ kind: "navigate", route: "/settings" });
  });

  it("maps ⌘\\ to sidebar toggle and ⌘K to the command palette", () => {
    expect(resolveMacShortcut(key("\\"))).toEqual({ kind: "toggle-sidebar" });
    expect(resolveMacShortcut(key("k"))).toEqual({ kind: "command-palette" });
  });

  it("maps ⌘. to cancel the focused job", () => {
    expect(resolveMacShortcut(key("."))).toEqual({ kind: "cancel-job" });
  });

  it("maps ⌘N and ⌘R to new generation and randomize seed", () => {
    expect(resolveMacShortcut(key("n"))).toEqual({ kind: "new-generation" });
    expect(resolveMacShortcut(key("r"))).toEqual({ kind: "randomize-seed" });
  });

  it("maps ⇧⌘C to copy seed", () => {
    expect(resolveMacShortcut(key("c", { shiftKey: true }))).toEqual({ kind: "copy-seed" });
    expect(resolveMacShortcut(key("C", { shiftKey: true }))).toEqual({ kind: "copy-seed" });
  });

  it("maps ⌘0 / ⌘+ / ⌘- to whole-app scaling", () => {
    expect(resolveMacShortcut(key("0"))).toEqual({ kind: "ui-scale", direction: "reset" });
    expect(resolveMacShortcut(key("="))).toEqual({ kind: "ui-scale", direction: "in" });
    expect(resolveMacShortcut(key("+"))).toEqual({ kind: "ui-scale", direction: "in" });
    expect(resolveMacShortcut(key("+", { shiftKey: true }))).toEqual({
      kind: "ui-scale",
      direction: "in",
    });
    expect(resolveMacShortcut(key("-"))).toEqual({ kind: "ui-scale", direction: "out" });
  });

  it("ignores keys without ⌘ or with disallowed modifiers", () => {
    expect(resolveMacShortcut(key("1", { metaKey: false }))).toBeNull();
    expect(resolveMacShortcut(key("1", { shiftKey: true }))).toBeNull();
    expect(resolveMacShortcut(key("1", { altKey: true }))).toBeNull();
    expect(resolveMacShortcut(key("1", { ctrlKey: true }))).toBeNull();
    expect(resolveMacShortcut(key("x"))).toBeNull();
  });

  it("uses Ctrl on Linux and rejects Meta-only shortcuts", () => {
    const linuxKey = key("k", { metaKey: false, ctrlKey: true });
    expect(resolveShellShortcut(linuxKey, "linux")).toEqual({ kind: "command-palette" });
    expect(resolveShellShortcut(key("k"), "linux")).toBeNull();
  });

  it("uses Ctrl when the platform is unknown, matching browser-preview labels", () => {
    const ctrlKey = key("k", { metaKey: false, ctrlKey: true });
    expect(resolveShellShortcut(ctrlKey, "unknown")).toEqual({ kind: "command-palette" });
    expect(resolveShellShortcut(key("k"), "unknown")).toBeNull();
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

  it("allows text-entry inputs and textareas", () => {
    expect(allowsNativeSelectAll(document.createElement("input"))).toBe(true);
    expect(allowsNativeSelectAll(document.createElement("textarea"))).toBe(true);
    for (const type of ["text", "search", "url", "tel", "email", "password", "number"]) {
      const input = document.createElement("input");
      input.type = type;
      expect(allowsNativeSelectAll(input), type).toBe(true);
    }
  });

  it("denies non-text inputs — a focused checkbox must not re-enable chrome select-all", () => {
    for (const type of ["checkbox", "radio", "range", "file", "button", "submit", "color"]) {
      const input = document.createElement("input");
      input.type = type;
      document.body.appendChild(input);
      expect(allowsNativeSelectAll(input), type).toBe(false);
    }
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
