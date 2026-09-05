import { describe, expect, it } from "vitest";
import {
  NAV_ROUTES,
  allowsNativeContextMenu,
  allowsNativeSelectAll,
  isSelectAllChord,
  overlayOwnsKeyboard,
  ownsBareBackspace,
  resolveFocusSensitiveShortcut,
  resolveShellShortcut,
  type ShellKeyContext,
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
  it("maps ⌘1–⌘5 and ⌘, to the sidebar's destinations in order", () => {
    expect(resolveMacShortcut(key("1"))).toEqual({ kind: "navigate", route: "/create" });
    expect(resolveMacShortcut(key("2"))).toEqual({ kind: "navigate", route: "/queue" });
    expect(resolveMacShortcut(key("3"))).toEqual({ kind: "navigate", route: "/library" });
    expect(resolveMacShortcut(key("4"))).toEqual({ kind: "navigate", route: "/models" });
    expect(resolveMacShortcut(key("5"))).toEqual({ kind: "navigate", route: "/machines" });
    expect(resolveMacShortcut(key(","))).toEqual({ kind: "navigate", route: "/settings" });
  });

  it("does not retain the retired ⌘6 alias", () => {
    expect(resolveMacShortcut(key("6"))).toBeNull();
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

  it("leaves ⌥↩ to the focus-sensitive map, so a field can claim it", () => {
    const chord = key("Enter", { metaKey: false, altKey: true });
    expect(resolveMacShortcut(chord)).toBeNull();
    expect(resolveShellShortcut(chord, "windows")).toBeNull();
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

  it("resolves to exactly the six destinations", () => {
    expect(new Set(Object.values(NAV_ROUTES))).toEqual(
      new Set(["/create", "/queue", "/library", "/models", "/machines", "/settings"]),
    );
  });
});

describe("resolveFocusSensitiveShortcut", () => {
  const space = (over: Partial<Parameters<typeof resolveFocusSensitiveShortcut>[0]> = {}) => ({
    key: " ",
    metaKey: false,
    ctrlKey: false,
    altKey: false,
    shiftKey: false,
    ...over,
  });
  const context = (over: Partial<ShellKeyContext> = {}): ShellKeyContext => ({
    target: null,
    overlayOpen: false,
    route: "/create",
    canPauseQueue: true,
    ...over,
  });

  it("pauses the queue on a bare Space outside a field", () => {
    expect(resolveFocusSensitiveShortcut(space(), context())).toEqual({
      kind: "toggle-queue-pause",
    });
  });

  it("leaves Space to the field being typed in", () => {
    for (const tag of ["input", "textarea", "select"]) {
      const el = document.createElement(tag);
      expect(resolveFocusSensitiveShortcut(space(), context({ target: el })), tag).toBeNull();
    }
    const editable = document.createElement("div");
    editable.setAttribute("contenteditable", "true");
    document.body.appendChild(editable);
    expect(resolveFocusSensitiveShortcut(space(), context({ target: editable }))).toBeNull();
  });

  it("leaves Space to a focused button, which owns it as its own activation", () => {
    expect(
      resolveFocusSensitiveShortcut(space(), context({ target: document.createElement("button") })),
    ).toBeNull();
    const row = document.createElement("div");
    row.setAttribute("role", "button");
    expect(resolveFocusSensitiveShortcut(space(), context({ target: row }))).toBeNull();
  });

  it("stands down under an overlay, and in My images where Space is Quick Look", () => {
    expect(resolveFocusSensitiveShortcut(space(), context({ overlayOpen: true }))).toBeNull();
    expect(resolveFocusSensitiveShortcut(space(), context({ route: "/library" }))).toBeNull();
  });

  it("leaves Space alone on a machine whose queue cannot be paused", () => {
    // The shell claims a bare key only where it can act: a host that does not
    // advertise pause used to swallow Space, fire a queue read, and do
    // nothing. The status bar already hides its Space hint on such a host.
    expect(resolveFocusSensitiveShortcut(space(), context({ canPauseQueue: false }))).toBeNull();
    // ⌥↩ is unrelated to the queue and keeps working there.
    expect(
      resolveFocusSensitiveShortcut(
        space({ key: "Enter", altKey: true }),
        context({ canPauseQueue: false }),
      ),
    ).toEqual({ kind: "make-variations" });
  });

  it("ignores a modified or repeating Space, and every other bare key", () => {
    expect(resolveFocusSensitiveShortcut(space({ metaKey: true }), context())).toBeNull();
    expect(resolveFocusSensitiveShortcut(space({ shiftKey: true }), context())).toBeNull();
    expect(resolveFocusSensitiveShortcut(space({ repeat: true }), context())).toBeNull();
    expect(resolveFocusSensitiveShortcut(space({ key: "p" }), context())).toBeNull();
  });

  const optionReturn = () => space({ key: "Enter", altKey: true });

  it("makes four variations on ⌥↩ outside a field, on every route", () => {
    expect(resolveFocusSensitiveShortcut(optionReturn(), context())).toEqual({
      kind: "make-variations",
    });
    expect(resolveFocusSensitiveShortcut(optionReturn(), context({ route: "/library" }))).toEqual({
      kind: "make-variations",
    });
  });

  it("leaves Option+Return to the prompt, where it inserts a newline", () => {
    const textarea = document.createElement("textarea");
    expect(resolveFocusSensitiveShortcut(optionReturn(), context({ target: textarea }))).toBeNull();
  });

  it("stands down on ⌥↩ under an open dialog", () => {
    expect(
      resolveFocusSensitiveShortcut(optionReturn(), context({ overlayOpen: true })),
    ).toBeNull();
  });

  it("claims ⌥↩ only without the primary modifier or Shift", () => {
    expect(
      resolveFocusSensitiveShortcut(
        space({ key: "Enter", altKey: true, metaKey: true }),
        context(),
      ),
    ).toBeNull();
    expect(
      resolveFocusSensitiveShortcut(
        space({ key: "Enter", altKey: true, shiftKey: true }),
        context(),
      ),
    ).toBeNull();
    expect(resolveFocusSensitiveShortcut(space({ key: "Enter" }), context())).toBeNull();
  });
});

describe("ownsBareBackspace", () => {
  it("keeps Backspace only inside a text-editing surface", () => {
    // Backspace outside a field is the webview's history Back. Nothing in the
    // shell binds it, and a Back inside a single-page app unmounts the whole
    // window — so the shell swallows it everywhere the caret is not.
    expect(ownsBareBackspace(document.createElement("input"))).toBe(true);
    expect(ownsBareBackspace(document.createElement("textarea"))).toBe(true);
    const editable = document.createElement("div");
    editable.setAttribute("contenteditable", "true");
    document.body.appendChild(editable);
    expect(ownsBareBackspace(editable)).toBe(true);
  });

  it("claims Backspace on chrome, including a focused control or nothing at all", () => {
    expect(ownsBareBackspace(null)).toBe(false);
    expect(ownsBareBackspace(document.body)).toBe(false);
    expect(ownsBareBackspace(document.createElement("button"))).toBe(false);
    // A range slider is an input, but it is chrome: Backspace does nothing in
    // it and everything to the window.
    const range = document.createElement("input");
    range.type = "range";
    expect(ownsBareBackspace(range)).toBe(false);
    expect(ownsBareBackspace(document.createElement("select"))).toBe(false);
  });
});

describe("overlayOwnsKeyboard", () => {
  it("reads the aria-modal marker every kit panel sets", () => {
    const root = document.createElement("div");
    expect(overlayOwnsKeyboard(root)).toBe(false);
    const panel = document.createElement("div");
    panel.setAttribute("aria-modal", "true");
    root.appendChild(panel);
    expect(overlayOwnsKeyboard(root)).toBe(true);
  });
});

describe("isSelectAllChord", () => {
  it("recognizes plain ⌘A in either case", () => {
    expect(isSelectAllChord(key("a"), "macos")).toBe(true);
    expect(isSelectAllChord(key("A"), "macos")).toBe(true);
  });

  it("recognizes plain Ctrl+A on Windows and Linux", () => {
    const ctrlA = key("a", { metaKey: false, ctrlKey: true });
    expect(isSelectAllChord(ctrlA, "windows")).toBe(true);
    expect(isSelectAllChord(ctrlA, "linux")).toBe(true);
  });

  it("does not cross the platform modifier", () => {
    expect(isSelectAllChord(key("a"), "windows")).toBe(false);
    expect(isSelectAllChord(key("a", { metaKey: false, ctrlKey: true }), "macos")).toBe(false);
  });

  it("rejects other keys and extra modifiers", () => {
    expect(isSelectAllChord(key("a", { metaKey: false }), "macos")).toBe(false);
    expect(isSelectAllChord(key("a", { shiftKey: true }), "macos")).toBe(false);
    expect(isSelectAllChord(key("a", { altKey: true }), "macos")).toBe(false);
    expect(isSelectAllChord(key("a", { ctrlKey: true }), "macos")).toBe(false);
    expect(isSelectAllChord(key("b"), "macos")).toBe(false);
    expect(isSelectAllChord(key("a", { metaKey: true, ctrlKey: true }), "windows")).toBe(false);
    expect(
      isSelectAllChord(key("a", { metaKey: false, ctrlKey: true, altKey: true }), "windows"),
    ).toBe(false);
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

describe("allowsNativeContextMenu", () => {
  it("allows only text-editing surfaces (spellcheck / paste menus)", () => {
    expect(allowsNativeContextMenu(document.createElement("textarea"))).toBe(true);
    for (const type of ["text", "search", "url", "tel", "email", "password", "number"]) {
      const input = document.createElement("input");
      input.type = type;
      expect(allowsNativeContextMenu(input), type).toBe(true);
    }
  });

  it("suppresses everywhere else — a range slider is chrome, not text", () => {
    expect(allowsNativeContextMenu(null)).toBe(false);
    expect(allowsNativeContextMenu(document.body)).toBe(false);
    expect(allowsNativeContextMenu(document.createElement("div"))).toBe(false);
    for (const type of ["checkbox", "radio", "range", "file", "button", "submit", "color"]) {
      const input = document.createElement("input");
      input.type = type;
      expect(allowsNativeContextMenu(input), type).toBe(false);
    }
  });
});
