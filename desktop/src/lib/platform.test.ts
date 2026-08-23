import { describe, expect, it } from "vitest";
import {
  applyPlatformAttribute,
  detectPlatform,
  platformUi,
  primaryModifierPressed,
} from "./platform";

describe("detectPlatform", () => {
  it("falls back to the browser platform when the Tauri build variable is unavailable", () => {
    expect(detectPlatform(undefined, "MacIntel", true)).toBe("macos");
  });

  it("prefers the Tauri build platform over the browser fallback", () => {
    expect(detectPlatform("linux", "MacIntel", true)).toBe("linux");
  });

  it("keeps ordinary browser previews platform-neutral", () => {
    expect(detectPlatform(undefined, "MacIntel", false)).toBe("unknown");
  });
});

describe("applyPlatformAttribute", () => {
  it("exposes Linux to CSS before the desktop shell mounts", () => {
    const root = document.createElement("html");
    applyPlatformAttribute(root, "linux");
    expect(root.dataset.platform).toBe("linux");
  });
});

describe("platformUi", () => {
  it("uses Command conventions on macOS", () => {
    expect(platformUi("darwin")).toEqual({
      isMacOS: true,
      modifier: "Meta",
      modifierLabel: "⌘",
      shiftLabel: "⇧",
      deviceLabel: "This Mac",
      fileManagerLabel: "Finder",
    });
  });

  it("uses Control and neutral device conventions on Linux", () => {
    expect(platformUi("linux")).toEqual({
      isMacOS: false,
      modifier: "Control",
      modifierLabel: "Ctrl+",
      shiftLabel: "Shift+",
      deviceLabel: "This device",
      // Linux has no single file manager to name, so the generic stands.
      fileManagerLabel: "file manager",
    });
  });

  it("names Windows conventions, including File Explorer", () => {
    expect(platformUi("win32")).toEqual({
      isMacOS: false,
      modifier: "Control",
      modifierLabel: "Ctrl+",
      shiftLabel: "Shift+",
      deviceLabel: "This device",
      fileManagerLabel: "File Explorer",
    });
  });

  it("uses the neutral Control conventions when the platform is unknown", () => {
    expect(platformUi("unknown")).toMatchObject({
      isMacOS: false,
      modifier: "Control",
      modifierLabel: "Ctrl+",
      deviceLabel: "This device",
    });
  });
});

describe("primaryModifierPressed", () => {
  it("rejects Alt-modified primary chords", () => {
    expect(primaryModifierPressed({ metaKey: false, ctrlKey: true, altKey: true })).toBe(false);
  });
});
