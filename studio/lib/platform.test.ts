import { describe, expect, it } from "vitest";
import {
  normalizePlatform,
  platformUi,
  primaryModifierPressed,
} from "./platform";

describe("normalizePlatform", () => {
  it("reads every spelling of the Apple platform", () => {
    expect(normalizePlatform("MacIntel")).toBe("macos");
    expect(normalizePlatform("darwin")).toBe("macos");
  });

  it("reads Linux and Windows", () => {
    expect(normalizePlatform("Linux x86_64")).toBe("linux");
    expect(normalizePlatform("Win32")).toBe("windows");
  });

  it("answers unknown for absent or unrecognized platforms", () => {
    expect(normalizePlatform(undefined)).toBe("unknown");
    expect(normalizePlatform("   ")).toBe("unknown");
    expect(normalizePlatform("PlayStation")).toBe("unknown");
  });
});

describe("platformUi", () => {
  it("uses Command conventions on macOS", () => {
    expect(platformUi("darwin")).toEqual({
      isMacOS: true,
      modifier: "Meta",
      modifierLabel: "⌘",
      shiftLabel: "⇧",
      altLabel: "⌥",
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
      altLabel: "Alt+",
      deviceLabel: "This device",
      fileManagerLabel: "file manager",
    });
  });

  it("names Windows conventions, including File Explorer", () => {
    expect(platformUi("win32").fileManagerLabel).toBe("File Explorer");
  });

  it("falls back to the neutral Control conventions when nothing is known", () => {
    expect(platformUi().isMacOS).toBe(false);
    expect(platformUi("unknown").modifierLabel).toBe("Ctrl+");
  });
});

describe("primaryModifierPressed", () => {
  it("is Command on macOS and Control everywhere else", () => {
    const meta = { metaKey: true, ctrlKey: false, altKey: false };
    const ctrl = { metaKey: false, ctrlKey: true, altKey: false };
    expect(primaryModifierPressed(meta, "macos")).toBe(true);
    expect(primaryModifierPressed(ctrl, "macos")).toBe(false);
    expect(primaryModifierPressed(ctrl, "linux")).toBe(true);
    expect(primaryModifierPressed(meta, "linux")).toBe(false);
  });

  it("rejects Alt-modified primary chords on either platform", () => {
    expect(
      primaryModifierPressed(
        { metaKey: true, ctrlKey: false, altKey: true },
        "macos",
      ),
    ).toBe(false);
    expect(
      primaryModifierPressed(
        { metaKey: false, ctrlKey: true, altKey: true },
        "linux",
      ),
    ).toBe(false);
  });

  it("gives an unknown platform the Control chord", () => {
    expect(
      primaryModifierPressed(
        { metaKey: false, ctrlKey: true, altKey: false },
        "unknown",
      ),
    ).toBe(true);
  });
});
