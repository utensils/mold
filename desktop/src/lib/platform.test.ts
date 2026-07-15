import { describe, expect, it } from "vitest";
import { platformUi, primaryModifierPressed } from "./platform";

describe("platformUi", () => {
  it("uses Command conventions on macOS", () => {
    expect(platformUi("darwin")).toEqual({
      isMacOS: true,
      modifier: "Meta",
      modifierLabel: "⌘",
      deviceLabel: "This Mac",
      fileManagerLabel: "Finder",
    });
  });

  it("uses Control and neutral device conventions on Linux", () => {
    expect(platformUi("linux")).toEqual({
      isMacOS: false,
      modifier: "Control",
      modifierLabel: "Ctrl+",
      deviceLabel: "This device",
      fileManagerLabel: "file manager",
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
