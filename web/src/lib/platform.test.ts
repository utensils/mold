import { describe, expect, it } from "vitest";
import { detectWebPlatform } from "./platform";

describe("detectWebPlatform", () => {
  it("reads the Mac desktop platform", () => {
    expect(detectWebPlatform("MacIntel")).toBe("macos");
    expect(detectWebPlatform("macOS")).toBe("macos");
  });

  // An iPad or an iPhone with a hardware keyboard types ⌘, and neither
  // reports "mac" — Safari answers "iPhone"/"iPad" and the UA-Client-Hints
  // platform is "iOS".
  it("counts iPhone, iPad and iPod as the Apple platform", () => {
    expect(detectWebPlatform("iPad")).toBe("macos");
    expect(detectWebPlatform("iPhone")).toBe("macos");
    expect(detectWebPlatform("iPod touch")).toBe("macos");
    expect(detectWebPlatform("iOS")).toBe("macos");
  });

  it("reads Linux and Windows as themselves", () => {
    expect(detectWebPlatform("Linux x86_64")).toBe("linux");
    expect(detectWebPlatform("Win32")).toBe("windows");
    expect(detectWebPlatform("Windows")).toBe("windows");
  });

  it("answers unknown for an absent or unrecognized platform", () => {
    expect(detectWebPlatform(undefined)).toBe("unknown");
    expect(detectWebPlatform("")).toBe("unknown");
    expect(detectWebPlatform("Android")).toBe("unknown");
  });
});

describe("shortcutLabel and primaryModifierPressed", () => {
  it("spells and gates a chord with the detected platform's modifier", async () => {
    // The browser in this suite is not a Mac, so the neutral Ctrl chord is
    // what the module binds; the platform-specific spellings are pinned in
    // `@studio/lib/platform`.
    const { PLATFORM_UI, primaryModifierPressed, shortcutLabel } =
      await import("./platform");
    expect(shortcutLabel("E")).toBe(`${PLATFORM_UI.modifierLabel}E`);
    const primary = PLATFORM_UI.isMacOS
      ? { metaKey: true, ctrlKey: false, altKey: false }
      : { metaKey: false, ctrlKey: true, altKey: false };
    expect(primaryModifierPressed(primary)).toBe(true);
    expect(primaryModifierPressed({ ...primary, altKey: true })).toBe(false);
  });
});
