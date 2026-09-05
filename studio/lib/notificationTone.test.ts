import { describe, expect, it } from "vitest";
import {
  NOTIFICATION_BADGE_INK,
  NOTIFICATION_TONES,
  mostSevereKind,
  notificationTone,
} from "./notificationTone";

describe("notificationTone", () => {
  it("maps severity to the green / yellow / red token trio", () => {
    expect(notificationTone("success").color).toBe("var(--mold-success)");
    expect(notificationTone("warning").color).toBe("var(--mold-warning)");
    expect(notificationTone("error").color).toBe("var(--mold-error)");
  });

  it("reads an ordinary notice as green — only warnings and errors stand out", () => {
    expect(notificationTone("info").color).toBe("var(--mold-success)");
    expect(notificationTone("info").badge).toBe("var(--mold-success)");
    // The glyph, not the hue, separates a notice from a success.
    expect(notificationTone("info").glyph).not.toBe(
      notificationTone("success").glyph,
    );
  });

  it("carries a text label so severity is never color alone", () => {
    expect(Object.keys(NOTIFICATION_TONES).sort()).toEqual([
      "error",
      "info",
      "success",
      "warning",
    ]);
    for (const kind of Object.keys(NOTIFICATION_TONES)) {
      expect(
        notificationTone(kind as keyof typeof NOTIFICATION_TONES).label.length,
      ).toBeGreaterThan(0);
    }
    expect(notificationTone("warning").label).toBe("Warning");
  });

  it("gives every severity a distinct visible glyph, not just a hue", () => {
    const glyphs = Object.values(NOTIFICATION_TONES).map((tone) => tone.glyph);
    expect(new Set(glyphs).size).toBe(glyphs.length);
    expect(notificationTone("warning").glyph).not.toBe(
      notificationTone("error").glyph,
    );
  });

  it("fills a counted badge with an opaque token, never translucent hint ink", () => {
    // A translucent ink such as --mold-text-dim is a color-mix: a count printed on it
    // has no predictable contrast against whatever sits behind the badge.
    for (const tone of Object.values(NOTIFICATION_TONES)) {
      expect(tone.badge).not.toContain("ink-3");
    }
    // One per-theme ink is legible on every badge fill (guarded for contrast
    // in desktop/src/styles/tokens.contrast.test.ts).
    expect(NOTIFICATION_BADGE_INK).toBe("var(--mold-on-accent)");
  });
});

describe("mostSevereKind", () => {
  it("lets the worst unread entry color the badge", () => {
    expect(mostSevereKind(["info", "success", "warning"])).toBe("warning");
    expect(mostSevereKind(["warning", "error", "success"])).toBe("error");
    expect(mostSevereKind(["success"])).toBe("success");
  });

  it("falls back to info for an empty set", () => {
    expect(mostSevereKind([])).toBe("info");
  });
});
