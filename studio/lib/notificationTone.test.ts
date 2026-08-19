import { describe, expect, it } from "vitest";
import {
  NOTIFICATION_TONES,
  mostSevereKind,
  notificationTone,
} from "./notificationTone";

describe("notificationTone", () => {
  it("maps severity to the green / yellow / red token trio", () => {
    expect(notificationTone("success").color).toBe("var(--success)");
    expect(notificationTone("warning").color).toBe("var(--warning)");
    expect(notificationTone("error").color).toBe("var(--stop)");
  });

  it("keeps plain info neutral so severity colors stay meaningful", () => {
    expect(notificationTone("info").color).toBe("var(--ink-3)");
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
