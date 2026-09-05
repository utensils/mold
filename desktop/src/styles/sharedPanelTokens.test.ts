import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/*
 * `studio/` panels that the desktop Settings and Machines pages embed. They
 * are shared with web and the phone, so they carry no surface's colours — but
 * they must not carry a made-up variable either: an undefined custom property
 * silently falls through to its literal fallback, which is how DevicePanel
 * painted a light-grey hairline on every dark theme.
 *
 * Radius and font-size literals are already ratcheted by tokens.legacy.test.ts;
 * this pins the two things that guard cannot see.
 */

const PANELS = [
  "../../../studio/components/DevicePanel.vue",
  "../../../studio/components/MobilePairingCard.vue",
  "../../../studio/components/PairingAccessPanel.vue",
];

function source(name: string): string {
  return readFileSync(fileURLToPath(new URL(name, import.meta.url)), "utf8");
}

describe("shared studio panels", () => {
  it("name only variables the token sheet actually defines", () => {
    const tokens = source("../../../ui/tokens.css");
    for (const panel of PANELS) {
      const used = [...source(panel).matchAll(/var\(\s*(--[a-z0-9-]+)/g)].map((m) => m[1]!);
      const undefined_ = [...new Set(used)].filter(
        (name) => !name.startsWith("--mold-") || !tokens.includes(`${name}:`),
      );
      expect(`${panel}: ${undefined_.join(", ")}`).toBe(`${panel}: `);
    }
  });

  it("carry no drop shadow inside the app window", () => {
    // Elevation is --mold-shadow-md on the window and dialogs only; inside,
    // depth is surface value alone (ui/mold-desktop.css).
    for (const panel of PANELS) {
      expect(source(panel)).not.toMatch(/box-shadow:\s*[^;]*\d+px/);
    }
  });
});
