import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { THEMES, THEME_TONE, type ThemeId } from "@ui/theme";

// The palette is owned by the shared design system (ui/tokens.css); the
// desktop app only maps it into Tailwind. Guard every theme at the source so a
// map edit can never ship an unreadable rank. The rule (style guide §08):
// contrast is measured against the LOWEST-contrast surface a rank appears on,
// never the base background alone. Path is relative to the Vitest cwd (the
// desktop package root).
const css = readFileSync("../ui/tokens.css", "utf8");

type ThemeMap = Record<string, string>;

/** Every key a complete theme map declares, in style-guide §03 order. */
const THEME_KEYS = [
  "bg",
  "bg-deep",
  "bg-crust",
  "surface",
  "surface-2",
  "surface-3",
  "text",
  "text-2",
  "text-dim",
  "text-faint",
  "border",
  "border-focus",
  "border-control",
  "blue",
  "on-accent",
  "success",
  "warning",
  "error",
  "star",
  "sapphire",
  "mauve",
  "teal",
  "lavender",
  "font-sans",
  "font-mono",
  "fs-micro",
  "fs-xs",
  "fs-sm",
  "fs-base",
  "fs-md",
  "fs-lg",
  "fs-xl",
  "lh-snug",
  "lh-body",
  "radius-1",
  "radius-2",
  "radius-3",
] as const;

/** The declarations a selector opens. A theme map is also selectable on a
 *  nested element, so the selector may lead a list rather than stand alone. */
function block(selector: string): string {
  const start = css.indexOf(selector);
  if (start < 0) throw new Error(`Missing CSS block: ${selector}`);
  const bodyStart = css.indexOf("{", start) + 1;
  return css.slice(bodyStart, css.indexOf("}", bodyStart));
}

function themeMap(id: ThemeId): ThemeMap & { colorScheme: string } {
  const body = block(`:root[data-theme="${id}"]`);
  const map: ThemeMap = Object.fromEntries(
    [...body.matchAll(/--mold-([\w-]+):\s*([^;]+);/g)].map((m) => [
      m[1]!,
      m[2]!.replace(/\s+/g, " ").trim(),
    ]),
  );
  const scheme = body.match(/color-scheme:\s*(\w+);/)?.[1] ?? "";
  return { ...map, colorScheme: scheme };
}

function rgb(hex: string): [number, number, number] {
  if (!/^#[0-9a-f]{6}$/i.test(hex)) throw new Error(`Expected a 6-digit hex colour, got ${hex}`);
  return [1, 3, 5].map((offset) => parseInt(hex.slice(offset, offset + 2), 16)) as [
    number,
    number,
    number,
  ];
}

function luminance(color: string): number {
  const channels = rgb(color).map((channel) => {
    const value = channel / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * channels[0]! + 0.7152 * channels[1]! + 0.0722 * channels[2]!;
}

function contrast(foreground: string, background: string): number {
  const values = [luminance(foreground), luminance(background)].sort((a, b) => b - a);
  return (values[0]! + 0.05) / (values[1]! + 0.05);
}

/** Flatten `foreground` at `opacity` over `background`. */
function composite(foreground: string, background: string, opacity: number): string {
  const fg = rgb(foreground);
  const bg = rgb(background);
  return `#${fg
    .map((channel, index) =>
      Math.round(channel * opacity + bg[index]! * (1 - opacity))
        .toString(16)
        .padStart(2, "0"),
    )
    .join("")}`;
}

/** Resolve a map value to a hex: literal, `var(--mold-x)`, or a text mix. */
function colour(theme: ThemeMap, key: string, over?: string): string {
  const raw = theme[key];
  if (raw === undefined) throw new Error(`Missing theme token: ${key}`);
  const ref = raw.match(/^var\(--mold-([\w-]+)\)$/);
  if (ref) return colour(theme, ref[1]!, over);
  const mix = raw.match(/^color-mix\(in srgb, var\(--mold-([\w-]+)\) ([0-9.]+)%, transparent\)$/);
  if (mix) {
    if (!over) throw new Error(`${key} is a mix; a background is required`);
    return composite(colour(theme, mix[1]!), over, Number(mix[2]!) / 100);
  }
  return raw;
}

const px = (value: string) => Number(value.replace(/px$/, ""));

/** The planes a text rank sits on. surface-2 is a selected row, where the
 * rank is promoted to --mold-text; surface-3 is a divider, never a plane. */
const INK_PLANES = ["bg-crust", "bg-deep", "bg", "surface"] as const;
const CHROME_PLANES = ["bg-crust", "bg-deep", "bg"] as const;

describe("six-theme contrast (style guide §08)", () => {
  for (const id of THEMES) {
    const theme = themeMap(id);

    it(`${id} declares every key of the map`, () => {
      // A partial map would inherit stale values from the :root default.
      for (const key of THEME_KEYS) expect(theme[key], key).toBeDefined();
      expect(theme.colorScheme).toBe(THEME_TONE[id]);
    });

    it(`${id} keeps every readable text rank at WCAG AA on every plane it touches`, () => {
      for (const plane of INK_PLANES) {
        const bg = colour(theme, plane);
        for (const rank of ["text", "text-2", "text-dim"]) {
          expect(contrast(colour(theme, rank), bg), `${rank} on ${plane}`).toBeGreaterThanOrEqual(
            4.5,
          );
        }
      }
      // text-faint is decorative only (disabled glyphs, thumbnail mocks) and is
      // bounded on both sides: never readable enough to be mistaken for an
      // information rank, never so faint it disappears.
      const faint = contrast(colour(theme, "text-faint"), colour(theme, "bg"));
      expect(faint, "text-faint on bg").toBeGreaterThanOrEqual(1.8);
      expect(faint, "text-faint on bg").toBeLessThanOrEqual(3.6);
    });

    it(`${id} keeps the accent and status hues legible on the chrome planes`, () => {
      // Accent and error carry text (links, keycaps, failure lines) on the
      // chrome and content planes; the canvas bed only ever carries glyphs.
      for (const plane of ["bg-deep", "bg"] as const) {
        const bg = colour(theme, plane);
        for (const hue of ["blue", "error"]) {
          expect(contrast(colour(theme, hue), bg), `${hue} on ${plane}`).toBeGreaterThanOrEqual(
            4.5,
          );
        }
      }
      // Status and data-series hues fill dots, meters and glyphs: non-text 3:1.
      for (const plane of CHROME_PLANES) {
        const bg = colour(theme, plane);
        for (const hue of [
          "blue",
          "error",
          "success",
          "warning",
          "star",
          "sapphire",
          "mauve",
          "teal",
          "lavender",
        ]) {
          expect(contrast(colour(theme, hue), bg), `${hue} on ${plane}`).toBeGreaterThanOrEqual(3);
        }
      }
    });

    it(`${id} inks accent and status fills with one per-theme colour`, () => {
      // "Never white by reflex": the ink is bg-deep on dark themes and the
      // lightest surface on light ones, and it must clear AA on every fill a
      // label can sit on (primary action, badges, seam chips).
      const ink = colour(theme, "on-accent");
      expect(ink).toBe(colour(theme, THEME_TONE[id] === "dark" ? "bg-deep" : "surface"));
      for (const fill of ["blue", "success", "warning", "error", "star"]) {
        expect(contrast(ink, colour(theme, fill)), `ink on ${fill}`).toBeGreaterThanOrEqual(4.5);
      }
    });

    it(`${id} keeps borders visible and focus equal to the accent`, () => {
      expect(theme["border-focus"], "focus is the accent").toBe(theme.blue);
      for (const plane of ["bg", "surface"] as const) {
        const bg = colour(theme, plane);
        // Native control boundaries need 3:1 (WCAG 1.4.11); the resting
        // hairline only has to read as a separation.
        expect(
          contrast(colour(theme, "border-control", bg), bg),
          `control edge on ${plane}`,
        ).toBeGreaterThanOrEqual(3);
      }
      expect(
        contrast(colour(theme, "border"), colour(theme, "bg")),
        "hairline",
      ).toBeGreaterThanOrEqual(1.2);
    });

    it(`${id} keeps the type scale monotonic and the radius scale ordered`, () => {
      const scale = ["fs-micro", "fs-xs", "fs-sm", "fs-base", "fs-md", "fs-lg", "fs-xl"].map((k) =>
        px(theme[k]!),
      );
      expect(scale[0], "micro floor").toBeGreaterThanOrEqual(10);
      for (let i = 1; i < scale.length; i += 1) {
        expect(scale[i], `step ${i}`).toBeGreaterThan(scale[i - 1]!);
      }
      const radii = ["radius-1", "radius-2", "radius-3"].map((k) => px(theme[k]!));
      expect(radii[0]).toBeLessThanOrEqual(radii[1]!);
      expect(radii[1]).toBeLessThanOrEqual(radii[2]!);
      expect(radii[2], "never a pill").toBeLessThanOrEqual(16);
    });
  }

  it("inlines the default theme as the :root map, byte for byte", () => {
    // The :root block carries the default theme so a document with no
    // data-theme paints correctly; it must never drift from the named block.
    const root = Object.fromEntries(
      [...block(":root").matchAll(/--mold-([\w-]+):\s*([^;]+);/g)].map((m) => [
        m[1]!,
        m[2]!.replace(/\s+/g, " ").trim(),
      ]),
    );
    for (const [key, value] of Object.entries(themeMap("mocha"))) {
      if (key === "colorScheme") continue;
      expect(root[key], key).toBe(value);
    }
  });

  it("declares exactly the ThemeId set and nothing else", () => {
    const declared = [...css.matchAll(/:root\[data-theme="([\w-]+)"\]/g)].map((m) => m[1]!);
    expect([...declared].sort()).toEqual([...THEMES].sort());
  });
});
