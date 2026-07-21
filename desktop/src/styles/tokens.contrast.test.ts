import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

// The palette is owned by the shared design system now; the desktop app only
// maps it into Tailwind (see tokens.css). Guard contrast at the source so a
// change to ui/tokens.css can't silently ship an unreadable theme to desktop.
// Path is relative to the Vitest cwd (the desktop package root).
const css = readFileSync("../ui/tokens.css", "utf8");

type Palette = Record<string, string>;

function block(selector: string): string {
  const marker = `${selector} {`;
  const start = css.indexOf(marker);
  if (start < 0) throw new Error(`Missing CSS block: ${selector}`);
  const bodyStart = start + marker.length;
  return css.slice(bodyStart, css.indexOf("}", bodyStart));
}

function declarations(selector: string): Palette {
  return Object.fromEntries(
    [...block(selector).matchAll(/--([\w-]+):\s*([^;]+);/g)].map((match) => [
      match[1]!,
      match[2]!.trim(),
    ]),
  );
}

function rgb(hex: string): [number, number, number] {
  const value = hex.trim();
  const expanded =
    value.length === 4
      ? `#${value[1]}${value[1]}${value[2]}${value[2]}${value[3]}${value[3]}`
      : value;
  if (!/^#[0-9a-f]{6}$/i.test(expanded)) throw new Error(`Expected hex color, got ${hex}`);
  return [1, 3, 5].map((offset) => parseInt(expanded.slice(offset, offset + 2), 16)) as [
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
  const [r, g, b] = [channels[0]!, channels[1]!, channels[2]!];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(foreground: string, background: string): number {
  const values = [luminance(foreground), luminance(background)].sort((a, b) => b - a);
  return (values[0]! + 0.05) / (values[1]! + 0.05);
}

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

function percent(value: string): number {
  const match = value.match(/([0-9.]+)%/);
  if (!match) throw new Error(`Expected percentage mix, got ${value}`);
  return Number(match[1]!) / 100;
}

function token(theme: Palette, name: string): string {
  const value = theme[name];
  if (value === undefined) throw new Error(`Missing theme token: ${name}`);
  return value;
}

const base = declarations(":root");
const themes = {
  "Safelight dark": base,
  "Safelight system light": { ...base, ...declarations(':root:not([data-theme="dark"])') },
  "Safelight light": { ...base, ...declarations(':root[data-theme="light"]') },
  "Mold dark": { ...base, ...declarations(':root[data-theme-family="mold"]') },
  "Mold system light": {
    ...base,
    ...declarations(':root[data-theme-family="mold"]'),
    ...declarations(':root[data-theme-family="mold"]:not([data-theme="dark"])'),
  },
  "Mold light": {
    ...base,
    ...declarations(':root[data-theme-family="mold"]'),
    ...declarations(':root[data-theme-family="mold"][data-theme="light"]'),
  },
};

describe("shared theme contrast", () => {
  for (const [name, theme] of Object.entries(themes)) {
    it(`${name} keeps ink and semantic text at WCAG AA`, () => {
      // Primary and secondary ink must stay readable on every raised plane
      // (bath, bench, and the new surface card tone).
      for (const background of [theme.bath, theme.bench, theme.surface]) {
        expect(contrast(token(theme, "rebate"), background!), "primary ink").toBeGreaterThanOrEqual(
          4.5,
        );
        const ink2 = composite(theme.rebate!, background!, percent(theme["ink-2"]!));
        expect(contrast(ink2, background!), "secondary ink").toBeGreaterThanOrEqual(4.5);
      }
      // Semantic accents carry text/icons on the two chrome planes.
      for (const background of [theme.bath, theme.bench]) {
        for (const semantic of ["halide", "safelight", "stop"]) {
          expect(contrast(theme[semantic]!, background!), semantic).toBeGreaterThanOrEqual(4.5);
        }
        // Tertiary ink is hint-only: large-text / non-text 3:1 is the bar.
        const ink3 = composite(theme.rebate!, background!, percent(theme["ink-3"]!));
        expect(contrast(ink3, background!), "tertiary ink").toBeGreaterThanOrEqual(3);
      }
    });

    it(`${name} keeps controls, focus, and accent fills distinguishable`, () => {
      for (const background of [theme.bath, theme.bench]) {
        // The control edge (--ce, overridden per light family) is a divider,
        // not text — it only has to read as a visible separation.
        const edge = composite(theme.rebate!, background!, percent(theme.ce!));
        expect(contrast(edge, background!), "control edge").toBeGreaterThanOrEqual(2);
        expect(contrast(theme.safelight!, background!), "focus indicator").toBeGreaterThanOrEqual(
          3,
        );
      }
      expect(
        contrast(theme["on-accent"]!, theme.safelight!),
        "primary action",
      ).toBeGreaterThanOrEqual(4.5);
      expect(
        contrast(theme["on-accent"]!, theme.stop!),
        "destructive action",
      ).toBeGreaterThanOrEqual(4.5);
    });
  }

  it("darkens the control edge in every light family (lights on)", () => {
    // The light "lights on" families raise --ce above the dark base mix so
    // dividers stay legible on paper. This asserts the override actually lands.
    const baseEdge = percent(base.ce!);
    for (const name of ["Safelight system light", "Safelight light", "Mold light"] as const) {
      expect(percent(themes[name].ce!), name).toBeGreaterThan(baseEdge);
    }
  });
});
