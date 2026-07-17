import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

const css = readFileSync("src/styles/tokens.css", "utf8");

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

describe("desktop theme contrast", () => {
  for (const [name, theme] of Object.entries(themes)) {
    it(`${name} keeps text and semantic colors at WCAG AA`, () => {
      for (const background of [theme.bath, theme.bench]) {
        expect(contrast(token(theme, "rebate"), background!), "primary ink").toBeGreaterThanOrEqual(
          4.5,
        );
        for (const token of ["ink-2", "ink-3"]) {
          const color = composite(theme.rebate!, background!, percent(theme[token]!));
          expect(contrast(color, background!), token).toBeGreaterThanOrEqual(4.5);
        }
        for (const token of ["halide", "safelight", "stop"]) {
          expect(contrast(theme[token]!, background!), token).toBeGreaterThanOrEqual(4.5);
        }
      }
    });

    it(`${name} keeps controls, focus, and accent fills distinguishable`, () => {
      for (const background of [theme.bath, theme.bench]) {
        const edge = composite(theme.rebate!, background!, percent(theme["control-edge"]!));
        expect(contrast(edge, background!), "control edge").toBeGreaterThanOrEqual(3);
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

  it("uses a light, readable empty work surface in light themes", () => {
    for (const name of [
      "Safelight system light",
      "Safelight light",
      "Mold system light",
      "Mold light",
    ] as const) {
      const theme = themes[name];
      expect(luminance(theme["empty-surface"]!), name).toBeGreaterThan(0.9);
      expect(contrast(theme.rebate!, theme["empty-surface"]!), name).toBeGreaterThanOrEqual(4.5);
    }
  });
});
