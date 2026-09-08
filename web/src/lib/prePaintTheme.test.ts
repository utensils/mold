import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { migrateLegacyTheme, resolveTheme, type ThemeId } from "@ui/theme";

/*
 * The anti-FOUC scripts are inlined in the two HTML entry points, so they
 * cannot import ui/theme.ts — they are a hand-written second implementation of
 * the same decision, and until now nothing checked them against the first.
 *
 * That mattered: they each carried their own copy of the old pairing table,
 * which had to be edited in lockstep with ui/theme.ts by memory alone. This
 * runs the real script text against the real contract over every saved shape
 * either surface can hold.
 */

const ENTRIES = [
  { name: "web/index.html", path: "index.html", key: "mold.web.theme.v1" },
  {
    name: "desktop/index.mobile.html",
    path: "../desktop/index.mobile.html",
    key: "mold.mobile.settings.v1",
  },
] as const;

/** Pull the IIFE out of the entry point and run it against a fake document. */
function runPrePaint(
  html: string,
  saved: string | null,
  prefersLight: boolean,
): string {
  const script = html.match(
    /<script>\s*(\/\/ Apply the persisted[\s\S]*?\}\)\(\);)\s*<\/script>/,
  );
  if (!script) throw new Error("pre-paint script not found");
  const root = { dataset: {} as Record<string, string> };
  const fn = new Function(
    "localStorage",
    "window",
    "document",
    `${script[1]}; return document.documentElement.dataset.theme;`,
  );
  return fn(
    { getItem: (k: string) => (k === null ? null : saved) },
    {
      matchMedia: (q: string) => ({
        matches: q.includes("light") && prefersLight,
      }),
    },
    { documentElement: root },
  ) as string;
}

/** Every saved shape the two surfaces can hold, old and new. */
const SAVED: { label: string; value: unknown }[] = [
  { label: "empty storage", value: null },
  ...(
    [
      "mocha-dark",
      "mocha-light",
      "safelight-dark",
      "safelight-light",
      "blueprint-dark",
      "blueprint-light",
      "graphite-dark",
      "graphite-light",
      "nebula-dark",
      "nebula-light",
    ] as ThemeId[]
  ).flatMap((theme) => [
    { label: `${theme} pinned`, value: { theme, matchSystem: false } },
    { label: `${theme} on system`, value: { theme, matchSystem: true } },
  ]),
  // The pre-tone ids.
  ...["mocha", "safelight", "blueprint", "graphite", "porcelain", "nebula"].map(
    (theme) => ({
      label: `renamed ${theme}`,
      value: { theme, matchSystem: false },
    }),
  ),
  // The pre-redesign appearance + family pair, under both key spellings.
  ...["dark", "light", "system"].flatMap((theme) =>
    ["mold", "safelight"].flatMap((family) => [
      { label: `legacy ${theme}/${family}`, value: { theme, family } },
      {
        label: `legacy ${theme}/${family} (themeFamily)`,
        value: { theme, themeFamily: family },
      },
    ]),
  ),
  { label: "garbage", value: { theme: "sepia", family: "vaporwave" } },
];

describe.each(ENTRIES)("$name pre-paint script", ({ path, key }) => {
  const html = readFileSync(path, "utf8");

  it("reads the surface's own storage key", () => {
    expect(html).toContain(`localStorage.getItem("${key}")`);
  });

  it.each(SAVED)(
    "resolves $label exactly as resolveTheme does",
    ({ value }) => {
      const raw = value === null ? null : JSON.stringify(value);
      for (const prefersLight of [true, false]) {
        const painted = runPrePaint(html, raw, prefersLight);

        const record = (value ?? {}) as Record<string, unknown>;
        const saved =
          value === null
            ? { theme: "safelight-dark" as ThemeId, matchSystem: false }
            : migrateLegacyTheme(
                record.theme,
                record.family ?? record.themeFamily,
                record.matchSystem,
              );

        expect(
          painted,
          `${JSON.stringify(value)} prefersLight=${prefersLight}`,
        ).toBe(resolveTheme(saved.theme, saved.matchSystem, prefersLight));
      }
    },
  );
});
