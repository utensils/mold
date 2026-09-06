import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

/*
 * The pre-redesign darkroom vocabulary (desk/bath/bench/rebate, safelight and
 * halide, the utility/body/display type roles, the chrome/media radii) lives
 * on only in ui/tokens.css's LEGACY BRIDGE, and only for the web SPA and the
 * phone surface. Desktop, ui/, and studio/ speak --mold-* and the utility
 * names desktop/src/styles/tokens.css maps from it. This is the migration's
 * definition of done.
 */

const ROOTS = ["src", "../ui", "../studio"];
// The desktop @theme map and the shared token sheet DEFINE the vocabulary;
// the phone and the font sheets sit on the bridge by design.
const SKIP = new Set(["src/mobile", "src/styles/tokens.css", "../ui/tokens.css", "../ui/fonts"]);
const EXTENSIONS = new Set([".vue", ".ts", ".css"]);

/** Whole-token boundaries: `--edge` must not match `--edge-code`'s successor
 * nor a data-test id like `generation-edge-code`. */
const BOUND = "(?<![\\w-])";
const END = "(?![\\w-])";
const LEGACY_PATTERNS = [
  new RegExp(
    `${BOUND}--(desk|bath|bench|rebate|halide|safelight|stop|ink|ink-2|ink-3|danger|edge|ce|sel-[a-z]+|card-hi|grad|print|on-media|on-status|f-(display|body|mono)|radius-(control|control-sm|control-lg|card|card-lg|pill)|control-edge|empty-surface|dur-(quick|base|slow)|ease)${END}`,
  ),
  new RegExp(
    `${BOUND}(bg|text|border|fill|stroke|ring|from|to|via|shadow|accent|caret|divide|placeholder)-(desk|bath|bench|rebate|halide|safelight|stop|ink|ink-2|ink-3|edge|ce|control-edge|print-surface|empty-surface|card-hi)${END}`,
  ),
  new RegExp(
    `${BOUND}(font-(display|body|utility)|text-(display|display-sm|display-lg|body|body-lg|caption|data|edge-code)|rounded-(chrome|media|pill|card|card-lg)|shadow-raised|edge-code|data-mono|kbd-hint|grain-shimmer)${END}`,
  ),
];

function* walk(dir: string): Generator<string> {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (SKIP.has(path) || entry === "node_modules" || entry.startsWith("dist")) continue;
    if (statSync(path).isDirectory()) yield* walk(path);
    else if (EXTENSIONS.has(path.slice(path.lastIndexOf(".")))) yield path;
  }
}

function offenders(text: string): string[] {
  const hits: string[] = [];
  text.split("\n").forEach((line, index) => {
    // `font-display:` is a CSS property, not the retired type role.
    const probe = line.replace(/font-display:/g, "");
    for (const pattern of LEGACY_PATTERNS) {
      const match = probe.match(pattern);
      if (match) hits.push(`${index + 1}: ${match[0]}`);
    }
  });
  return hits;
}

/** Two font-size utilities on one element: the later one wins silently. */
const SIZE_UTILITY = /(?<![\w:-])text-(micro|xs|sm|base|md|lg|xl)(?![\w-])/g;
function doubleSized(text: string): string[] {
  // Static class attributes only: a `:class` ternary names alternatives.
  return [...text.matchAll(/(?<![:\w-])class="([^"]*)"/g)]
    .map((m) => m[1]!)
    .filter((attr) => [...attr.matchAll(SIZE_UTILITY)].length > 1);
}

describe("font-size utilities", () => {
  it("are recognised by the guard (positive control)", () => {
    expect(doubleSized('class="font-mono text-xs text-sm"')).toHaveLength(1);
    expect(doubleSized('class="text-xs sm:text-sm"')).toHaveLength(0);
    expect(doubleSized('class="text-micro text-fg-dim"')).toHaveLength(0);
  });

  it("never stack on one element in desktop sources", () => {
    const found: string[] = [];
    for (const file of walk("src")) {
      if (!file.endsWith(".vue")) continue;
      for (const hit of doubleSized(readFileSync(file, "utf8"))) found.push(`${file}: ${hit}`);
    }
    expect(found).toEqual([]);
  });
});

/*
 * README §7: a component references --mold-radius-1/2/3 and the seven-step type
 * scale, never a literal. A literal survives a theme swap unchanged, so Nebula's
 * square corners and Blueprint's 10.5px micro never reach it. `0` is not a
 * literal here, and a line may opt out with `/* literal: <reason> *​/` beside it.
 */
const LITERAL_STYLE = /\b(border-radius|font-size)\s*:\s*[^;{}]*\b\d+(?:\.\d+)?px/;

function literalStyles(text: string): string[] {
  const hits: string[] = [];
  text.split("\n").forEach((line, index) => {
    if (line.includes("/* literal:")) return;
    const match = line.match(LITERAL_STYLE);
    if (match) hits.push(`${index + 1}: ${match[0].trim()}`);
  });
  return hits;
}

/*
 * A RATCHET, not an allowlist. These files still carry literals from before the
 * token scale existed. Most are shared ui/ and studio/ primitives the redesign
 * has yet to reach, where converting blind would move web and the phone by a
 * pixel apiece with nothing watching. Every create/ file is converted — the
 * New-image restructure rebuilt them — so none is left here. Each remaining
 * file may keep the count it has and never more, and a file that reaches zero
 * must leave the table.
 */
const UNCONVERTED: Record<string, number> = {
  "../ui/components/AccordionSection.vue": 2,
  "../ui/components/ActionBlocker.vue": 4,
  "../ui/components/BadgePill.vue": 1,
  "../ui/components/CatalogLayoutToggle.vue": 1,
  "../ui/components/Chip.vue": 1,
  "../ui/components/EmptyStateBlock.vue": 3,
  "../ui/components/LiveActivityList.vue": 6,
  "../ui/components/MediaTile.vue": 3,
  "../ui/components/MeshExportDialog.vue": 9,
  "../ui/components/MeshGeometryFields.vue": 7,
  "../ui/components/NavItem.vue": 1,
  "../ui/components/PalettePanel.vue": 7,
  "../ui/components/ProgressBar.vue": 1,
  "../ui/components/ProgressRing.vue": 1,
  "../ui/components/ResolutionSelector.vue": 4,
  "../ui/components/SegmentedControl.vue": 3,
  "../ui/components/ShapePicker.vue": 2,
  "../ui/components/SheetPanel.vue": 5,
  "../ui/components/Stepper.vue": 2,
  "../ui/components/ToastShelf.vue": 4,
  "../ui/components/Tooltip.vue": 2,
  "../ui/components/UpscaleDialog.vue": 12,
  "../ui/components/VideoDurationSlider.vue": 1,
  "../ui/components/VideoExportDialog.vue": 11,
  "../studio/components/IdentityPhotoWell.vue": 1,
  "../studio/components/ImageDropWell.vue": 7,
  "../studio/components/LicenseAcceptanceDialog.vue": 4,
  "../studio/components/MeshViewer.vue": 5,
  "../studio/components/MinimaxH3AuthoringPanel.vue": 10,
  "../studio/components/MinimaxH3InventoryPanel.vue": 15,
  "../studio/components/NotificationsCenter.vue": 14,
  "../studio/components/QueueEntryDetail.vue": 14,
  "../studio/components/QueuePlanWorkList.vue": 4,
  "../studio/components/ReferenceCropEditor.vue": 3,
  "../studio/components/SourceMediaWells.vue": 1,
  "../studio/components/SwipeActionRow.vue": 2,
};

describe("literal radii and font sizes", () => {
  it("are recognised by the guard (positive control)", () => {
    expect(literalStyles("  border-radius: 9px;")).toHaveLength(1);
    expect(literalStyles("  font-size: 13.5px;")).toHaveLength(1);
    expect(literalStyles("  border-radius: 22px 22px 0 0;")).toHaveLength(1);
    expect(literalStyles("  border-radius: var(--mold-radius-2);")).toHaveLength(0);
    expect(literalStyles("  font-size: 0;")).toHaveLength(0);
    expect(literalStyles("  padding: 2px 6px;")).toHaveLength(0);
    expect(literalStyles("  border-radius: 3px; /* literal: QR quiet zone */")).toHaveLength(0);
  });

  it("appear only where the ratchet still allows them", () => {
    const counts: Record<string, number> = {};
    for (const root of ROOTS) {
      for (const file of walk(root)) {
        // Tests quote the literals they refuse; the guard is about shipped styles.
        if (file.endsWith(".test.ts")) continue;
        const hits = literalStyles(readFileSync(file, "utf8"));
        if (hits.length) counts[file] = hits.length;
      }
    }
    const over = Object.entries(counts)
      .filter(([file, count]) => count > (UNCONVERTED[file] ?? 0))
      .map(([file, count]) => `${file}: ${count} > ${UNCONVERTED[file] ?? 0}`);
    expect(over).toEqual([]);
    const cleared = Object.keys(UNCONVERTED).filter((file) => !counts[file]);
    expect(cleared).toEqual([]);
  });
});

describe("legacy token vocabulary", () => {
  it("is recognised by the guard (positive control)", () => {
    // One hit per pattern family: the colour utilities and the type roles.
    expect(offenders('class="bg-bench text-ink-3 font-utility rounded-chrome"')).toHaveLength(2);
    expect(offenders("color: var(--safelight);")).toHaveLength(1);
    expect(offenders("border-radius: var(--radius-control);")).toHaveLength(1);
    expect(offenders("font-display: block;")).toHaveLength(0);
    expect(offenders('data-test="generation-edge-code"')).toHaveLength(0);
    expect(offenders("--mold-border-control")).toHaveLength(0);
  });

  it("appears nowhere in desktop, ui, or studio sources", () => {
    const found: string[] = [];
    for (const root of ROOTS) {
      for (const file of walk(root)) {
        if (file === "src/styles/tokens.legacy.test.ts") continue;
        for (const hit of offenders(readFileSync(file, "utf8"))) found.push(`${file}:${hit}`);
      }
    }
    expect(found).toEqual([]);
  });
});
