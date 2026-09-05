import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/*
 * Machines is a master/detail workspace: a 326px list beside the pane. At the
 * app's own `minWidth` (1080) the pane is 484px, so its panes must lay out
 * against THEMSELVES. These are source contracts because the defect is a class
 * string — a fixed track or a viewport breakpoint renders "correctly" under
 * jsdom, which has no layout, and only clips on a real 1080px window.
 */

function source(name: string): string {
  return readFileSync(fileURLToPath(new URL(name, import.meta.url)), "utf8");
}

describe("Machines pane layout", () => {
  it("never sizes a pane against the viewport", () => {
    // `minWidth: 1080` means every Tailwind viewport breakpoint below `xl` is
    // permanently true, so `lg:grid-cols-2` is not responsive — it is two
    // columns in a 484px pane, always.
    for (const file of ["HostDetailView.vue", "MachinesView.vue", "RunPodView.vue"]) {
      const offenders = source(file)
        .split("\n")
        .map((line, index) => [index + 1, line] as const)
        .filter(([, line]) => /\b(sm|md|lg|xl|2xl):/.test(line))
        .map(([n, line]) => `${file}:${n} ${line.trim()}`);
      expect(offenders).toEqual([]);
    }
  });

  it("pairs Storage with Downloads only once the pane itself is wide enough", () => {
    const detail = source("HostDetailView.vue");
    expect(detail).toMatch(/\.host-pair-shell\s*\{[^}]*container-type:\s*inline-size/s);
    expect(detail).toMatch(
      /@container \(min-width:[^)]*\)\s*\{[\s\S]*?\.host-pair\s*\{[^}]*grid-template-columns/s,
    );
  });

  it("keeps both RunPod tracks shrinkable so the console is not squeezed out", () => {
    const runpod = source("RunPodView.vue");
    expect(runpod).not.toContain("grid-cols-[340px_1fr]");
    expect(runpod.match(/grid-cols-\[minmax\(0,340px\)_minmax\(0,1fr\)\]/g)).toHaveLength(2);
  });

  it("wraps the RunPod pod row instead of collapsing its name and status", () => {
    expect(source("RunPodView.vue")).toContain('<div class="flex flex-wrap items-center gap-3">');
  });
});

describe("meter widths", () => {
  it("are rounded at the binding, never carried as raw floats into style", () => {
    for (const file of ["HostDetailView.vue", "MachinesView.vue"]) {
      const raw = source(file)
        .split("\n")
        .filter((line) => /width: `\$\{/.test(line))
        .filter((line) => !/width: `\$\{Math\.round\(/.test(line));
      expect(raw).toEqual([]);
    }
  });
});
