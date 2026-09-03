import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import routerSource from "../router.ts?raw";
import sidebarSource from "../components/shell/Sidebar.vue?raw";
import paletteSource from "../components/shell/CommandPalette.vue?raw";
import inspectorSource from "../components/create/InspectorPanel.vue?raw";

/**
 * The binding lexicon (docs/design/README.md §02): plain words in sans,
 * technical truth in mono, on the same row. This pins the words every
 * destination is called by — in the sidebar, the router's titles, the
 * command palette, and the native menu — and the inspector's primary
 * labels, so a rename on one surface cannot leave the others behind.
 */
const DESTINATIONS = ["New image", "Queue", "My images", "Styles", "Machines"] as const;

/** Words that may never be a destination's primary label again. */
const NEVER_A_DESTINATION = ["Create", "Library", "Models", "Hosts", "Gallery", "Catalog"];

// vitest runs from desktop/, so the Rust source is read relative to it.
const menuSource = readFileSync("src-tauri/src/menu.rs", "utf8");

function quoted(source: string): string[] {
  return [...source.matchAll(/"([^"\n]+)"/g)].map((m) => m[1]!);
}

describe("lexicon — destinations", () => {
  it("titles every route in the lexicon", () => {
    const titles = [...routerSource.matchAll(/meta: \{ title: "([^"]+)" \}/g)].map((m) => m[1]);
    expect(titles).toEqual([...DESTINATIONS, "Settings"]);
  });

  it("labels the sidebar's five destinations in the lexicon, in ⌘1–⌘5 order", () => {
    const labels = [...sidebarSource.matchAll(/label: "([^"]+)"/g)].map((m) => m[1]);
    expect(labels.slice(0, DESTINATIONS.length)).toEqual([...DESTINATIONS]);
  });

  it("names the native View menu's destinations exactly like the sidebar", () => {
    const items = [...menuSource.matchAll(/\("nav:([^"]+)", "([^"]+)", "(\d)"\)/g)].map((m) => ({
      route: m[1],
      label: m[2],
      digit: m[3],
    }));
    // The first occurrence is the constant; the test below it repeats it.
    const menu = items.slice(0, DESTINATIONS.length);
    expect(menu.map((i) => i.label)).toEqual([...DESTINATIONS]);
    expect(menu.map((i) => i.digit)).toEqual(["1", "2", "3", "4", "5"]);
    expect(menu.map((i) => i.route)).toEqual([
      "/create",
      "/queue",
      "/library",
      "/models",
      "/machines",
    ]);
  });

  it("names the palette's destinations in the lexicon, never 'Go to <old name>'", () => {
    const titles = [...paletteSource.matchAll(/title: "([^"]+)"/g)].map((m) => m[1]!);
    for (const word of DESTINATIONS) expect(titles).toContain(word);
    for (const old of NEVER_A_DESTINATION) {
      expect(
        titles.some((t) => t === old || t === `Go to ${old}`),
        old,
      ).toBe(false);
    }
  });

  it("never lets an old name back in as a primary label", () => {
    for (const source of [routerSource, sidebarSource, menuSource]) {
      for (const old of NEVER_A_DESTINATION) {
        expect(quoted(source), old).not.toContain(old);
      }
    }
  });
});

describe("lexicon — the inspector", () => {
  it("says Repeat this look, Keep | Surprise me, and Add-on looks", () => {
    expect(inspectorSource).toContain(">Repeat this look<");
    expect(inspectorSource).toMatch(/seed-mode-random"[\s\S]{0,300}?Surprise me/);
    expect(inspectorSource).toMatch(/seed-mode-fixed"[\s\S]{0,300}?Keep/);
    expect(inspectorSource).not.toContain(">Seed<");
  });
});
