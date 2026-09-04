import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import routerSource from "../router.ts?raw";
import sidebarSource from "../components/shell/Sidebar.vue?raw";
import paletteSource from "../components/shell/CommandPalette.vue?raw";
import inspectorSource from "../components/create/InspectorPanel.vue?raw";
import appSource from "../App.vue?raw";
import chainJobsSource from "../stores/chainJobs.ts?raw";
import composerSource from "../components/create/ComposerCard.vue?raw";
import expandSource from "../components/generate/ExpandControl.vue?raw";
import advancedSource from "../components/create/AdvancedSettings.vue?raw";
import generateViewSource from "../views/GenerateView.vue?raw";
import strengthSource from "@studio/lib/strengthSemantics.ts?raw";
import { ENGINE_KEY_SCHEMAS, SECTIONS } from "./settingsSchema";

/**
 * The binding lexicon (docs/design/README.md §02): plain words in sans,
 * technical truth in mono, on the same row. This pins the words every
 * destination is called by — in the sidebar, the router's titles, the
 * command palette, and the native menu — the native File/Generate items,
 * the shell's finished-work toasts, Settings' section and row labels, the
 * inspector's primary controls, and the Styles/Machines views, so a rename
 * on one surface cannot leave the others behind.
 */
const DESTINATIONS = ["New image", "Queue", "My images", "Styles", "Machines"] as const;

/** Words that may never be a destination's primary label again. */
const NEVER_A_DESTINATION = ["Create", "Library", "Models", "Hosts", "Gallery", "Catalog"];

// vitest runs from desktop/, so the Rust source is read relative to it.
const menuSource = readFileSync("src-tauri/src/menu.rs", "utf8");

function quoted(source: string): string[] {
  return [...source.matchAll(/"([^"\n]+)"/g)].map((m) => m[1]!);
}

/**
 * Template text only: strips the `<script>` block, every tag's attributes,
 * and every `{{ … }}` interpolation, so an identifier, a route path, or a
 * `data-test` hook can never trip a never-say scan of what a person reads.
 */
function templateText(source: string): string {
  const template = source.replace(/<script[\s\S]*?<\/script>/g, "");
  return template
    .replace(/<[^>]*>/g, " ")
    .replace(/\{\{[\s\S]*?\}\}/g, " ")
    .replace(/\s+/g, " ");
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

describe("lexicon — the native menu", () => {
  it("says New Image and New Clip in File", () => {
    expect(menuSource).toContain('("new-generation", "New Image", Some("N"))');
    expect(menuSource).toContain('("new-sequence", "New Clip", None)');
  });

  it("says Generate, Write More For Me, Surprise Me, and Stop", () => {
    for (const item of [
      '("generate", "Generate", "Return")',
      '("expand-prompt", "Write More For Me", "E")',
      '("randomize-seed", "Surprise Me", "R")',
      '("cancel-job", "Stop", ".")',
    ]) {
      expect(menuSource).toContain(item);
    }
  });

  it("keeps the retired verbs out of every menu label", () => {
    for (const old of [
      "New Generation",
      "New Sequence",
      "Expand Prompt",
      "Randomize Seed",
      "Cancel Job",
    ]) {
      expect(quoted(menuSource), old).not.toContain(old);
    }
  });
});

describe("lexicon — finished work is saved to My images", () => {
  it("says so in the print toast", () => {
    expect(appSource).toContain('"Generated — saved to My images"');
  });

  it("says so in the clip toast, and calls it a clip", () => {
    expect(chainJobsSource).toContain('"Clip ready — saved to My images"');
    expect(chainJobsSource).toContain('"Clip failed"');
  });

  it("keeps Library and Sequence out of both files' user text", () => {
    for (const source of [appSource, chainJobsSource]) {
      for (const old of ["Library", "Sequence"]) {
        expect(
          quoted(source).some((s) => s.includes(old)),
          old,
        ).toBe(false);
      }
    }
  });
});

describe("lexicon — Settings", () => {
  it("names its sections in plain words", () => {
    expect(SECTIONS.map((s) => s.label)).toEqual([
      "Look",
      "Defaults for new images",
      "Write more for me",
      "Machines",
      "Saving pictures & clips",
      "My images & trash",
      "Style licences",
      "Phone pairing",
      "Speed & memory",
      "Accounts & tokens",
      "Profiles",
      "Advanced",
      "Updates & about",
    ]);
  });

  it("labels every Defaults-for-new-images row in the lexicon, keys unchanged", () => {
    const rows = ENGINE_KEY_SCHEMAS.filter((s) => s.section === "generation").map((s) => [
      s.key,
      s.label,
    ]);
    expect(rows).toEqual([
      ["default_model", "Style to start with"],
      ["default_width", "Width"],
      ["default_height", "Height"],
      ["default_steps", "Detail"],
      ["default_negative_prompt", "Words to avoid"],
      ["embed_metadata", "Keep the recipe in the file"],
      ["t5_variant", "How FLUX reads your words"],
      ["qwen3_variant", "How Flux.2 and Z-Image read your words"],
    ]);
  });

  it("keeps the engine words out of that section's labels and help", () => {
    const copy = ENGINE_KEY_SCHEMAS.filter((s) => s.section === "generation")
      .map((s) => `${s.label} ${s.help ?? ""}`)
      .join(" ")
      .toLowerCase();
    for (const old of ["model", "steps", "print", "composer", "denoise", "quantization", "vram"]) {
      expect(copy, old).not.toContain(old);
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

  it("counts Detail in passes and calls guidance Stick to my words", () => {
    expect(inspectorSource).toContain('label="Detail"');
    expect(inspectorSource).toContain("`${form.steps} passes`");
    expect(inspectorSource).toContain('label="Stick to my words"');
    expect(inspectorSource).not.toContain('label="Prompt strength"');
  });

  it("names the 3-D group and its controls in plain words", () => {
    expect(inspectorSource).toContain(">3-D object<");
    expect(inspectorSource).toContain('label="Surface detail"');
    expect(inspectorSource).toContain('label="How tight to the photo"');
    expect(inspectorSource).toContain("Simplify to");
    expect(inspectorSource).toContain('placeholder="keep every detail"');
    for (const old of ["Octree detail", "Iso threshold", "Target faces", "keep raw surface"]) {
      expect(inspectorSource, old).not.toContain(old);
    }
  });

  it("titles the add-on looks section without the engine word", () => {
    expect(advancedSource).toContain('title="Add-on looks"');
    expect(advancedSource).not.toContain('title="LoRA stack"');
    expect(advancedSource).not.toContain('summary="Style adapters"');
  });
});

describe("lexicon — the composer", () => {
  it("offers to write more for me, on ⌘E", () => {
    expect(expandSource).toContain('"Write more for me"');
    expect(expandSource).toContain("'Write more for me'");
    expect(expandSource).not.toContain('"Expand"');
    expect(expandSource).not.toContain("'Expand prompt'");
  });

  it("keeps Generate to one word and says the queue depth beside it", () => {
    expect(generateViewSource).toContain(
      'const buttonLabel = computed(() => (submissionPlanning.value ? "Cancel" : "Generate"));',
    );
    expect(generateViewSource).toContain("`+${generation.pending.length} queued`");
    expect(composerSource).toContain('data-test="generate-queued-note"');
  });
});

describe("lexicon — how much to change it", () => {
  it("is the one label for the source-strength wire field", () => {
    expect(strengthSource).toContain('const LABEL = "How much to change it";');
    for (const old of ["Denoise strength", "Source strength"]) {
      expect(strengthSource, old).not.toContain(old);
    }
  });
});

describe("lexicon — Styles and Machines", () => {
  // Template text only: `host` lives in identifiers and route paths all over
  // these files, and none of that is copy a person reads.
  const surfaces: [string, string][] = [
    ["CatalogCard", "../components/models/CatalogCard.vue"],
    ["CatalogTableRow", "../components/models/CatalogTableRow.vue"],
    ["CatalogDetailDrawer", "../components/models/CatalogDetailDrawer.vue"],
    ["DownloadTargetDialog", "../components/models/DownloadTargetDialog.vue"],
    ["ModelTableRow", "../components/models/ModelTableRow.vue"],
    ["DownloadsTray", "../components/models/DownloadsTray.vue"],
  ];
  const NEVER_SAID = [/\bhost\b/i, /\bmodel page\b/i, /\bPull\b/, /\binstalled\b/i, /\bInstall\b/];

  const sources = new Map(
    surfaces.map(([name, path]) => [name, readFileSync(new URL(path, import.meta.url), "utf8")]),
  );

  it.each(surfaces.map(([name]) => name))("keeps the retired words out of %s", (name) => {
    const text = templateText(sources.get(name)!);
    for (const banned of NEVER_SAID) {
      expect(banned.test(text), `${name}: ${banned}`).toBe(false);
    }
  });

  it("says ● ready and Getting it… on every catalog surface", () => {
    expect(sources.get("CatalogDetailDrawer")!).toContain("● ready");
    expect(sources.get("CatalogDetailDrawer")!).toContain('"Getting it…"');
    expect(sources.get("CatalogDetailDrawer")!).not.toContain('"Pulling…"');
  });

  it("asks where a style should go, and offers Get it or a repair", () => {
    const dialog = sources.get("DownloadTargetDialog")!;
    expect(dialog).toContain("Where should ${props.modelName} go?");
    expect(dialog).toContain('"Get it" : "Already here · repair"');
  });

  it("says the style's page, never the model's", () => {
    for (const name of ["CatalogCard", "ModelTableRow"]) {
      expect(sources.get(name)!, name).toContain("'s page");
    }
    expect(sources.get("CatalogCard")!).toContain("Open the style's page");
  });

  it("puts plain words on every download status", () => {
    const tray = sources.get("DownloadsTray")!;
    for (const word of ["Downloading", "Waiting", "Finished", "Failed", "Cancelled"]) {
      expect(tray, word).toContain(`"${word}"`);
    }
  });
});
