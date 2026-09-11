import { describe, expect, it } from "vitest";
import {
  CONTROL_WORDS,
  DESTINATIONS,
  NEVER_A_DESTINATION,
  NEVER_SAID_ON_STYLES_AND_MACHINES,
  RETIRED_CONTROL_LABELS,
  RETIRED_STYLES_LABELS,
  STYLES_WORDS,
  templateText,
} from "@studio/lib/lexicon";
import { WORKSPACES } from "./workspaces";
import modelsPage from "../pages/ModelsPage.vue?raw";
import machinesPage from "../pages/MachinesPage.vue?raw";
import catalogCard from "../components/CatalogCard.vue?raw";
import installedRow from "../components/models/InstalledModelRow.vue?raw";
import installTargetDialog from "../components/models/ModelInstallTargetDialog.vue?raw";
import detailDrawer from "../components/models/ModelDetailDrawer.vue?raw";
import hostCard from "../components/machines/HostCard.vue?raw";
import controlsAside from "../components/create/ControlsAside.vue?raw";
import advancedDrawer from "../components/create/AdvancedDrawer.vue?raw";
import hostRoutingPicker from "../components/create/HostRoutingPicker.vue?raw";
import composerCard from "../components/create/ComposerCard.vue?raw";
import loraPicker from "../components/LoraPicker.vue?raw";
import lightbox from "../components/gallery/Lightbox.vue?raw";
import appNav from "../components/shell/AppNav.vue?raw";
import commands from "./commands.ts?raw";

/*
 * The binding lexicon (docs/design/README.md §2) on the web surface. Desktop's
 * `lexicon.test.ts` reads the same tables from `studio/lib/lexicon.ts`; this
 * file reads web's own sources, because a desktop test failing for a web edit
 * is the wrong signal and a web surface no test reads is where the retired
 * words came back.
 */

describe("lexicon — destinations", () => {
  it("names the five workspaces in the lexicon, in order", () => {
    expect(WORKSPACES.map((w) => w.label)).toEqual([...DESTINATIONS]);
  });

  it("never uses a retired word as a workspace label", () => {
    for (const label of WORKSPACES.map((w) => w.label)) {
      expect(NEVER_A_DESTINATION).not.toContain(label);
    }
  });
});

describe("lexicon — Styles", () => {
  it("has two shelves, Ready to use and Browse more, and one verb, Get it", () => {
    expect(modelsPage).toContain(`"${STYLES_WORDS.ready}"`);
    expect(modelsPage).toContain(`"${STYLES_WORDS.browse}"`);
    for (const retired of RETIRED_STYLES_LABELS) {
      expect(modelsPage, retired).not.toMatch(
        new RegExp(`label: "${retired}"`),
      );
    }
    expect(catalogCard).toContain(`"${STYLES_WORDS.get}"`);
    expect(catalogCard).toContain(STYLES_WORDS.readyBadge);
    expect(installedRow).toContain(`"${STYLES_WORDS.get}"`);
    expect(installTargetDialog).toContain(`"${STYLES_WORDS.get}"`);
    expect(detailDrawer).toContain(`"${STYLES_WORDS.get}"`);
    expect(commands).toContain(`section: "${STYLES_WORDS.get}"`);
  });

  it.each([
    ["ModelsPage", modelsPage],
    ["CatalogCard", catalogCard],
    ["InstalledModelRow", installedRow],
    ["ModelInstallTargetDialog", installTargetDialog],
    ["ModelDetailDrawer", detailDrawer],
    ["MachinesPage", machinesPage],
    ["HostCard", hostCard],
  ])("keeps the retired words out of %s", (_name, source) => {
    const text = templateText(source);
    for (const banned of NEVER_SAID_ON_STYLES_AND_MACHINES) {
      expect(text, String(banned)).not.toMatch(banned);
    }
  });
});

describe("lexicon — New image controls", () => {
  it("names the technical controls in plain words", () => {
    expect(controlsAside).toContain(`label="${CONTROL_WORDS.guidance}"`);
    expect(controlsAside).toContain(`label="${CONTROL_WORDS.octree}"`);
    expect(controlsAside).toContain(`label="${CONTROL_WORDS.isoThreshold}"`);
    expect(controlsAside).toContain(CONTROL_WORDS.resetToStyleDefaults);
    expect(controlsAside).toContain('aria-label="Seed number"');
    expect(controlsAside).toContain("passes`");
    expect(controlsAside).not.toContain("steps`");
    expect(controlsAside).toContain(">Make<");
    expect(controlsAside).not.toContain(">Batch<");
    expect(hostRoutingPicker).toContain(
      `aria-label="${CONTROL_WORDS.whereItRuns}"`,
    );
    expect(advancedDrawer).toContain(`title="${CONTROL_WORDS.loras}"`);
    expect(loraPicker).toContain(CONTROL_WORDS.loras);
    expect(composerCard).toContain(`"${CONTROL_WORDS.expand}"`);
    expect(composerCard).toContain("} passes`");
    expect(composerCard).not.toContain("} steps`");
  });

  it("never brings a pre-lexicon label back", () => {
    const sources = [
      controlsAside,
      advancedDrawer,
      hostRoutingPicker,
      composerCard,
      loraPicker,
    ];
    for (const source of sources) {
      for (const retired of RETIRED_CONTROL_LABELS) {
        expect(source, retired).not.toContain(`"${retired}"`);
      }
      expect(templateText(source)).not.toMatch(/\bLoRA stack\b/);
    }
  });

  it("says seed only in mono, never as a label", () => {
    expect(advancedDrawer).not.toContain("<label>Seed</label>");
    expect(advancedDrawer).not.toContain('placeholder="Seed"');
    expect(controlsAside).not.toContain('placeholder="Seed"');
  });
});

describe("lexicon — a print's facts", () => {
  it("names the Lightbox rows the way the inspector names the controls", () => {
    for (const row of [
      "Style",
      "Size",
      CONTROL_WORDS.steps,
      CONTROL_WORDS.guidance,
      CONTROL_WORDS.seed,
      "Add-on look",
      "Made on",
    ]) {
      expect(lightbox, row).toContain(`<span class="lb__rowk">${row}</span`);
    }
    for (const retired of [
      "Model",
      "Dimensions",
      "Steps",
      "Guidance",
      "Seed",
      "LoRA",
      "Host",
    ]) {
      expect(lightbox, retired).not.toContain(
        `<span class="lb__rowk">${retired}</span`,
      );
    }
    expect(lightbox).not.toContain("CFG {{");
  });
});

describe("lexicon — the shell", () => {
  it("searches your images, spelled in the wordmark's own case", () => {
    expect(appNav).toContain('placeholder="Search your images…"');
    expect(appNav).not.toContain("Search prompts");
    // The wordmark is lowercase mono; the product name in a sentence stays Mold.
    expect(appNav).not.toMatch(/brand__word[^>]*>\s*Mold\b/);
    expect(appNav).toMatch(/brand__word[^>]*>\s*mold\b/);
  });
});
