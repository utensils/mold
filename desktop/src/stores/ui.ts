import { defineStore } from "pinia";

export type CatalogLayout = "grid" | "table";

/** The intents New image owns. Every raiser routes to `/create`. */
export type CreateIntent =
  "newGeneration" | "generate" | "makeVariations" | "expand" | "randomizeSeed";

const INTENT_TICK = {
  newGeneration: "newGenerationTick",
  generate: "generateTick",
  makeVariations: "makeVariationsTick",
  expand: "expandTick",
  randomizeSeed: "randomizeSeedTick",
} as const satisfies Record<CreateIntent, string>;

/**
 * Shell-level UI signals the keyboard map and command palette raise for views
 * to react to. The tick counters are fire-and-forget: a view watches the tick
 * and acts on the change, so repeated presses always re-fire.
 *
 * The New image intents are CONSUMED rather than merely watched. The view that
 * acts on one is usually mounting in response to it — the palette raises the
 * intent and then navigates to New image — so a watcher that only saw changes
 * it was present for dropped every cross-workspace command on the floor.
 *
 * Also holds session-scoped view preferences (deliberately not persisted to
 * disk): the catalog layout survives navigating away and back, and resets to
 * the table default on the next app launch.
 */
export const useUiStore = defineStore("ui", {
  state: () => ({
    paletteOpen: false,
    newGenerationTick: 0,
    generateTick: 0,
    makeVariationsTick: 0,
    expandTick: 0,
    randomizeSeedTick: 0,
    copySeedTick: 0,
    consumedTicks: {
      newGeneration: 0,
      generate: 0,
      makeVariations: 0,
      expand: 0,
      randomizeSeed: 0,
    } as Record<CreateIntent, number>,
    catalogLayout: "table" as CatalogLayout,
  }),
  actions: {
    togglePalette() {
      this.paletteOpen = !this.paletteOpen;
    },
    setCatalogLayout(layout: CatalogLayout) {
      this.catalogLayout = layout;
    },
    closePalette() {
      this.paletteOpen = false;
    },
    /** True exactly once per raise of `intent`, whoever is mounted for it. */
    consumeIntent(intent: CreateIntent): boolean {
      const raised = this[INTENT_TICK[intent]];
      if (this.consumedTicks[intent] === raised) return false;
      this.consumedTicks[intent] = raised;
      return true;
    },
    newGeneration() {
      this.newGenerationTick++;
    },
    generate() {
      this.generateTick++;
    },
    makeVariations() {
      this.makeVariationsTick++;
    },
    expandPrompt() {
      this.expandTick++;
    },
    randomizeSeed() {
      this.randomizeSeedTick++;
    },
    copySeed() {
      this.copySeedTick++;
    },
  },
});
