import { defineStore } from "pinia";

export type CatalogLayout = "grid" | "table";

/**
 * Shell-level UI signals the keyboard map and command palette raise for views
 * to react to. The tick counters are fire-and-forget: a view watches the tick
 * and acts on the change, so repeated presses always re-fire.
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
