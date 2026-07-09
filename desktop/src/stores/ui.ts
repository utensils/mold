import { defineStore } from "pinia";

/**
 * Shell-level UI signals the keyboard map and command palette raise for views
 * to react to. The tick counters are fire-and-forget: a view watches the tick
 * and acts on the change, so repeated presses always re-fire.
 */
export const useUiStore = defineStore("ui", {
  state: () => ({
    paletteOpen: false,
    newGenerationTick: 0,
    generateTick: 0,
    expandTick: 0,
    randomizeSeedTick: 0,
    copySeedTick: 0,
    galleryZoomTick: 0,
    galleryZoomDir: "reset" as "reset" | "in" | "out",
  }),
  actions: {
    togglePalette() {
      this.paletteOpen = !this.paletteOpen;
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
    expandPrompt() {
      this.expandTick++;
    },
    randomizeSeed() {
      this.randomizeSeedTick++;
    },
    copySeed() {
      this.copySeedTick++;
    },
    zoomGallery(dir: "reset" | "in" | "out") {
      this.galleryZoomDir = dir;
      this.galleryZoomTick++;
    },
  },
});
