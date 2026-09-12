import { defineStore } from "pinia";
import { appIsBackground } from "../lib/notify";

/**
 * Prints that landed while this app was in the background — on ANY connected
 * machine, made by ANY client. This is what the Dock badge counts: "things
 * arrived while you were away", not "this app is busy". A print that lands
 * while the window has focus is already announced by the canvas, the toast
 * and the sidebar's new-print pill, so it never badges.
 *
 * Best-effort by design. The count is whatever the live `/api/events` streams
 * and this app's own completions reported; a machine whose server predates
 * `/api/events` contributes nothing, and the badge is never a ledger.
 */
export const useLandedPrintsStore = defineStore("landedPrints", {
  state: () => ({
    /** `${hostId}:${filename}` per unseen print. One machine can publish the
     *  same file name as another, so the machine is part of the key. */
    unseen: [] as string[],
  }),
  getters: {
    count: (state): number => state.unseen.length,
  },
  actions: {
    noteLanded(hostId: string, filename: string | null | undefined): void {
      if (!filename || !appIsBackground()) return;
      const key = `${hostId}:${filename}`;
      if (this.unseen.includes(key)) return;
      // Replaced, never mutated in place: this runs from SSE callbacks.
      this.unseen = [...this.unseen, key];
    },
    /** The window came back — the badge has done its job. */
    markSeen(): void {
      if (this.unseen.length === 0) return;
      this.unseen = [];
    },
  },
});
