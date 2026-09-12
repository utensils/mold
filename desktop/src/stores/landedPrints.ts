import { defineStore } from "pinia";
import { appIsBackground } from "../lib/notify";

/**
 * Prints that landed while this app was in the background — on ANY connected
 * machine, made by ANY client. This is what the Dock badge counts: "things
 * arrived while you were away", not "this app is busy". A print that lands
 * while the window has focus is already announced by the canvas, the toast
 * and the sidebar's new-print pill, so it never badges.
 *
 * Best-effort by design, and in memory only: the count is whatever the live
 * `/api/events` streams and this app's own completions reported since launch,
 * a machine whose server predates `/api/events` contributes nothing, and a
 * relaunch starts at zero. The badge is a nudge, never a ledger.
 */
export const useLandedPrintsStore = defineStore("landedPrints", {
  state: () => ({
    /**
     * filename → the machine that first reported the print. Keyed by FILENAME,
     * because a remote print auto-saved to this Mac keeps the origin's name
     * and raises its own `gallery_added` here: that is one print with two
     * copies, which is exactly what the Library's All view collapses into one
     * tile. A Map rather than a scan, because a machine rendering overnight
     * makes this thousands of entries and every frame asks whether it already
     * holds one.
     */
    unseen: new Map<string, string>(),
    /**
     * Names this app is about to import as its own copy of a print it has
     * already counted. The mirror loop imports the origin's name AND the name
     * the gallery gave the copy, so the second frame arrives under a name
     * nothing has seen; only the mirror knows they are one print. Each entry
     * is consumed by the frame it predicts, and `markSeen` clears the rest.
     */
    expectedCopies: new Set<string>(),
  }),
  getters: {
    count: (state): number => state.unseen.size,
  },
  actions: {
    noteLanded(hostId: string, filename: string | null | undefined): void {
      if (!filename || !appIsBackground()) return;
      if (this.expectedCopies.has(filename)) {
        // Replaced, never mutated in place: this runs from SSE callbacks.
        const remaining = new Set(this.expectedCopies);
        remaining.delete(filename);
        this.expectedCopies = remaining;
        return;
      }
      if (this.unseen.has(filename)) return;
      this.unseen = new Map(this.unseen).set(filename, hostId);
    },
    /** This app is importing its own copy of a print it already counted. */
    expectCopy(filename: string | null | undefined): void {
      if (!filename || this.expectedCopies.has(filename)) return;
      this.expectedCopies = new Set(this.expectedCopies).add(filename);
    },
    /** Trashed or deleted: a print the person declined to keep never landed. */
    forgetLanded(filename: string | null | undefined): void {
      if (!filename || !this.unseen.has(filename)) return;
      const remaining = new Map(this.unseen);
      remaining.delete(filename);
      this.unseen = remaining;
    },
    /** The window came back — the badge has done its job. */
    markSeen(): void {
      if (this.unseen.size > 0) this.unseen = new Map();
      if (this.expectedCopies.size > 0) this.expectedCopies = new Set();
    },
  },
});
