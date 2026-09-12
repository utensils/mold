import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";

vi.mock("../lib/notify", () => ({ appIsBackground: vi.fn(() => true) }));

import { appIsBackground } from "../lib/notify";
import { useLandedPrintsStore } from "./landedPrints";

const background = (away: boolean) => vi.mocked(appIsBackground).mockReturnValue(away);

beforeEach(() => {
  setActivePinia(createPinia());
  vi.clearAllMocks();
  background(true);
});

describe("landed prints", () => {
  it("counts a print that landed while the app was in the background", () => {
    const landed = useLandedPrintsStore();
    landed.noteLanded("local", "a.png");
    landed.noteLanded("plato", "b.png");
    expect(landed.count).toBe(2);
  });

  /*
   * A print made while the person is watching already has the canvas, the
   * toast and the sidebar's "new" pill. Badging it would mark as unseen the
   * one thing they definitely saw.
   */
  it("ignores a print that landed while the window was focused", () => {
    background(false);
    const landed = useLandedPrintsStore();
    landed.noteLanded("local", "a.png");
    expect(landed.count).toBe(0);
  });

  it("counts one print once, however many times it is reported", () => {
    const landed = useLandedPrintsStore();
    landed.noteLanded("local", "a.png");
    landed.noteLanded("local", "a.png");
    expect(landed.count).toBe(1);
  });

  /*
   * A remote print that auto-saves to this Mac keeps the origin's file name,
   * and the import raises its own `gallery_added` here. That is ONE print with
   * two copies — exactly what the Library's All view collapses into one tile —
   * so the badge must not say two.
   */
  it("counts a print once however many machines hold a copy of it", () => {
    const landed = useLandedPrintsStore();
    landed.noteLanded("plato", "a.png");
    landed.noteLanded("local", "a.png");
    expect(landed.count).toBe(1);
  });

  /*
   * The mirror imports BOTH names when the gallery renamed the copy, so a
   * second `gallery_added` arrives here under a name nothing has seen. The one
   * place that knows those names are one print is the mirror loop, which says
   * so before it imports.
   */
  it("does not count a copy this app announced it was importing", () => {
    const landed = useLandedPrintsStore();
    landed.noteLanded("plato", "a.png");
    landed.expectCopy("a.png");
    landed.expectCopy("a-1.png");

    landed.noteLanded("local", "a.png");
    landed.noteLanded("local", "a-1.png");

    expect(landed.count).toBe(1);
  });

  it("counts an unrelated print that happens to follow an expected copy", () => {
    const landed = useLandedPrintsStore();
    landed.expectCopy("a-1.png");
    landed.noteLanded("local", "a-1.png");
    // The expectation is consumed, so the NEXT print under that name counts.
    landed.noteLanded("local", "a-1.png");
    expect(landed.count).toBe(1);
  });

  /* A print the person declined to keep never landed. */
  it("forgets a print that was trashed or removed", () => {
    const landed = useLandedPrintsStore();
    landed.noteLanded("plato", "a.png");
    landed.noteLanded("plato", "b.png");

    landed.forgetLanded("a.png");

    expect(landed.count).toBe(1);
    landed.forgetLanded("never-counted.png");
    expect(landed.count).toBe(1);
  });

  it("ignores a frame with no file name", () => {
    const landed = useLandedPrintsStore();
    landed.noteLanded("local", "");
    landed.noteLanded("local", null);
    expect(landed.count).toBe(0);
  });

  it("clears everything once the person has looked", () => {
    const landed = useLandedPrintsStore();
    landed.noteLanded("local", "a.png");
    landed.markSeen();
    expect(landed.count).toBe(0);
    // And the cleared key can be counted again next time away.
    landed.noteLanded("local", "a.png");
    expect(landed.count).toBe(1);
  });
});
