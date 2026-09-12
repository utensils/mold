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

  /* Two machines can publish the same filename; they are two prints. */
  it("keys on the machine as well as the file name", () => {
    const landed = useLandedPrintsStore();
    landed.noteLanded("local", "a.png");
    landed.noteLanded("plato", "a.png");
    expect(landed.count).toBe(2);
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
