import { describe, expect, it } from "vitest";

import {
  LTX2_MAX_FRAMES_ABSOLUTE,
  LTX2_MAX_RUNTIME_SECONDS,
  MAX_FRAMES_GLOBAL,
  ltx2MaxFramesAtFps,
  ltx2MaxFramesOnGridAtFps,
  maxFramesForFamilyAtFps,
} from "./videoBudget";

describe("ltx2MaxFramesAtFps", () => {
  it("derives the ceiling from fps because the budget is a duration", () => {
    // Mirrors crates/mold-core/src/validation.rs: F <= seconds * fps + 4.
    expect(ltx2MaxFramesAtFps(24)).toBe(484);
    expect(ltx2MaxFramesAtFps(25)).toBe(504);
    expect(ltx2MaxFramesAtFps(12)).toBe(244);
    expect(ltx2MaxFramesAtFps(8)).toBe(164);
  });

  it("is tighter than the old flat 153 at low frame rates", () => {
    // 6 fps only buys 20s of runtime; the old constant over-admitted here.
    expect(ltx2MaxFramesAtFps(6)).toBe(124);
    expect(ltx2MaxFramesAtFps(6)).toBeLessThan(153);
  });

  it("clamps to the absolute resource guard above 30 fps", () => {
    expect(ltx2MaxFramesAtFps(60)).toBe(LTX2_MAX_FRAMES_ABSOLUTE);
    expect(ltx2MaxFramesAtFps(120)).toBe(LTX2_MAX_FRAMES_ABSOLUTE);
  });

  it("falls back to the default fps rather than collapsing to one frame", () => {
    expect(ltx2MaxFramesAtFps(null)).toBe(484);
    expect(ltx2MaxFramesAtFps(undefined)).toBe(484);
    expect(ltx2MaxFramesAtFps(0)).toBe(LTX2_MAX_RUNTIME_SECONDS + 4);
  });

  it("honours a server-advertised budget over the built-in fallback", () => {
    expect(ltx2MaxFramesAtFps(24, 10)).toBe(244);
    expect(ltx2MaxFramesAtFps(24, 20, 200)).toBe(200);
  });
});

describe("maxFramesForFamilyAtFps", () => {
  it("only ltx2 has an fps-dependent ceiling", () => {
    // The advertised value is grid-snapped so a client that clamps to it can
    // actually submit: 244 is the raw 20s budget at 12 fps, but 243 % 8 == 3.
    expect(maxFramesForFamilyAtFps("ltx2", 12)).toBe(241);
    expect(maxFramesForFamilyAtFps("ltx-2", 12)).toBe(241);
    expect(maxFramesForFamilyAtFps("ltx-video", 12)).toBe(MAX_FRAMES_GLOBAL);
    expect(maxFramesForFamilyAtFps("ltx-video", 30)).toBe(MAX_FRAMES_GLOBAL);
  });

  it("gives wan a flat ceiling that does not move with fps", () => {
    // Mirrors `"wan" => Some(MAX_FRAMES_GLOBAL)` in
    // `max_frames_for_family_at_fps`. Wan's temporal RoPE indexes latent
    // frames against a 1024-entry table, so 257 pixel frames is nowhere near
    // binding and memory is the real limit — unlike LTX-2, whose ceiling is a
    // duration and therefore fps-dependent.
    for (const fps of [6, 16, 24, 30, 60]) {
      expect(maxFramesForFamilyAtFps("wan", fps), `${fps} fps`).toBe(
        MAX_FRAMES_GLOBAL,
      );
    }
    expect(maxFramesForFamilyAtFps("WAN", 16)).toBe(MAX_FRAMES_GLOBAL);
  });

  it("advertises a wan ceiling that sits on wan's own 4n+1 grid", () => {
    // A cap a client clamps to must itself be submittable; the LTX-2 arm
    // needed grid-snapping for exactly this reason.
    const cap = maxFramesForFamilyAtFps("wan", 16)!;
    expect((cap - 1) % 4).toBe(0);
  });

  it("returns null for families that publish no frame ceiling", () => {
    expect(maxFramesForFamilyAtFps("flux", 24)).toBeNull();
    expect(maxFramesForFamilyAtFps("", 24)).toBeNull();
    expect(maxFramesForFamilyAtFps(null, 24)).toBeNull();
  });
});

describe("ltx2MaxFramesOnGridAtFps", () => {
  it("snaps the duration budget onto the 8n+1 grid the server enforces", () => {
    for (const fps of [6, 12, 24, 30, 48, 60, 120]) {
      const cap = ltx2MaxFramesOnGridAtFps(fps);
      expect((cap - 1) % 8, `${fps} fps`).toBe(0);
      expect(cap).toBeLessThanOrEqual(ltx2MaxFramesAtFps(fps));
    }
    expect(ltx2MaxFramesOnGridAtFps(24)).toBe(481);
    expect(ltx2MaxFramesOnGridAtFps(12)).toBe(241);
    // Above 30 fps the absolute guard binds first, and 604 is off-grid too.
    expect(ltx2MaxFramesOnGridAtFps(48)).toBe(601);
  });
});
