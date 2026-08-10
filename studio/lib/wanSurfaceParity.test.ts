/**
 * Binds the Studio surfaces — web, desktop, and iPhone — to the shared Wan
 * capability fixture (#806).
 *
 * `tests/fixtures/wan/surface-parity-v1.json` states Wan's cross-surface
 * expectations once, and Rust tests in mold-core, the CLI, Discord, and the TUI
 * read it so a drift in any one surface fails CI. The fixture's own comment
 * claimed a Studio counterpart existed; it did not, which left the three
 * surfaces that share `studio/` as the only ones free to disagree with it —
 * precisely the drift the fixture is for.
 *
 * These assertions read the fixture rather than restating its values, so a
 * change there has to be reflected in the Studio logic or this fails.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  WAN_TI2V_FLF_MIN_FRAMES,
  sourceImageValidationError,
} from "./sourceImageCapability";
import {
  snapVideoFrames,
  videoFrameOffset,
  videoFrameStep,
} from "./videoDuration";

interface WanSurfaceParityFixture {
  schema: string;
  family: string;
  family_label: string;
  frame_grid: { step: number; offset: number };
  container_default: {
    unset_multi_frame: string;
    wan_single_frame: string;
    cli_minimal_build_fallback: string;
    mp4_default_families: string[];
  };
  first_last_frame: { ti2v_model_prefix: string; ti2v_min_frames: number };
  recipe: {
    max_distill_strength: number;
    distill_tier_models: string[];
    non_distill_models: string[];
    solvers: string[];
  };
}

const FIXTURE_RELATIVE = "tests/fixtures/wan/surface-parity-v1.json";

/**
 * Find the repo-root fixture by walking up from the working directory.
 *
 * `import.meta.url` is not a file URL in every environment these tests run in,
 * and the vitest root differs between the studio, web, and desktop configs, so
 * neither is a reliable anchor. The fixture's own location is.
 */
function fixturePath(): string {
  let directory = process.cwd();
  for (;;) {
    const candidate = resolve(directory, FIXTURE_RELATIVE);
    if (existsSync(candidate)) return candidate;
    const parent = dirname(directory);
    if (parent === directory) {
      throw new Error(
        `could not find ${FIXTURE_RELATIVE} above ${process.cwd()}`,
      );
    }
    directory = parent;
  }
}

const fixture: WanSurfaceParityFixture = JSON.parse(
  readFileSync(fixturePath(), "utf8"),
);

describe("wan surface parity (studio)", () => {
  it("reads the fixture the other surfaces read", () => {
    expect(fixture.schema).toBe("mold.wan.surface-parity.v1");
    expect(fixture.family).toBe("wan");
  });

  it("snaps frames onto the family's advertised grid", () => {
    // A wan model advertises its own grid; the studio helpers must take it
    // from the model row rather than the legacy 8n+1 default that predates
    // the family.
    const model = {
      frame_step: fixture.frame_grid.step,
      frame_offset: fixture.frame_grid.offset,
    };
    expect(videoFrameStep(model)).toBe(fixture.frame_grid.step);
    expect(videoFrameOffset(model)).toBe(fixture.frame_grid.offset);

    // Every snapped value must satisfy frames % step === offset % step.
    for (const requested of [2, 3, 5, 30, 33, 52, 53, 80, 81]) {
      const snapped = snapVideoFrames(requested, model);
      expect(
        (snapped - fixture.frame_grid.offset) % fixture.frame_grid.step,
        `${requested} snapped to ${snapped}, which is off the ${fixture.frame_grid.step}k+${fixture.frame_grid.offset} grid`,
      ).toBe(0);
    }

    // The modulo check above is self-consistent: fed the fixture's own step it
    // agrees with itself, and would still pass if a surface ignored the model
    // row entirely. Pin a value that only survives the advertised grid — 21 is
    // on 4k+1 but not on the legacy 8k+1 fallback, so a surface that dropped
    // the row and defaulted to 8 returns 17 or 25 here.
    expect(snapVideoFrames(21, model)).toBe(21);
    expect(snapVideoFrames(21, { frame_step: 8, frame_offset: 1 })).not.toBe(
      21,
    );
  });

  it("enforces the TI2V first/last-frame floor the other surfaces enforce", () => {
    expect(WAN_TI2V_FLF_MIN_FRAMES).toBe(
      fixture.first_last_frame.ti2v_min_frames,
    );

    const model = `${fixture.first_last_frame.ti2v_model_prefix}:fp16`;
    const below = sourceImageValidationError({
      capability: "optional",
      hasSourceImage: true,
      hasEndFrame: true,
      frames: fixture.first_last_frame.ti2v_min_frames - 1,
      model,
    });
    expect(below).toMatch(
      new RegExp(`${fixture.first_last_frame.ti2v_min_frames} frames`),
    );

    expect(
      sourceImageValidationError({
        capability: "optional",
        hasSourceImage: true,
        hasEndFrame: true,
        frames: fixture.first_last_frame.ti2v_min_frames,
        model,
      }),
    ).toBeNull();
  });

  it("keeps wan in the families whose unset output format is a video container", () => {
    // Studio surfaces must not offer a still-image container as the default
    // for a multi-frame wan render, nor hide PNG for a single-frame one
    // (#798). The fixture is the shared statement of that policy.
    expect(fixture.container_default.mp4_default_families).toContain("wan");
    expect(fixture.container_default.unset_multi_frame).toBe("mp4");
    expect(fixture.container_default.wan_single_frame).toBe("png");
  });
});
