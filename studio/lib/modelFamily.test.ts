/**
 * The Studio half of #806's "present the family as **Wan Video** everywhere".
 *
 * `tests/fixtures/wan/surface-parity-v1.json` states the label once; the CLI
 * (`mold-cli/src/ui.rs`) and TUI (`mold-tui/src/ui/models.rs`) already read it.
 * Web, desktop, and iPhone share this module, so this is the third reader —
 * and the reason the table moved out of `web/`, where desktop and iPhone could
 * not see it and rendered the raw `wan` slug instead.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  catalogFamily,
  familyLabel,
  matchesCatalogFamily,
} from "./modelFamily";

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

const fixture: { family: string; family_label: string } = JSON.parse(
  readFileSync(fixturePath(), "utf8"),
);

describe("model family labels", () => {
  it("names wan the way every other surface names it", () => {
    expect(familyLabel(fixture.family)).toBe(fixture.family_label);
    // Guard the guard: a fixture whose label equalled the slug would make the
    // assertion above pass against the raw-slug regression it exists to catch.
    expect(fixture.family_label).not.toBe(fixture.family);
  });

  it("keeps the established labels for the families that have one", () => {
    expect(familyLabel("flux")).toBe("FLUX");
    expect(familyLabel("ltx-2")).toBe("LTX-2");
    expect(familyLabel("qwen-image-edit")).toBe("Qwen Image Edit");
    expect(familyLabel("minimax-h3")).toBe("MiniMax H3");
  });

  it("title-cases an unknown family rather than showing the slug", () => {
    expect(familyLabel("some-new-family")).toBe("Some New Family");
    expect(familyLabel("another_one")).toBe("Another One");
    expect(familyLabel("")).toBe("");
  });

  it("groups Qwen Image Edit under Qwen Image for catalog browsing", () => {
    expect(catalogFamily("qwen-image-edit")).toBe("qwen-image");
    expect(matchesCatalogFamily("qwen-image-edit", "qwen-image")).toBe(true);
    expect(catalogFamily("flux")).toBe("flux");
  });

  it("does not make a family chip wider than it already is", () => {
    // Desktop renders the label in a fixed-width code chip whose width is set
    // by today's longest slug. Only wan grows (3 -> 9); every other label is
    // the same length or shorter than the slug it replaces. A future label
    // past that bound needs the chip resized deliberately, not truncated.
    const widest = "qwen-image-edit".length;
    for (const family of [
      "flux",
      "flux2",
      "sd15",
      "sdxl",
      "sd3.5",
      "z-image",
      "qwen-image",
      "qwen-image-edit",
      "wuerstchen",
      "ltx-video",
      "ltx-2",
      "wan",
      "minimax-h3",
    ]) {
      expect(familyLabel(family).length, family).toBeLessThanOrEqual(widest);
    }
  });
});
