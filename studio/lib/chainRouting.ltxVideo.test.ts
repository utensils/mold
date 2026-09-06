import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { decideChainRouting, textOnlyAutoChainRefusal } from "./chainRouting";

const FIXTURE_RELATIVE = "tests/fixtures/ltx-video/surface-parity-v1.json";

function fixturePath(): string {
  let directory = process.cwd();
  for (;;) {
    const candidate = resolve(directory, FIXTURE_RELATIVE);
    if (existsSync(candidate)) return candidate;
    const parent = dirname(directory);
    if (parent === directory)
      throw new Error(`could not find ${FIXTURE_RELATIVE}`);
    directory = parent;
  }
}

const refusal = JSON.parse(readFileSync(fixturePath(), "utf8")).auto_chain
  .text_only_refusal as {
  template: string;
  model: string;
  total_frames: number;
  clip_frames: number;
};

function expectedRefusal(): string {
  return refusal.template
    .replaceAll("{model}", refusal.model)
    .replaceAll("{total_frames}", String(refusal.total_frames))
    .replaceAll("{clip_frames}", String(refusal.clip_frames));
}

describe("legacy LTX-Video chain routing", () => {
  it("renders the shared refusal byte-identically", () => {
    expect(
      textOnlyAutoChainRefusal(
        "ltx-video",
        refusal.model,
        "unsupported",
        refusal.total_frames,
        refusal.clip_frames,
      ),
    ).toBe(expectedRefusal());
  });

  it("keeps the request single instead of automatically splitting at 97", () => {
    expect(
      decideChainRouting(refusal.clip_frames, "ltx-video", refusal.model),
    ).toEqual({ kind: "single" });
    expect(
      decideChainRouting(refusal.total_frames, "ltx-video", refusal.model),
    ).toEqual({ kind: "single" });
    expect(decideChainRouting(265, "ltx-video", refusal.model)).toMatchObject({
      kind: "reject",
      reason: expect.stringContaining("257 or less"),
    });
  });
});
