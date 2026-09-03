/**
 * Wan's chain routing (#783).
 *
 * Two things separate wan from the LTX families and both are load-bearing:
 *
 * 1. Wan has **no latent motion tail**. Its smooth handoff is last-frame image
 *    conditioning, which only an image-conditioned checkpoint accepts — so the
 *    carryover is a property of the checkpoint, read from the advertised
 *    `source_image` contract, never of the family. Offering "Continue motion"
 *    on a T2V-only checkpoint would promise continuity the model cannot
 *    produce.
 * 2. Wan's grid is `4k+1`, not `8k+1`, and its auto-chain clip length is a
 *    VRAM envelope rather than a ceiling.
 *
 * This mirrors `mold_inference::chain::wan_carryover`; the Rust side has the
 * matching test.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  WAN_DEFAULT_CLIP_FRAMES,
  WAN_HANDOFF_DUPLICATED_FRAMES,
  WAN_SINGLE_EXPERT_CLIP_FRAMES,
  decideChainRouting,
  wanCarriesContext,
} from "./chainRouting";

/** The shared cross-surface fixture the CLI, the server, and mold-core read
 * (#806). Located by walking up from the working directory: the vitest root
 * differs between the studio, web, and desktop configs, so the fixture's own
 * location is the only reliable anchor. */
const FIXTURE_RELATIVE = "tests/fixtures/wan/surface-parity-v1.json";

interface TextOnlyRefusalTier {
  model: string;
  source_image: string;
  tier_default_frames: number;
  clip_frames: number;
}

interface TextOnlyRefusalFixture {
  template: string;
  total_frames: number;
  refused: TextOnlyRefusalTier[];
  chained: TextOnlyRefusalTier[];
}

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

const textOnlyRefusal: TextOnlyRefusalFixture = JSON.parse(
  readFileSync(fixturePath(), "utf8"),
).auto_chain.text_only_refusal;

function renderRefusal(model: string, clipFrames: number): string {
  return textOnlyRefusal.template
    .replaceAll("{model}", model)
    .replaceAll("{total_frames}", String(textOnlyRefusal.total_frames))
    .replaceAll("{clip_frames}", String(clipFrames));
}

describe("wan chain routing", () => {
  it("carries context only for an image-conditioned checkpoint", () => {
    expect(wanCarriesContext("required")).toBe(true);
    expect(wanCarriesContext("optional")).toBe(true);
    expect(wanCarriesContext("unsupported")).toBe(false);
    // Unclassified is unknown, never an assumed handoff — which is also what
    // an older server that does not advertise the field yields.
    expect(wanCarriesContext(null)).toBe(false);
    expect(wanCarriesContext(undefined)).toBe(false);
  });

  it("auto-chains past the clip envelope instead of rejecting", () => {
    // The pre-#783 behaviour: any wan request over the cap was rejected
    // outright because the family was not chain-capable.
    const decision = decideChainRouting(
      300,
      "wan",
      "wan22-ti2v-5b:fp16",
      undefined,
      24,
      "optional",
    );
    expect(decision.kind).toBe("chain");
    if (decision.kind !== "chain") return;
    expect(decision.clipFrames).toBe(WAN_SINGLE_EXPERT_CLIP_FRAMES);
    expect(decision.stageCount).toBeGreaterThan(1);
  });

  it("uses the two-expert envelope for A14B and the wider one for 5B", () => {
    // Unclassified rather than "unsupported": a declared text-to-video tier
    // is refused outright now (see the text-only case below), and the point
    // here is the envelope, not the contract.
    const a14b = decideChainRouting(
      300,
      "wan",
      "wan22-t2v-a14b:q5",
      undefined,
      16,
      null,
    );
    expect(a14b.kind === "chain" && a14b.clipFrames).toBe(
      WAN_DEFAULT_CLIP_FRAMES,
    );

    const fiveB = decideChainRouting(
      300,
      "wan",
      "wan22-ti2v-5b:q8",
      undefined,
      24,
      "optional",
    );
    expect(fiveB.kind === "chain" && fiveB.clipFrames).toBe(
      WAN_SINGLE_EXPERT_CLIP_FRAMES,
    );
  });

  it("treats a tier's advertised default as one generation", () => {
    const model = "wan22-i2v-a14b:q5";
    expect(
      decideChainRouting(81, "wan", model, undefined, 16, "required", 81),
    ).toEqual({ kind: "single" });

    expect(
      decideChainRouting(85, "wan", model, undefined, 16, "required", 81),
    ).toMatchObject({
      kind: "chain",
      clipFrames: 81,
      motionTail: WAN_HANDOFF_DUPLICATED_FRAMES,
      stageCount: 2,
    });
  });

  it("keeps every clip length on wan's 4k+1 grid", () => {
    for (const model of ["wan22-t2v-a14b:q5", "wan22-ti2v-5b:fp16"]) {
      const decision = decideChainRouting(
        300,
        "wan",
        model,
        undefined,
        16,
        "optional",
      );
      expect(decision.kind).toBe("chain");
      if (decision.kind !== "chain") continue;
      expect(
        (decision.clipFrames - 1) % 4,
        `${model} clip ${decision.clipFrames} is off the 4k+1 grid`,
      ).toBe(0);
    }
  });

  it("sizes the seam from the checkpoint, not from the caller's LTX tail", () => {
    // The seam is observable directly, so assert it rather than a stage-count
    // side effect: at 53-frame clips a tail of 0 and a tail of 1 need the same
    // number of stages, so a count comparison would pass whatever the tail was.
    const routed = (sourceImage: string | null) => {
      const decision = decideChainRouting(
        300,
        "wan",
        "wan22-t2v-a14b:q5",
        // A caller-supplied LTX-shaped tail must not reach wan in either
        // direction: wan has no latent motion tail to size.
        24,
        16,
        sourceImage,
      );
      if (decision.kind !== "chain") throw new Error("expected a chain");
      return decision.motionTail;
    };

    expect(routed("optional")).toBe(WAN_HANDOFF_DUPLICATED_FRAMES);
    expect(routed("required")).toBe(WAN_HANDOFF_DUPLICATED_FRAMES);
    // An unclassified checkpoint may or may not be image-conditioned, so
    // nothing is assumed to cross the seam and nothing is trimmed. (A tier
    // that DECLARES it carries nothing is refused rather than seamed — see
    // the text-only case below.)
    expect(routed(null)).toBe(0);

    // The seeded frame is the only duplicate; 17 would discard sixteen good
    // frames at every boundary.
    expect(WAN_HANDOFF_DUPLICATED_FRAMES).toBe(1);
  });

  it("refuses a one-shot auto-chain on a text-only tier, in the fixture's words", () => {
    // The bug this closes: a 259-frame one-shot on `wan21-t2v-1.3b:turbo`
    // submitted from the web Studio became a three-stage ephemeral chain
    // (121/121/17, every stage the same seed, motion tail 0) whose video reset
    // at both boundaries. The CLI had refused the same request since #1508;
    // the Studio and the HTTP chain-jobs door had not.
    for (const tier of textOnlyRefusal.refused) {
      const decision = decideChainRouting(
        textOnlyRefusal.total_frames,
        "wan",
        tier.model,
        undefined,
        24,
        tier.source_image,
        tier.tier_default_frames,
      );
      expect(decision.kind, `${tier.model} must refuse`).toBe("reject");
      if (decision.kind !== "reject") continue;
      expect(decision.reason).toBe(renderRefusal(tier.model, tier.clip_frames));

      // At or below its own clip size there is nothing to chain, so the same
      // tier still renders one continuous clip.
      expect(
        decideChainRouting(
          tier.clip_frames,
          "wan",
          tier.model,
          undefined,
          24,
          tier.source_image,
          tier.tier_default_frames,
        ),
      ).toEqual({ kind: "single" });
    }

    // An image-conditioned tier seeds every continuation from the previous
    // clip's final frame, so it still chains.
    for (const tier of textOnlyRefusal.chained) {
      expect(
        decideChainRouting(
          textOnlyRefusal.total_frames,
          "wan",
          tier.model,
          undefined,
          24,
          tier.source_image,
          tier.tier_default_frames,
        ),
      ).toMatchObject({ kind: "chain", clipFrames: tier.clip_frames });
    }

    // Unclassified is "unknown", never a declared refusal: an opaque catalog
    // id must keep routing.
    expect(
      decideChainRouting(
        textOnlyRefusal.total_frames,
        "wan",
        "cv:12345",
        undefined,
        24,
        null,
      ).kind,
    ).toBe("chain");
  });

  it("leaves the LTX families exactly as they were", () => {
    // The added parameter must not reach a family that does not read it: a
    // wan-shaped `source_image` hint must not rewrite LTX-2's real handoff.
    const withHint = decideChainRouting(
      500,
      "ltx2",
      "ltx-2-19b-distilled:fp8",
      24,
      24,
      "unsupported",
    );
    const withoutHint = decideChainRouting(
      500,
      "ltx2",
      "ltx-2-19b-distilled:fp8",
      24,
      24,
    );
    expect(withHint).toEqual(withoutHint);
  });

  // The host is authoritative about its own single-request ceiling on a family
  // this router cannot chain: without the row's own `max_frames` the cap fell
  // back to the family table and refused a count the server accepts, and with
  // it a row that advertises LESS than the table is held to its own number.
  it("prefers a row's advertised ceiling over the family table when it cannot chain", () => {
    expect(
      decideChainRouting(
        200,
        "brandnew",
        "cv:1",
        undefined,
        24,
        null,
        100,
        257,
      ),
    ).toEqual({ kind: "single" });

    const refused = decideChainRouting(
      200,
      "minimax-h3",
      "hf:x",
      undefined,
      25,
      "optional",
      124,
      124,
    );
    expect(refused.kind).toBe("reject");
    expect(refused.kind === "reject" && refused.reason).toContain("124");
  });
});
