import { describe, expect, it } from "vitest";
import {
  chainMetadataToClips,
  clampClipsToMotionTail,
  isPrintOfChainJob,
  planSequenceReuse,
  sequenceEditAvailability,
  sequenceGoneMessage,
  sequenceHostUnreachableMessage,
  sequenceReuseClampNote,
  sequenceReuseNote,
} from "./sequenceReuse";
import { DEFAULT_FADE_FRAMES, newSequenceClip } from "./sequenceForm";
import type { ChainOutputMetadata } from "./api/chainTypes";

function chain(
  stages: ChainOutputMetadata["stages"],
  motionTailFrames = 8,
): ChainOutputMetadata {
  return {
    stage_count: stages.length,
    motion_tail_frames: motionTailFrames,
    stages,
  };
}

describe("chainMetadataToClips", () => {
  it("maps every stage's prompt, frames, transition and fade length", () => {
    const clips = chainMetadataToClips(
      chain([
        { prompt: "a harbour at dawn", frames: 97, transition: "smooth" },
        { prompt: "the boats leave", frames: 65, transition: "cut" },
        {
          prompt: "gulls over the water",
          frames: 33,
          transition: "fade",
          fade_frames: 12,
        },
      ]),
    );

    expect(clips.map((c) => c.prompt)).toEqual([
      "a harbour at dawn",
      "the boats leave",
      "gulls over the water",
    ]);
    expect(clips.map((c) => c.frames)).toEqual([97, 65, 33]);
    expect(clips.map((c) => c.transition)).toEqual(["smooth", "cut", "fade"]);
    expect(clips[2]?.fadeFrames).toBe(12);
    // Fresh client ids — the rail keys on them, and reusing recorded ones
    // would collide with an open edit session.
    const ids = clips.map((c) => c.id);
    expect(new Set(ids).size).toBe(3);
    expect(ids.every((id) => id.length > 0)).toBe(true);
  });

  it("coerces clip 1's transition to smooth whatever was recorded", () => {
    const clips = chainMetadataToClips(
      chain([
        { prompt: "one", frames: 25, transition: "fade", fade_frames: 4 },
        { prompt: "two", frames: 25, transition: "cut" },
      ]),
    );
    expect(clips[0]?.transition).toBe("smooth");
  });

  it("falls back to the default fade length when none was recorded", () => {
    const clips = chainMetadataToClips(
      chain([
        { prompt: "one", frames: 25, transition: "smooth" },
        { prompt: "two", frames: 25, transition: "fade" },
      ]),
    );
    expect(clips[1]?.fadeFrames).toBe(DEFAULT_FADE_FRAMES);
  });

  it("puts the recorded negative on clip 1 only", () => {
    // `synthetic_generate_request` records the FIRST stage's negative prompt.
    // Spraying it across every clip would invent authorship.
    const clips = chainMetadataToClips(
      chain([
        { prompt: "one", frames: 25, transition: "smooth" },
        { prompt: "two", frames: 25, transition: "smooth" },
        { prompt: "three", frames: 25, transition: "smooth" },
      ]),
      { negativePromptForFirstClip: "blurry, watermark" },
    );
    expect(clips.map((c) => c.negativePrompt)).toEqual([
      "blurry, watermark",
      "",
      "",
    ]);
  });

  it("never carries a per-stage seed into the form", () => {
    const clips = chainMetadataToClips(
      chain([
        { prompt: "one", frames: 25, transition: "smooth", seed: "42" },
        {
          prompt: "two",
          frames: 25,
          transition: "smooth",
          seed: "18446744073709551615",
        },
      ]),
    );
    for (const clip of clips) {
      expect(Object.keys(clip)).not.toContain("seed");
      expect(JSON.stringify(clip)).not.toContain("18446744073709551615");
    }
  });

  it("leaves every clip's source image empty", () => {
    const clips = chainMetadataToClips(
      chain([
        { prompt: "one", frames: 25, transition: "smooth" },
        { prompt: "two", frames: 25, transition: "smooth" },
      ]),
    );
    expect(clips.every((c) => c.sourceImage === null)).toBe(true);
  });
});

describe("planSequenceReuse", () => {
  it("returns null for a legacy row with no chain block", () => {
    expect(planSequenceReuse({})).toBeNull();
    expect(planSequenceReuse({ chain: null })).toBeNull();
    // A job id without stages is half a record — treat it as legacy, not as
    // provenance worth trusting.
    expect(planSequenceReuse({ chain_job_id: "job-1" })).toBeNull();
  });

  it("returns null for a chain block with no stages", () => {
    expect(planSequenceReuse({ chain: chain([]) })).toBeNull();
  });

  it("reports the recorded motion tail without applying it", () => {
    const plan = planSequenceReuse({
      chain: chain(
        [
          { prompt: "one", frames: 25, transition: "smooth" },
          { prompt: "two", frames: 25, transition: "smooth" },
        ],
        16,
      ),
    });
    expect(plan?.recordedMotionTailFrames).toBe(16);
    expect(plan?.clips).toHaveLength(2);
  });

  it("reports nothing lossy for a row that had neither negatives nor sources", () => {
    const plan = planSequenceReuse({
      chain: chain([
        { prompt: "one", frames: 25, transition: "smooth" },
        { prompt: "two", frames: 25, transition: "smooth" },
      ]),
    });
    expect(plan?.lossy).toEqual({
      negatives: false,
      clipSources: false,
      perClipSeeds: false,
    });
  });

  it("flags negatives, clip sources and per-clip seeds when the row had them", () => {
    const plan = planSequenceReuse({
      negative_prompt: "blurry",
      source_image_sha256: "abc123",
      chain: chain([
        { prompt: "one", frames: 25, transition: "smooth", seed: "7" },
        { prompt: "two", frames: 25, transition: "smooth", seed: "8" },
      ]),
    });
    expect(plan?.lossy).toEqual({
      negatives: true,
      clipSources: true,
      perClipSeeds: true,
    });
    // The one recorded negative still lands on clip 1 — it is that clip's.
    expect(plan?.clips[0]?.negativePrompt).toBe("blurry");
    expect(plan?.clips[1]?.negativePrompt).toBe("");
  });

  it("does not call a single-clip row's negative lossy", () => {
    const plan = planSequenceReuse({
      negative_prompt: "blurry",
      chain: chain([{ prompt: "only", frames: 25, transition: "smooth" }]),
    });
    expect(plan?.lossy.negatives).toBe(false);
  });
});

describe("sequenceReuseNote", () => {
  it("names only the parts that were actually lossy", () => {
    expect(
      sequenceReuseNote(5, {
        negatives: true,
        clipSources: true,
        perClipSeeds: true,
      }),
    ).toBe(
      "reused 5 clips · negatives and clip sources aren't recorded in prints",
    );
    expect(
      sequenceReuseNote(3, {
        negatives: false,
        clipSources: true,
        perClipSeeds: false,
      }),
    ).toBe("reused 3 clips · clip sources aren't recorded in prints");
    expect(
      sequenceReuseNote(2, {
        negatives: false,
        clipSources: false,
        perClipSeeds: true,
      }),
    ).toBe("reused 2 clips");
    expect(
      sequenceReuseNote(1, {
        negatives: false,
        clipSources: false,
        perClipSeeds: false,
      }),
    ).toBe("reused 1 clip");
  });
});

describe("clampClipsToMotionTail", () => {
  it("raises only the clips at or below the tail", () => {
    const clips = [9, 25, 41].map((frames) => {
      const clip = newSequenceClip(frames);
      clip.prompt = `f${frames}`;
      return clip;
    });
    const result = clampClipsToMotionTail(clips, 25, 33);
    expect(result.clips.map((c) => c.frames)).toEqual([33, 33, 41]);
    expect(result.raised).toBe(2);
    // Untouched clips keep their identity so the rail doesn't remount them.
    expect(result.clips[2]).toBe(clips[2]);
    expect(result.clips.map((c) => c.prompt)).toEqual(["f9", "f25", "f41"]);
  });

  it("reports nothing raised when every clip already clears the tail", () => {
    const clips = [33, 49].map((frames) => newSequenceClip(frames));
    const result = clampClipsToMotionTail(clips, 25, 33);
    expect(result.raised).toBe(0);
    expect(result.clips).toEqual(clips);
  });

  it("never lands on a floor that is itself at or below the tail", () => {
    // A caller passing a stale minimum must still not produce an invalid
    // draft: the floor walks up the 8n+1 grid until it clears the tail.
    const result = clampClipsToMotionTail([newSequenceClip(9)], 25, 17);
    expect(result.clips[0]!.frames).toBeGreaterThan(25);
    expect((result.clips[0]!.frames - 1) % 8).toBe(0);
    expect(result.raised).toBe(1);
  });

  it("is a no-op for a zero-tail model", () => {
    const clips = [9, 25].map((frames) => newSequenceClip(frames));
    const result = clampClipsToMotionTail(clips, 0, 9);
    expect(result.raised).toBe(0);
    expect(result.clips.map((c) => c.frames)).toEqual([9, 25]);
  });
});

describe("sequenceEditAvailability", () => {
  it("is absent without a durable job id", () => {
    expect(
      sequenceEditAvailability({
        chainJobId: null,
        hostId: "plato",
        knownJobIds: ["job-1"],
      }),
    ).toBe("absent");
    expect(
      sequenceEditAvailability({
        chainJobId: "",
        hostId: "plato",
        knownJobIds: null,
      }),
    ).toBe("absent");
  });

  it("is unknown-host when the print's origin host does not resolve", () => {
    expect(
      sequenceEditAvailability({
        chainJobId: "job-1",
        hostId: null,
        knownJobIds: null,
      }),
    ).toBe("unknown-host");
  });

  it("is available when the host's listing has not been loaded", () => {
    // Absence of evidence is not evidence of absence — the click-time check
    // is cheap, and hiding here would strand a job that still exists.
    expect(
      sequenceEditAvailability({
        chainJobId: "job-1",
        hostId: "plato",
        knownJobIds: null,
      }),
    ).toBe("available");
  });

  it("is absent when the loaded listing lacks the id", () => {
    expect(
      sequenceEditAvailability({
        chainJobId: "job-1",
        hostId: "plato",
        knownJobIds: ["job-2", "job-3"],
      }),
    ).toBe("absent");
  });

  it("is available when the loaded listing carries the id", () => {
    expect(
      sequenceEditAvailability({
        chainJobId: "job-1",
        hostId: "plato",
        knownJobIds: ["job-2", "job-1"],
      }),
    ).toBe("available");
  });
});

describe("copy", () => {
  it("names the host in the gone-job fallback", () => {
    expect(sequenceGoneMessage("plato")).toBe(
      "That sequence job is gone from plato. Reused its settings instead.",
    );
  });

  it("points an unreachable host at Machines without promising a reuse", () => {
    expect(sequenceHostUnreachableMessage("plato:7680")).toBe(
      "Can't reach plato:7680. Check the host in Machines.",
    );
  });

  it("says once when durations were raised", () => {
    expect(sequenceReuseClampNote("ltx-2.3-22b-distilled:fp8")).toBe(
      "Clip durations raised to fit ltx-2.3-22b-distilled:fp8 — the model's motion tail changed.",
    );
  });
});

describe("isPrintOfChainJob", () => {
  it("matches the print a durable sequence produced", () => {
    expect(isPrintOfChainJob({ chain_job_id: "job-1" }, "job-1")).toBe(true);
  });

  it("rejects a row that merely looks similar", () => {
    // A legacy or ephemeral chain output carries no job id; matching it would
    // hand the canvas somebody else's video.
    expect(isPrintOfChainJob({}, "job-1")).toBe(false);
    expect(isPrintOfChainJob({ chain_job_id: null }, "job-1")).toBe(false);
    expect(isPrintOfChainJob({ chain_job_id: "job-2" }, "job-1")).toBe(false);
    expect(isPrintOfChainJob({ chain_job_id: "job-1" }, "")).toBe(false);
  });
});
