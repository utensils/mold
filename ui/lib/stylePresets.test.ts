import { describe, expect, it } from "vitest";
import {
  LEGACY_STYLE_IDS,
  STYLE_ANGLES,
  STYLE_PRESETS,
  angleForIndex,
  composeStyle,
  mergeStyleNegative,
  resolveStyleId,
  styleHint,
  stylePresetById,
  stylePresetLabel,
} from "./stylePresets";

/** Every id the two pre-kit tables ever persisted (desktop + web). */
const DESKTOP_LEGACY_IDS = [
  "photographic",
  "cinematic",
  "anime",
  "illustration",
  "3d-render",
  "watercolor",
  "neon",
  "minimal",
];
const WEB_LEGACY_IDS = [
  "photoreal",
  "cinematic",
  "portrait",
  "illustration",
  "anime",
  "3d",
  "watercolor",
  "product",
];

describe("STYLE_PRESETS", () => {
  it("ships the eight prototype presets in chip order", () => {
    expect(STYLE_PRESETS.map((p) => p.id)).toEqual([
      "photoreal",
      "cinematic",
      "portrait",
      "illustration",
      "anime",
      "3d",
      "watercolor",
      "product",
    ]);
  });

  it("gives every preset a label and a non-empty positive", () => {
    for (const preset of STYLE_PRESETS) {
      expect(preset.label.length, preset.id).toBeGreaterThan(0);
      expect(preset.positive.trim().length, preset.id).toBeGreaterThan(0);
      if (preset.negative !== undefined) {
        expect(preset.negative.trim().length, preset.id).toBeGreaterThan(0);
        // Negatives are plain fragment lists — never templated.
        expect(preset.negative, preset.id).not.toContain("{prompt}");
      }
    }
  });

  it("uses the {prompt} placeholder at most once per positive", () => {
    for (const preset of STYLE_PRESETS) {
      const count = preset.positive.split("{prompt}").length - 1;
      expect(count, preset.id).toBeLessThanOrEqual(1);
    }
  });

  it("labels a preset, falling back to None", () => {
    expect(stylePresetLabel("cinematic")).toBe("Cinematic");
    expect(stylePresetLabel("photographic")).toBe("Photoreal");
    expect(stylePresetLabel("")).toBe("None");
    expect(stylePresetLabel("nope")).toBe("None");
  });
});

describe("composeStyle templating", () => {
  it("substitutes {prompt} into a templated positive", () => {
    const { prompt } = composeStyle("a cat in a top hat", "cinematic", {
      supportsNegativePrompt: false,
    });
    expect(prompt).toBe(
      "cinematic film still of a cat in a top hat, cinematic lighting, anamorphic, dramatic mood, subtle film grain",
    );
  });

  it("appends with a comma when the positive has no placeholder", () => {
    const { prompt } = composeStyle("a red sneaker", "product", {
      supportsNegativePrompt: false,
    });
    expect(prompt).toBe(
      "a red sneaker, product photography, seamless backdrop, softbox lighting",
    );
  });

  it("trims the prompt before templating", () => {
    const { prompt } = composeStyle("  a fox  ", "portrait", {
      supportsNegativePrompt: false,
    });
    expect(prompt).toBe(
      "studio portrait of a fox, soft key light, 85mm, crisp detail",
    );
  });

  it("returns the prompt verbatim for an empty prompt, empty id, or unknown id", () => {
    expect(
      composeStyle("   ", "cinematic", { supportsNegativePrompt: true }).prompt,
    ).toBe("   ");
    expect(
      composeStyle("a cat", "", { supportsNegativePrompt: true }).prompt,
    ).toBe("a cat");
    expect(
      composeStyle("a cat", "does-not-exist", { supportsNegativePrompt: true })
        .prompt,
    ).toBe("a cat");
  });

  it("applies the canonical preset when given a legacy id", () => {
    const canonical = composeStyle("a fox", "photoreal", {
      supportsNegativePrompt: false,
    });
    const legacy = composeStyle("a fox", "photographic", {
      supportsNegativePrompt: false,
    });
    expect(legacy).toEqual(canonical);
  });
});

describe("composeStyle negative gating", () => {
  it("never adds preset fragments when the model has no negative prompt", () => {
    const out = composeStyle("a cat", "photoreal", {
      supportsNegativePrompt: false,
      negative: "text, watermark",
    });
    expect(out.negative).toBe("text, watermark");
  });

  it("uses the preset negative when the user gave none", () => {
    const preset = STYLE_PRESETS.find((p) => p.id === "photoreal");
    const out = composeStyle("a cat", "photoreal", {
      supportsNegativePrompt: true,
    });
    expect(out.negative).toBe(preset?.negative);
  });

  it("merges the preset negative after the user's fragments", () => {
    const out = composeStyle("a cat", "photoreal", {
      supportsNegativePrompt: true,
      negative: "text, watermark",
    });
    expect(out.negative).toBe(
      "text, watermark, drawing, painting, illustration, deformed, blurry",
    );
  });

  it("dedupes fragments the user already wrote", () => {
    const out = composeStyle("a cat", "photoreal", {
      supportsNegativePrompt: true,
      negative: "blurry,  drawing , text",
    });
    expect(out.negative).toBe(
      "blurry, drawing, text, painting, illustration, deformed",
    );
  });

  it("passes the user negative through untouched for a preset without one", () => {
    const preset = STYLE_PRESETS.find((p) => p.id === "watercolor");
    expect(preset?.negative).toBeUndefined();
    const out = composeStyle("a heron", "watercolor", {
      supportsNegativePrompt: true,
      negative: "text, watermark",
    });
    expect(out.negative).toBe("text, watermark");
  });

  it("leaves negative undefined when neither side has one", () => {
    const out = composeStyle("a heron", "watercolor", {
      supportsNegativePrompt: true,
    });
    expect(out.negative).toBeUndefined();
  });
});

describe("mergeStyleNegative", () => {
  it("merges the preset negative after the user's fragments", () => {
    expect(
      mergeStyleNegative("text, watermark", "cinematic", {
        supportsNegativePrompt: true,
      }),
    ).toBe("text, watermark, anime, cartoon, graphic, washed out");
  });

  it("uses the preset negative alone when the user gave none", () => {
    expect(
      mergeStyleNegative("", "cinematic", { supportsNegativePrompt: true }),
    ).toBe("anime, cartoon, graphic, washed out");
    expect(
      mergeStyleNegative(undefined, "cinematic", {
        supportsNegativePrompt: true,
      }),
    ).toBe("anime, cartoon, graphic, washed out");
  });

  it("dedupes fragments the user already wrote", () => {
    expect(
      mergeStyleNegative("cartoon,  text ", "cinematic", {
        supportsNegativePrompt: true,
      }),
    ).toBe("cartoon, text, anime, graphic, washed out");
  });

  it("returns the user negative untouched when the family has no negative", () => {
    expect(
      mergeStyleNegative("text", "cinematic", {
        supportsNegativePrompt: false,
      }),
    ).toBe("text");
  });

  it("returns the user negative untouched for a preset without one", () => {
    expect(
      mergeStyleNegative("text", "watercolor", {
        supportsNegativePrompt: true,
      }),
    ).toBe("text");
  });

  it("returns the user negative untouched for an empty or unknown id", () => {
    expect(
      mergeStyleNegative("text", "", { supportsNegativePrompt: true }),
    ).toBe("text");
    expect(
      mergeStyleNegative("text", "nope", { supportsNegativePrompt: true }),
    ).toBe("text");
    expect(
      mergeStyleNegative(undefined, "", { supportsNegativePrompt: true }),
    ).toBe("");
  });

  it("resolves legacy ids", () => {
    expect(
      mergeStyleNegative("", "photographic", { supportsNegativePrompt: true }),
    ).toBe(
      mergeStyleNegative("", "photoreal", { supportsNegativePrompt: true }),
    );
  });

  it("is the same merge composeStyle applies", () => {
    const composed = composeStyle("a cat", "photoreal", {
      supportsNegativePrompt: true,
      negative: "text, watermark",
    });
    expect(composed.negative).toBe(
      mergeStyleNegative("text, watermark", "photoreal", {
        supportsNegativePrompt: true,
      }),
    );
  });
});

describe("legacy id resolution", () => {
  it("maps every id from both pre-kit tables to a canonical preset", () => {
    const canonical = new Set(STYLE_PRESETS.map((p) => p.id));
    for (const id of [...DESKTOP_LEGACY_IDS, ...WEB_LEGACY_IDS]) {
      expect(id in LEGACY_STYLE_IDS, id).toBe(true);
      expect(canonical.has(resolveStyleId(id)), id).toBe(true);
    }
  });

  it("re-homes the desktop-only ids where the look fits", () => {
    expect(resolveStyleId("photographic")).toBe("photoreal");
    expect(resolveStyleId("3d-render")).toBe("3d");
    expect(resolveStyleId("neon")).toBe("cinematic");
    expect(resolveStyleId("minimal")).toBe("illustration");
  });

  it("keeps canonical ids as themselves", () => {
    for (const preset of STYLE_PRESETS) {
      expect(resolveStyleId(preset.id)).toBe(preset.id);
    }
  });

  it("passes unknown and empty ids through unchanged", () => {
    expect(resolveStyleId("")).toBe("");
    expect(resolveStyleId("mystery")).toBe("mystery");
  });

  it("finds a preset through an alias", () => {
    expect(stylePresetById("photographic")?.id).toBe("photoreal");
    expect(stylePresetById("")).toBeNull();
    expect(stylePresetById("nope")).toBeNull();
  });
});

describe("styleHint", () => {
  it("folds a templated positive into a readable directive", () => {
    expect(styleHint("cinematic")).toBe(
      "Cinematic look — cinematic film still, cinematic lighting, anamorphic, dramatic mood, subtle film grain",
    );
  });

  it("uses the whole positive for append-style presets", () => {
    expect(styleHint("product")).toBe(
      "Product look — product photography, seamless backdrop, softbox lighting",
    );
  });

  it("never leaks the placeholder and always keeps the label — fragment shape", () => {
    for (const preset of STYLE_PRESETS) {
      const hint = styleHint(preset.id);
      expect(hint, preset.id).toMatch(/^.+ look — .+$/);
      expect(hint, preset.id).not.toContain("{prompt}");
      expect(hint, preset.id).not.toContain(" of,");
    }
  });

  it("resolves legacy ids and returns null for unknown ones", () => {
    expect(styleHint("photographic")).toBe(styleHint("photoreal"));
    expect(styleHint("")).toBeNull();
    expect(styleHint("nope")).toBeNull();
  });
});

describe("variation angles", () => {
  it("keeps the five prototype angles verbatim", () => {
    expect(STYLE_ANGLES).toEqual([
      "at golden hour",
      "in moody overcast light",
      "in dramatic low-key light",
      "with soft diffused daylight",
      "backlit with warm rim light",
    ]);
  });

  it("cycles by index, wrapping past the end and below zero", () => {
    expect(angleForIndex(0)).toBe("at golden hour");
    expect(angleForIndex(4)).toBe("backlit with warm rim light");
    expect(angleForIndex(5)).toBe("at golden hour");
    expect(angleForIndex(6)).toBe("in moody overcast light");
    expect(angleForIndex(-1)).toBe("backlit with warm rim light");
  });
});
