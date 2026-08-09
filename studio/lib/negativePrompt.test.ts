import { describe, expect, it } from "vitest";
import {
  WAN_FAMILY_DEFAULT_NEGATIVE_PROMPT,
  advertisedNegativeDefault,
  effectiveNegativeDefault,
  negativePromptOnDefaultChange,
  negativePromptWireValue,
  restoredNegativePrompt,
} from "./negativePrompt";

// A stand-in for wan's tuned default; the real value is the server's concern —
// clients only ever compare against what the row advertised.
const WAN_DEFAULT = "色调艳丽，过曝，静态，细节模糊不清";

// These cases are the browser half of the cross-surface parity contract;
// `crates/mold-tui/src/ui/create_form.rs` pins the identical set for the TUI.
describe("advertisedNegativeDefault", () => {
  it("normalizes the advertised row value and absence alike", () => {
    expect(
      advertisedNegativeDefault({ default_negative_prompt: WAN_DEFAULT }),
    ).toBe(WAN_DEFAULT);
    expect(
      advertisedNegativeDefault({
        default_negative_prompt: `  ${WAN_DEFAULT}  `,
      }),
    ).toBe(WAN_DEFAULT);
    expect(advertisedNegativeDefault({ default_negative_prompt: null })).toBe(
      "",
    );
    expect(advertisedNegativeDefault({})).toBe("");
    expect(advertisedNegativeDefault(null)).toBe("");
    expect(advertisedNegativeDefault(undefined)).toBe("");
  });
});

describe("effectiveNegativeDefault", () => {
  it("prefers the advertised row value, trimmed", () => {
    expect(
      effectiveNegativeDefault({ default_negative_prompt: " custom " }, "wan"),
    ).toBe("custom");
  });

  it("falls back to the wan family constant when the additive field is absent", () => {
    // Older servers omit `default_negative_prompt`; the known family default
    // must survive so an explicit "" opt-out keeps serializing as "".
    expect(effectiveNegativeDefault({}, "wan")).toBe(
      WAN_FAMILY_DEFAULT_NEGATIVE_PROMPT,
    );
    expect(effectiveNegativeDefault(null, "wan")).toBe(
      WAN_FAMILY_DEFAULT_NEGATIVE_PROMPT,
    );
    expect(
      effectiveNegativeDefault({ default_negative_prompt: null }, "wan"),
    ).toBe(WAN_FAMILY_DEFAULT_NEGATIVE_PROMPT);
  });

  it("stays empty for families without an engine fallback", () => {
    expect(effectiveNegativeDefault({}, "sdxl")).toBe("");
    expect(effectiveNegativeDefault(null, "")).toBe("");
    expect(effectiveNegativeDefault(undefined, undefined)).toBe("");
  });

  it("keeps an explicit clear an explicit clear against an older server", () => {
    // The regression this guards (#787 round 2): absent advertisement must
    // not decay the opt-out's wire value from "" to undefined.
    const decayedDefault = effectiveNegativeDefault({}, "wan");
    expect(negativePromptWireValue("", decayedDefault)).toBe("");
  });
});

describe("negativePromptWireValue", () => {
  it("keeps an untouched default absent so older servers behave identically", () => {
    expect(negativePromptWireValue(WAN_DEFAULT, WAN_DEFAULT)).toBeUndefined();
    expect(
      negativePromptWireValue(` ${WAN_DEFAULT} `, WAN_DEFAULT),
    ).toBeUndefined();
  });

  it("ships the explicit empty opt-out when the user cleared a defaulted field", () => {
    expect(negativePromptWireValue("", WAN_DEFAULT)).toBe("");
    expect(negativePromptWireValue("   ", WAN_DEFAULT)).toBe("");
  });

  it("sends typed text verbatim", () => {
    expect(negativePromptWireValue("blurry", WAN_DEFAULT)).toBe("blurry");
    expect(negativePromptWireValue("blurry", "")).toBe("blurry");
  });

  it("keeps today's behavior for models without a default", () => {
    expect(negativePromptWireValue("", "")).toBeUndefined();
    expect(negativePromptWireValue("  ", "")).toBeUndefined();
  });
});

describe("negativePromptOnDefaultChange", () => {
  it("prefills the default into an untouched control", () => {
    expect(negativePromptOnDefaultChange("", "", WAN_DEFAULT)).toBe(
      WAN_DEFAULT,
    );
  });

  it("follows the model while the control still shows the old default", () => {
    expect(negativePromptOnDefaultChange(WAN_DEFAULT, WAN_DEFAULT, "")).toBe(
      "",
    );
    expect(
      negativePromptOnDefaultChange(WAN_DEFAULT, WAN_DEFAULT, "other"),
    ).toBe("other");
  });

  it("never clobbers typed text", () => {
    expect(negativePromptOnDefaultChange("hands", WAN_DEFAULT, "")).toBe(
      "hands",
    );
    expect(negativePromptOnDefaultChange("hands", "", WAN_DEFAULT)).toBe(
      "hands",
    );
  });

  it("preserves an explicit clear across a wan→wan switch", () => {
    expect(negativePromptOnDefaultChange("", WAN_DEFAULT, WAN_DEFAULT)).toBe(
      "",
    );
  });
});

describe("restoredNegativePrompt", () => {
  it("reads absence as the default that actually conditioned the render", () => {
    expect(restoredNegativePrompt(undefined, WAN_DEFAULT)).toBe(WAN_DEFAULT);
    expect(restoredNegativePrompt(null, WAN_DEFAULT)).toBe(WAN_DEFAULT);
    expect(restoredNegativePrompt(undefined, "")).toBe("");
  });

  it("keeps a recorded explicit empty uncond empty", () => {
    expect(restoredNegativePrompt("", WAN_DEFAULT)).toBe("");
  });

  it("keeps recorded text verbatim", () => {
    expect(restoredNegativePrompt("blurry", WAN_DEFAULT)).toBe("blurry");
  });
});
