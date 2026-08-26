import { describe, expect, it } from "vitest";
import mobileAppSource from "./MobileApp.vue?raw";

/**
 * iPhone half of the optional-prompt rule. A conditioned LTX-2 render may go
 * out undescribed, so the Develop button and the pre-submit guard both stop
 * demanding a prompt the host would accept — and, like desktop, they have to
 * move together or the enabled control becomes a dead end.
 */
describe("MobileApp prompt requirement", () => {
  it("derives the Develop gate from the shared predicate", () => {
    expect(mobileAppSource).toContain('from "@studio/lib/promptRequirement"');
    expect(mobileAppSource).toMatch(
      /const promptMissing = computed\(\(\) => promptRequired\(form\) && !form\.prompt\.trim\(\)\);/,
    );
  });

  it("uses that one gate for both the button and the pre-submit guard", () => {
    const occurrences = mobileAppSource.match(/promptMissing(\.value)?\b/g) ?? [];
    // definition + the shared Develop-disabled computation + the `generate()` guard
    expect(occurrences.length).toBeGreaterThanOrEqual(3);
    expect(mobileAppSource).toMatch(/promptMissing\.value \|\|\s*!form\.model\.trim\(\)/);
    expect(mobileAppSource).toContain(':disabled="developDisabled"');
    // The bare check must not survive anywhere in the generation path; only
    // "Use as prompt" may still skip a print with no recorded prompt.
    expect(mobileAppSource).not.toContain("!form.prompt.trim() ||");
  });

  it("softens the prompt placeholder through the shared helper", () => {
    expect(mobileAppSource).toMatch(
      /const promptFieldPlaceholder = computed\(\(\) => promptPlaceholder\(form, "Describe the print…"\)\);/,
    );
    expect(mobileAppSource).toContain(':placeholder="promptFieldPlaceholder"');
  });
});
