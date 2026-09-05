import { describe, expect, it } from "vitest";
import {
  licenseDetailLine,
  licenseFriendlyLine,
  acceptanceFor,
  licenseFromErrorBody,
  licenseRequirements,
  type LicenseTerms,
} from "./licenseAcceptance";

const terms: LicenseTerms = {
  id: "research-assets-v1",
  name: "Research assets",
  url: "https://example.test/pinned",
  canonical: "https://example.test/license",
  sha256: "a".repeat(64),
  summary: "Non-commercial research only.",
};

describe("license acceptance contract", () => {
  it("groups registry terms by install bundle without knowing ids", () => {
    const companionTerms = {
      ...terms,
      id: "future-runtime-v2",
      name: "Future runtime",
      url: "https://example.test/future-runtime-pinned",
      sha256: "b".repeat(64),
    };
    expect(
      licenseRequirements([
        {
          install_model: "future-bundle",
          licenses: [terms, companionTerms],
        },
        { install_model: "future-bundle", licenses: [{ ...terms }] },
        {
          install_model: "other-bundle",
          licenses: [{ ...terms, id: "other" }],
        },
      ]),
    ).toEqual([
      {
        installModel: "future-bundle",
        licenses: [terms, companionTerms],
      },
      { installModel: "other-bundle", licenses: [{ ...terms, id: "other" }] },
    ]);
  });

  it("sends the exact pinned terms that were displayed", () => {
    expect(acceptanceFor(terms)).toEqual({
      id: terms.id,
      url: terms.url,
      sha256: terms.sha256,
    });
  });

  it("reads refreshed server terms structurally and ignores prose", () => {
    expect(
      licenseFromErrorBody({ code: "LICENSE_TERMS_MISMATCH", license: terms }),
    ).toEqual(terms);
    expect(licenseFromErrorBody({ error: `accept ${terms.id}` })).toBeNull();
  });
});

/*
 * A licence row leads with what it unlocks. The registry's plain words when
 * the host lists them, the style ids on an older host, and the licence's own
 * name only when nothing requires it — in which case the id moves to the
 * mono line so the name is not said twice.
 */
describe("licence row lines", () => {
  const base = {
    id: "tencent-hunyuan3d-2.1",
    name: "Tencent Hunyuan 3D 2.1 Community License",
    url: "https://example.test/pinned",
    canonical: "https://example.test/project",
    sha256: "a".repeat(64),
    summary: "Community licence; some uses need Tencent's permission.",
    accepted: false,
  };

  it("leads with the styles' descriptions when the host lists them", () => {
    const status = {
      ...base,
      required_by: ["hunyuan3d-mini-turbo:fp16", "hunyuan3d-2.1:fp16"],
      required_by_styles: [
        {
          name: "hunyuan3d-mini-turbo:fp16",
          description: "3-D objects from a photo, fast",
        },
        {
          name: "hunyuan3d-2.1:fp16",
          description: "3-D objects from a photo, full quality",
        },
      ],
    };
    expect(licenseFriendlyLine(status)).toBe(
      "3-D objects from a photo, fast · 3-D objects from a photo, full quality",
    );
    expect(licenseDetailLine(status)).toBe(
      "Tencent Hunyuan 3D 2.1 Community License · Community licence; some uses need Tencent's permission.",
    );
  });

  it("falls back to the style ids on an older host, and skips blank descriptions", () => {
    const older = { ...base, required_by: ["hunyuan3d-2.1:fp16"] };
    expect(licenseFriendlyLine(older)).toBe("hunyuan3d-2.1:fp16");
    const blank = {
      ...older,
      required_by_styles: [{ name: "hunyuan3d-2.1:fp16", description: "  " }],
    };
    expect(licenseFriendlyLine(blank)).toBe("hunyuan3d-2.1:fp16");
  });

  it("leads with the licence's name when nothing requires it, and moves the id down", () => {
    const orphan = { ...base, required_by: [] };
    expect(licenseFriendlyLine(orphan)).toBe(base.name);
    expect(licenseDetailLine(orphan)).toBe(
      "tencent-hunyuan3d-2.1 · Community licence; some uses need Tencent's permission.",
    );
  });
});
