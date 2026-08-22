import { describe, expect, it } from "vitest";
import {
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
