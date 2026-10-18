import { expect, it } from "vitest";
import type { CopyReceipt } from "../api/galleryCopy";
import { provisionalCopyPrivacy } from "./copyPrivacy";
const receipt = (stage: CopyReceipt["stage"]): CopyReceipt => ({
  version: 1,
  source: "source",
  destination: "destination",
  sourceFilename: "print.png",
  destinationFilename: "print-2.png",
  digest: "digest",
  recipe: "recipe",
  size: 1,
  stage,
  sourceOrganization: {
    item: {},
    collections: [
      {
        id: "7",
        name: "Private",
        slug: "private",
        hidden: true,
        description: null,
        cover_filename: null,
        count: 1,
        created_at: 1,
        updated_at: 1,
      },
    ],
  },
});
it.each(["intent", "output", "sources", "organization"] as const)(
  "keeps %s collision-renamed output hidden before organization finishes",
  (stage) => {
    const row = receipt(stage);
    const privacy = provisionalCopyPrivacy({ pending: row });
    expect(privacy).toEqual([
      {
        source: "source",
        destination: "destination",
        sourceFilename: "print.png",
        destinationFilename: "print-2.png",
      },
    ]);
    expect(row.stage).toBe(stage);
    expect(Object.keys(privacy[0]!)).not.toContain("digest");
  },
);
it("stops provisional evidence after verified completion and never invents a target", () => {
  const completed = receipt("complete"),
    unknown = receipt("intent"),
    ordinary = receipt("output");
  delete unknown.destinationFilename;
  ordinary.sourceOrganization!.collections[0]!.hidden = false;
  expect(provisionalCopyPrivacy({ completed, unknown, ordinary })).toEqual([]);
});
