import { expect, it } from "vitest";
import {
  provisionalCopyHiddenKeys,
  retainCopyPrivacyUntilConfirmed,
} from "./copyPrivacy";
const identity = JSON.stringify([
  "destination",
  "https://destination",
  "regular",
  "instance",
]);
const evidence = [
  {
    source: "source",
    destination: identity,
    sourceFilename: "source.png",
    destinationFilename: "renamed.png",
  },
];
const host = {
  id: "destination",
  url: "https://destination",
  connectionMode: "regular" as const,
  instanceId: "instance",
};
it("hides a pending renamed physical output without seed or metadata identity", () => {
  expect(provisionalCopyHiddenKeys(evidence, [host])).toEqual(
    new Set(["destination|renamed.png"]),
  );
});
it.each([
  { ...host, url: "https://replacement" },
  { ...host, instanceId: "replacement" },
  { ...host, connectionMode: "secure" as const },
])("rejects changed destination authority %j", (changed) => {
  expect(provisionalCopyHiddenKeys(evidence, [changed])).toEqual(new Set());
});
it("withholds privacy until an unknown installation can be checked and rejects forgotten hosts", () => {
  expect(
    provisionalCopyHiddenKeys(evidence, [{ ...host, instanceId: undefined }]),
  ).toEqual(new Set(["destination|renamed.png"]));
  expect(provisionalCopyHiddenKeys(evidence, [])).toEqual(new Set());
});

it("keeps completed quarantine until a fresh physical privacy inventory confirms it", () => {
  const row = {
    ...evidence[0]!,
    stage: "complete",
    sourceOrganization: { item: {}, collections: [{ hidden: true }] },
  } as unknown as import("@studio/api/galleryCopy").CopyReceipt;
  expect(
    retainCopyPrivacyUntilConfirmed([], evidence, { row }, new Set()),
  ).toEqual(evidence);
  expect(
    retainCopyPrivacyUntilConfirmed(
      [],
      evidence,
      { row },
      new Set(["destination|renamed.png"]),
    ),
  ).toEqual([]);
  expect(retainCopyPrivacyUntilConfirmed([], evidence, {}, new Set())).toEqual(
    [],
  );
});

it("does not release a renamed destination from a peer's logical-group hidden membership", async () => {
  const { confirmedPhysicalCopyPrivacyKeys } =
    await import("@studio/lib/copyPrivacy");
  const rows = [
    {
      hostId: "source",
      filename: "source.png",
      collections: ["private-source"],
    },
    {
      hostId: "destination",
      filename: "renamed.png",
      collections: [] as string[],
    },
  ];
  const hosts = [
    {
      hostId: "source",
      collectionsListingOk: true,
      collections: [{ id: "private-source", hidden: true }],
    },
    {
      hostId: "destination",
      collectionsListingOk: true,
      collections: [{ id: "private-destination", hidden: false }],
    },
  ];
  const complete = {
    ...evidence[0]!,
    stage: "complete",
    sourceOrganization: { item: {}, collections: [{ hidden: true }] },
  } as unknown as import("@studio/api/galleryCopy").CopyReceipt;
  const confirmed = () => confirmedPhysicalCopyPrivacyKeys(rows, hosts);
  expect(confirmed()).toEqual(new Set(["source|source.png"]));
  expect(
    retainCopyPrivacyUntilConfirmed([], evidence, { complete }, confirmed()),
  ).toEqual(evidence);
  rows[1]!.collections = ["private-destination"];
  expect(
    retainCopyPrivacyUntilConfirmed([], evidence, { complete }, confirmed()),
  ).toEqual(evidence);
  hosts[1]!.collections[0]!.hidden = true;
  expect(
    retainCopyPrivacyUntilConfirmed([], evidence, { complete }, confirmed()),
  ).toEqual([]);
  hosts[1]!.collectionsListingOk = false;
  expect(
    retainCopyPrivacyUntilConfirmed([], evidence, { complete }, confirmed()),
  ).toEqual(evidence);
});
