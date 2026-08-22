import { describe, expect, it } from "vitest";
import { AUTO_TARGET_ID, CAPABLE_TARGET_ID } from "@studio/lib/hostRouting";
import type { MobileHost } from "./hosts";
import { mobileFileUnderAvailable, mobileFileUnderCollections } from "./fileUnder";

function host(id: string, extra: Partial<MobileHost> = {}): MobileHost {
  return {
    id,
    name: id,
    baseUrl: `http://${id}:7680`,
    apiKey: "",
    hostname: id,
    version: "0.23.0",
    online: true,
    ...extra,
  };
}

const organizes = { gallery: { can_delete: true, organize: true } };
const cannotOrganize = { gallery: { can_delete: true, organize: false } };

describe("mobileFileUnderAvailable", () => {
  it("offers the group when the pinned machine reports gallery.organize", () => {
    expect(mobileFileUnderAvailable("studio", [host("studio")], { studio: organizes })).toBe(true);
  });

  it("stays hidden when the pinned machine cannot file", () => {
    expect(mobileFileUnderAvailable("studio", [host("studio")], { studio: cannotOrganize })).toBe(
      false,
    );
  });

  it("treats an unread capability snapshot as no evidence", () => {
    // Positive knowledge only: an unread or failed probe hides the group and
    // sends nothing, exactly as the V3 Library gate does.
    expect(mobileFileUnderAvailable("studio", [host("studio")], {})).toBe(false);
    expect(mobileFileUnderAvailable("studio", [host("studio")], { studio: null })).toBe(false);
  });

  it("stays hidden when a peer can file but the pinned machine cannot", () => {
    expect(
      mobileFileUnderAvailable("studio", [host("studio"), host("plato")], {
        studio: cannotOrganize,
        plato: organizes,
      }),
    ).toBe(false);
  });

  it("offers the group under an automatic policy when any reachable machine can file", () => {
    const hosts = [host("studio"), host("plato")];
    const capabilities = { studio: cannotOrganize, plato: organizes };

    expect(mobileFileUnderAvailable(AUTO_TARGET_ID, hosts, capabilities)).toBe(true);
    expect(mobileFileUnderAvailable(CAPABLE_TARGET_ID, hosts, capabilities)).toBe(true);
  });

  it("ignores an unreachable or disconnected machine under an automatic policy", () => {
    expect(
      mobileFileUnderAvailable(AUTO_TARGET_ID, [host("studio"), host("plato", { online: false })], {
        studio: cannotOrganize,
        plato: organizes,
      }),
    ).toBe(false);
    expect(
      mobileFileUnderAvailable(
        AUTO_TARGET_ID,
        [host("studio"), host("plato", { connected: false })],
        { studio: cannotOrganize, plato: organizes },
      ),
    ).toBe(false);
  });

  it("answers for the candidate set it is given, not the whole fleet", () => {
    // Callers hand automatic routing the model-aware, access-filtered
    // candidates. A peer that can file but cannot run the checkpoint is not in
    // that list, so it must not qualify a group whose print routes elsewhere.
    const capabilities = { studio: cannotOrganize, plato: organizes };

    expect(mobileFileUnderAvailable(AUTO_TARGET_ID, [host("studio")], capabilities)).toBe(false);
    expect(
      mobileFileUnderAvailable(AUTO_TARGET_ID, [host("studio"), host("plato")], capabilities),
    ).toBe(true);
  });

  it("hides the group when nothing is targeted at all", () => {
    expect(mobileFileUnderAvailable("", [], {})).toBe(false);
  });
});

describe("mobileFileUnderCollections", () => {
  it("keeps only the merge key and the count the picker renders", () => {
    expect(
      mobileFileUnderCollections([
        {
          slug: "smurfs",
          name: "Smurfs",
          count: 12,
          hostIds: ["studio", "plato"],
          hostsLabel: "studio · plato",
          cover: null,
          hidden: false,
        },
      ]),
    ).toEqual([{ name: "Smurfs", slug: "smurfs", count: 12 }]);
  });
});
