import { describe, expect, it } from "vitest";
import {
  localFailureSupersededByShared,
  sharedRowIsLocallyOwned,
  type LocalActivityJob,
  type SharedActivityRow,
} from "./activityDedup";

const ORIGIN = "origin";

function job(overrides: Partial<LocalActivityJob> = {}): LocalActivityJob {
  return { serverId: "srv-1", hostId: null, state: "running", ...overrides };
}

function row(overrides: Partial<SharedActivityRow> = {}): SharedActivityRow {
  return { kind: "generation", id: "srv-1", hostId: ORIGIN, ...overrides };
}

describe("sharedRowIsLocallyOwned", () => {
  it("hides the server row while this session still streams the job", () => {
    expect(sharedRowIsLocallyOwned(row(), [job()], ORIGIN)).toBe(true);
  });

  it("keeps the live server row once the local row has settled", () => {
    // A retained job settles locally the moment its stream dies. The host is
    // still rendering it, so the server row is the truth and must survive —
    // otherwise the resumed job is both "failed" and invisible.
    expect(
      sharedRowIsLocallyOwned(row(), [job({ state: "error" })], ORIGIN),
    ).toBe(false);
  });

  it("never matches across hosts or across ids", () => {
    expect(
      sharedRowIsLocallyOwned(row({ hostId: "studio" }), [job()], ORIGIN),
    ).toBe(false);
    expect(sharedRowIsLocallyOwned(row({ id: "srv-2" }), [job()], ORIGIN)).toBe(
      false,
    );
    expect(
      sharedRowIsLocallyOwned(row(), [job({ serverId: null })], ORIGIN),
    ).toBe(false);
  });
});

describe("localFailureSupersededByShared", () => {
  it("drops the settled local failure the live server row replaces", () => {
    expect(
      localFailureSupersededByShared(job({ state: "error" }), [row()], ORIGIN),
    ).toBe(true);
  });

  it("keeps a settled failure the fleet does not claim is running", () => {
    expect(
      localFailureSupersededByShared(job({ state: "error" }), [], ORIGIN),
    ).toBe(false);
    expect(
      localFailureSupersededByShared(
        job({ state: "error", serverId: null }),
        [row()],
        ORIGIN,
      ),
    ).toBe(false);
    expect(
      localFailureSupersededByShared(
        job({ state: "error" }),
        [row({ kind: "download" })],
        ORIGIN,
      ),
    ).toBe(false);
  });

  it("never hides a running or completed local row", () => {
    expect(localFailureSupersededByShared(job(), [row()], ORIGIN)).toBe(false);
    expect(
      localFailureSupersededByShared(job({ state: "done" }), [row()], ORIGIN),
    ).toBe(false);
  });
});
