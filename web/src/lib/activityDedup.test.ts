import { describe, expect, it } from "vitest";
import {
  localRowHiddenFromStrip,
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

describe("localRowHiddenFromStrip", () => {
  it("hides a settled DETACHED row whatever the fleet is reporting", () => {
    // The retained job later finishes and leaves the host's active work, so
    // no shared row masks it any more. Rendering it would label a print that
    // landed successfully in the Library "Failed" for five minutes.
    const retained = job({ state: "error", detached: true });
    expect(localRowHiddenFromStrip(retained, [row()], ORIGIN)).toBe(true);
    expect(localRowHiddenFromStrip(retained, [], ORIGIN)).toBe(true);
  });

  it("keeps a detached row that is still running", () => {
    // Rehydrated after a reload with its server id known: the reconciler, not
    // this boot, decides its fate, and meanwhile it is live work.
    expect(
      localRowHiddenFromStrip(
        job({ state: "running", detached: true }),
        [],
        ORIGIN,
      ),
    ).toBe(false);
  });

  it("keeps an ordinary failure the fleet does not claim", () => {
    expect(localRowHiddenFromStrip(job({ state: "error" }), [], ORIGIN)).toBe(
      false,
    );
  });

  it("drops the settled local failure the live server row replaces", () => {
    expect(
      localRowHiddenFromStrip(job({ state: "error" }), [row()], ORIGIN),
    ).toBe(true);
  });

  it("keeps a settled failure the fleet does not claim is running", () => {
    expect(localRowHiddenFromStrip(job({ state: "error" }), [], ORIGIN)).toBe(
      false,
    );
    expect(
      localRowHiddenFromStrip(
        job({ state: "error", serverId: null }),
        [row()],
        ORIGIN,
      ),
    ).toBe(false);
    expect(
      localRowHiddenFromStrip(
        job({ state: "error" }),
        [row({ kind: "download" })],
        ORIGIN,
      ),
    ).toBe(false);
  });

  it("never hides a running or completed local row", () => {
    expect(localRowHiddenFromStrip(job(), [row()], ORIGIN)).toBe(false);
    expect(
      localRowHiddenFromStrip(job({ state: "done" }), [row()], ORIGIN),
    ).toBe(false);
  });
});
