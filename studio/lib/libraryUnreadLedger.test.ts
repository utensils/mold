import { describe, expect, it, vi } from "vitest";
import { LibraryUnreadLedger } from "./libraryUnreadLedger";

describe("client-local gallery viewing history", () => {
  it("baselines each host, preserves arrivals until individually viewed, and survives restart", () => {
    let ledger = new LibraryUnreadLedger();
    const old = ["one|old.png"],
      a = ["one|a.png"],
      b = ["one|b.png"];
    ledger.observe([old], ["one"]);
    expect(ledger.isUnread(old)).toBe(false);
    ledger.observe([old, a, b], ["one"]);
    expect(ledger.count([old, a, b])).toBe(2);
    ledger.view(a);
    ledger = new LibraryUnreadLedger(ledger.serialize());
    ledger.observe([old, a, b], ["one"]);
    expect(ledger.isUnread(a)).toBe(false);
    expect(ledger.isUnread(b)).toBe(true);
    ledger.observe([old, a, b, ["two|history.png"]], ["one", "two"]);
    expect(ledger.isUnread(["two|history.png"])).toBe(false);
  });

  it("inherits viewed status across merged renamed copies and distinguishes same-name outputs", () => {
    const ledger = new LibraryUnreadLedger();
    ledger.observe([], ["one", "two"]);
    ledger.observe([["one|same.png"], ["two|same.png"]], ["one", "two"]);
    expect(ledger.count([["one|same.png"], ["two|same.png"]])).toBe(2);
    ledger.view(["one|same.png"]);
    ledger.observe(
      [["one|same.png", "two|renamed.png"], ["two|same.png"]],
      ["one", "two"],
    );
    expect(ledger.isUnread(["two|renamed.png"])).toBe(false);
    expect(ledger.isUnread(["two|same.png"])).toBe(true);
  });

  it("does not confuse visibility/removal with viewing or leak state between clients", () => {
    const ledger = new LibraryUnreadLedger();
    ledger.observe([], ["one"]);
    ledger.observe([["one|a.png"]], ["one"]);
    expect(ledger.count([])).toBe(0);
    ledger.observe([], ["one"]);
    ledger.observe([["one|a.png"]], ["one"]);
    expect(ledger.count([["one|a.png"]])).toBe(1);
    const separate = new LibraryUnreadLedger();
    separate.observe([["one|a.png"]], ["one"]);
    expect(separate.isUnread(["one|a.png"])).toBe(false);
    expect(ledger.isUnread(["one|a.png"])).toBe(true);
  });
});

it("preserves a migrated known-unread record without re-baselining it", () => {
  const migrated = new LibraryUnreadLedger(
    JSON.stringify({
      version: 1,
      known: [["one", ["one|read.png", "one|unread.png"]]],
      unread: ["one|unread.png"],
    }),
  );
  migrated.observe(
    [["one|read.png"], ["one|unread.png"], ["one|later.png"]],
    ["one"],
  );
  expect(
    migrated.count([["one|read.png"], ["one|unread.png"], ["one|later.png"]]),
  ).toBe(2);
});

it("fits realistic 20k-print three-host history with quota headroom and preserves 100k records", () => {
  function inventory(count: number) {
    const ledger = new LibraryUnreadLedger();
    const hosts = [
      "gpu-workstation-192-168-1-100",
      "studio-mac-192-168-1-101",
      "render-node-192-168-1-102",
    ];
    ledger.observe([], hosts);
    ledger.observe(
      Array.from({ length: count }, (_, i) =>
        hosts.map(
          (host) =>
            `${host}|flux-2-klein-9b-distilled-q4-k-m-${String(i).padStart(10, "0")}-1723456789-original.png`,
        ),
      ),
      hosts,
    );
    return ledger;
  }
  const normal = inventory(20_000);
  const encoded = normal.serialize();
  // Conservative UTF-16 accounting and 512 KiB left for other app preferences.
  expect(encoded.length * 2).toBeLessThan(5 * 1024 * 1024 - 512 * 1024);
  expect(new LibraryUnreadLedger(encoded).unread.size).toBe(60_000);
  const large = inventory(100_000);
  const largeEncoded = large.serialize();
  expect(largeEncoded.length * 2).toBeGreaterThan(5 * 1024 * 1024);
  expect(new LibraryUnreadLedger(largeEncoded).unread.size).toBe(300_000);
}, 90_000);

it("reports failed persistence and preserves corrupt stored history", async () => {
  const { saveLibraryUnreadLedger, loadLibraryUnreadLedger } =
    await import("./libraryUnreadLedger");
  const spy = vi.spyOn(localStorage, "setItem").mockImplementation(() => {
    throw new DOMException("Quota full", "QuotaExceededError");
  });
  expect(
    saveLibraryUnreadLedger("fixture", new LibraryUnreadLedger()),
  ).toContain("could not be saved");
  spy.mockRestore();
  const corrupt = new LibraryUnreadLedger("{broken");
  const write = vi.spyOn(localStorage, "setItem");
  expect(saveLibraryUnreadLedger("fixture", corrupt)).toContain(
    "could not be read",
  );
  expect(write).not.toHaveBeenCalled();
  write.mockRestore();
  const read = vi.spyOn(localStorage, "getItem").mockImplementation(() => {
    throw new Error("Storage unavailable");
  });
  const unavailable = loadLibraryUnreadLedger("fixture");
  const blockedWrite = vi.spyOn(localStorage, "setItem");
  expect(saveLibraryUnreadLedger("fixture", unavailable)).toContain(
    "unavailable",
  );
  expect(blockedWrite).not.toHaveBeenCalled();
  read.mockRestore();
  blockedWrite.mockRestore();
});

it("persists viewing before inventory without consuming the host baseline", () => {
  let ledger = new LibraryUnreadLedger();
  ledger.view(["a|result.png"]);
  ledger = new LibraryUnreadLedger(ledger.serialize());
  ledger.observe([["a|result.png"], ["a|history.png"]], ["a"]);
  expect(ledger.count([["a|result.png"], ["a|history.png"]])).toBe(0);
  ledger.view(["a|arriving.png"]);
  ledger = new LibraryUnreadLedger(ledger.serialize());
  ledger.observe([["a|arriving.png"], ["a|other.png"]], ["a"]);
  expect(ledger.isUnread(["a|arriving.png"])).toBe(false);
  expect(ledger.isUnread(["a|other.png"])).toBe(true);
});
