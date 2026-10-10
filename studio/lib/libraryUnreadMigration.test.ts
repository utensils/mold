import { expect, it } from "vitest";
import { LibraryUnreadLedger } from "./libraryUnreadLedger";
import { observeTimestampViewingHistory } from "./libraryUnreadMigration";

it("preserves mobile timestamp read cutoffs while keeping future arrivals individually unread", () => {
  const ledger = new LibraryUnreadLedger();
  const old = { hostId: "one", filename: "old.png", timestamp: 10 };
  const fresh = { hostId: "one", filename: "fresh.png", timestamp: 30 };
  observeTimestampViewingHistory(ledger, [old, fresh], ["one"], { one: 20 });
  expect(ledger.isUnread(["one|old.png"])).toBe(false);
  expect(ledger.isUnread(["one|fresh.png"])).toBe(true);
  observeTimestampViewingHistory(ledger, [old, fresh], ["one"], { one: 99 });
  expect(ledger.isUnread(["one|fresh.png"])).toBe(true);
  ledger.view(["one|fresh.png"]);
  expect(
    new LibraryUnreadLedger(ledger.serialize()).isUnread(["one|fresh.png"]),
  ).toBe(false);
});
