import { beforeEach, describe, expect, it, vi } from "vitest";
import { apiJsonTo, type ApiTarget } from "./client";
import {
  clearScope,
  fetchHistoryAll,
  groupByDay,
  type HistoryEntry,
  type HistoryHostTarget,
} from "./history";

vi.mock("./client", () => ({
  apiJsonTo: vi.fn(),
  apiFetchTo: vi.fn(),
  currentTarget: vi.fn(),
}));

// used_at is epoch milliseconds on the wire (mold_core::HistoryEntry).
const at = (iso: string): number => new Date(iso).getTime();
const entry = (prompt: string, iso: string): HistoryEntry => ({
  prompt,
  model: "flux2-klein:q4",
  used_at: at(iso),
});

describe("groupByDay", () => {
  const now = new Date("2026-07-08T20:00:00");

  it("labels today and yesterday, then dates", () => {
    const groups = groupByDay(
      [
        entry("a", "2026-07-08T14:00:00"),
        entry("b", "2026-07-07T22:00:00"),
        entry("c", "2026-07-01T10:00:00"),
      ],
      now,
    );
    expect(groups.map((g) => g.label)).toEqual(["Today", "Yesterday", "July 1"]);
  });

  it("keeps consecutive same-day entries in one group", () => {
    const groups = groupByDay(
      [entry("a", "2026-07-08T14:00:00"), entry("b", "2026-07-08T09:00:00")],
      now,
    );
    expect(groups).toHaveLength(1);
    expect(groups[0]!.entries.map((e) => e.prompt)).toEqual(["a", "b"]);
  });

  it("handles empty input", () => {
    expect(groupByDay([], now)).toEqual([]);
  });

  it("carries extra fields (host tags) through grouping", () => {
    const tagged = [{ ...entry("a", "2026-07-08T14:00:00"), hostLabel: "hal9000" }];
    const groups = groupByDay(tagged, now);
    expect(groups[0]!.entries[0]!.hostLabel).toBe("hal9000");
  });
});

// ── Multi-host fan-out ───────────────────────────────────────────────────────

const host = (hostId: string, label: string, baseUrl: string): HistoryHostTarget => ({
  hostId,
  label,
  target: { baseUrl, apiKey: null },
});

const local = host("local", "This Mac", "http://127.0.0.1:49152");
const hal = host("hal9000-7680", "hal9000", "http://hal9000:7680");

describe("fetchHistoryAll", () => {
  beforeEach(() => {
    vi.mocked(apiJsonTo).mockReset();
  });

  it("merges, tags, and sorts entries from every host by used_at desc", async () => {
    vi.mocked(apiJsonTo).mockImplementation(
      (target, _path) =>
        Promise.resolve(
          (target as ApiTarget).baseUrl === local.target.baseUrl
            ? {
                entries: [
                  entry("newest local", "2026-07-08T18:00:00"),
                  entry("old local", "2026-07-06T10:00:00"),
                ],
              }
            : { entries: [entry("mid remote", "2026-07-07T12:00:00")] },
        ) as never,
    );
    const res = await fetchHistoryAll([local, hal], "");
    expect(res.entries.map((e) => e.prompt)).toEqual(["newest local", "mid remote", "old local"]);
    expect(res.entries.map((e) => e.hostId)).toEqual(["local", "hal9000-7680", "local"]);
    expect(res.entries.map((e) => e.hostLabel)).toEqual(["This Mac", "hal9000", "This Mac"]);
    expect(res.supportedHostIds).toEqual(["local", "hal9000-7680"]);
  });

  it("passes the query and a limit to every host", async () => {
    vi.mocked(apiJsonTo).mockResolvedValue({ entries: [] } as never);
    await fetchHistoryAll([local, hal], "owl");
    expect(apiJsonTo).toHaveBeenCalledTimes(2);
    for (const call of vi.mocked(apiJsonTo).mock.calls) {
      expect(call[1]).toContain("query=owl");
      expect(call[1]).toContain("limit=100");
    }
  });

  it("silently skips a host that fails (404/503 on older servers)", async () => {
    vi.mocked(apiJsonTo).mockImplementation((target, _path) =>
      (target as ApiTarget).baseUrl === hal.target.baseUrl
        ? Promise.reject(new Error("HISTORY_UNAVAILABLE"))
        : (Promise.resolve({ entries: [entry("kept", "2026-07-08T18:00:00")] }) as never),
    );
    const res = await fetchHistoryAll([local, hal], "");
    expect(res.entries.map((e) => e.prompt)).toEqual(["kept"]);
    expect(res.supportedHostIds).toEqual(["local"]);
  });

  it("is empty with no supported hosts when every host fails", async () => {
    vi.mocked(apiJsonTo).mockRejectedValue(new Error("HISTORY_UNAVAILABLE"));
    const res = await fetchHistoryAll([local, hal], "");
    expect(res.entries).toEqual([]);
    expect(res.supportedHostIds).toEqual([]);
  });

  it("handles an empty host list", async () => {
    const res = await fetchHistoryAll([], "");
    expect(res.entries).toEqual([]);
    expect(res.supportedHostIds).toEqual([]);
    expect(apiJsonTo).not.toHaveBeenCalled();
  });
});

// ── Clear scope ──────────────────────────────────────────────────────────────

describe("clearScope", () => {
  it("targets every host when the filter is All", () => {
    expect(clearScope("all", [local, hal])).toEqual([local, hal]);
  });

  it("targets only the filtered host", () => {
    expect(clearScope("hal9000-7680", [local, hal])).toEqual([hal]);
  });

  it("targets nothing when the filtered host is not in the list", () => {
    expect(clearScope("gone-7680", [local, hal])).toEqual([]);
  });
});
