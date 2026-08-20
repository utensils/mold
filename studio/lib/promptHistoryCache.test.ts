import { describe, expect, it } from "vitest";
import {
  PROMPT_HISTORY_CACHE_KEY,
  PROMPT_HISTORY_CACHE_CODE_UNIT_BUDGET,
  PROMPT_HISTORY_MAX_UTF8_BYTES,
  PROMPT_HISTORY_PER_HOST_LIMIT,
  PromptHistoryCoordinator,
  availablePromptHistoryStorage,
  promptHistoryHostSignature,
  readPromptHistoryCache,
  recordPromptHistoryCache,
  reconcilePromptHistoryCache,
  type CachedPromptHistoryEntry,
} from "./promptHistoryCache";

class MemoryStorage {
  values = new Map<string, string>();
  getItem(key: string) {
    return this.values.get(key) ?? null;
  }
  setItem(key: string, value: string) {
    this.values.set(key, value);
  }
}

const hosts = [
  { hostId: "local", hostLabel: "This device" },
  { hostId: "remote", hostLabel: "Studio" },
];
const row = (
  hostId: string,
  prompt: string,
  used_at: number,
): CachedPromptHistoryEntry => ({
  hostId,
  hostLabel: hostId,
  prompt,
  model: "flux",
  used_at,
});

describe("prompt history cache", () => {
  it("merges every host into one newest-first timeline", () => {
    const storage = new MemoryStorage();
    const merged = reconcilePromptHistoryCache(
      storage,
      hosts,
      [
        row("local", "middle", 20),
        row("remote", "newest", 30),
        row("local", "oldest", 10),
      ],
      ["local", "remote"],
    );
    expect(merged.map((entry) => entry.prompt)).toEqual([
      "newest",
      "middle",
      "oldest",
    ]);
  });

  it("retains an offline host while replacing each successful host slice", () => {
    const storage = new MemoryStorage();
    reconcilePromptHistoryCache(
      storage,
      hosts,
      [row("local", "old local", 10), row("remote", "cached remote", 20)],
      ["local", "remote"],
    );
    const merged = reconcilePromptHistoryCache(
      storage,
      [hosts[0]!, { hostId: "remote", hostLabel: "Renamed studio" }],
      [row("local", "fresh local", 30)],
      ["local"],
    );
    expect(merged.map((entry) => entry.prompt)).toEqual([
      "fresh local",
      "cached remote",
    ]);
    expect(merged.find((entry) => entry.hostId === "remote")?.hostLabel).toBe(
      "Renamed studio",
    );
  });

  it("treats a successful empty response as authoritative after Clear", () => {
    const storage = new MemoryStorage();
    reconcilePromptHistoryCache(
      storage,
      hosts,
      [row("remote", "gone", 20)],
      ["remote"],
    );
    expect(reconcilePromptHistoryCache(storage, hosts, [], ["remote"])).toEqual(
      [],
    );
  });

  it("drops forgotten hosts, corrupt rows, and excess per-host entries", () => {
    const storage = new MemoryStorage();
    storage.values.set(PROMPT_HISTORY_CACHE_KEY, "not json");
    expect(readPromptHistoryCache(storage)).toEqual([]);

    const many = Array.from(
      { length: PROMPT_HISTORY_PER_HOST_LIMIT + 5 },
      (_, i) => row("local", `prompt ${i}`, i),
    );
    reconcilePromptHistoryCache(
      storage,
      hosts,
      [...many, row("remote", "remote", 500)],
      ["local", "remote"],
    );
    const onlyLocal = reconcilePromptHistoryCache(storage, [hosts[0]!], [], []);
    expect(onlyLocal).toHaveLength(PROMPT_HISTORY_PER_HOST_LIMIT);
    expect(onlyLocal.some((entry) => entry.hostId === "remote")).toBe(false);
  });

  it("still returns live history when storage reads or writes throw", () => {
    const storage = {
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {
        throw new Error("full");
      },
    };
    expect(
      reconcilePromptHistoryCache(
        storage,
        hosts,
        [row("local", "live", 1)],
        ["local"],
      )[0]?.prompt,
    ).toBe("live");
  });

  it("survives a throwing localStorage property getter", async () => {
    const denied = Object.defineProperty({}, "localStorage", {
      get() {
        throw new DOMException("denied", "SecurityError");
      },
    });
    const storage = availablePromptHistoryStorage(denied);
    expect(storage).toBeNull();
    const live = await new PromptHistoryCoordinator().load(
      storage,
      [{ ...hosts[0]!, fetchable: true, source: "local" }],
      async () => [
        { prompt: "live without storage", model: "flux", used_at: 1 },
      ],
    );
    expect(live?.map((entry) => entry.prompt)).toEqual([
      "live without storage",
    ]);
  });

  it("accepts the server's exact 77,000-byte prompt limit", () => {
    const storage = new MemoryStorage();
    const prompt = "x".repeat(PROMPT_HISTORY_MAX_UTF8_BYTES);
    expect(
      reconcilePromptHistoryCache(
        storage,
        hosts,
        [row("local", prompt, 1)],
        ["local"],
      )[0]?.prompt,
    ).toBe(prompt);
    expect(
      reconcilePromptHistoryCache(
        storage,
        hosts,
        [row("local", `${prompt}x`, 2)],
        ["local"],
      ),
    ).toEqual([]);
  });

  it("bounds the serialized cache while keeping the full live timeline", () => {
    const storage = new MemoryStorage();
    const large = "x".repeat(PROMPT_HISTORY_MAX_UTF8_BYTES);
    const live = Array.from({ length: 100 }, (_, i) =>
      row("local", `${String(i).padStart(3, "0")}${large.slice(3)}`, i),
    );
    const merged = reconcilePromptHistoryCache(storage, hosts, live, ["local"]);
    const persisted = storage.getItem(PROMPT_HISTORY_CACHE_KEY)!;
    expect(merged).toHaveLength(100);
    expect(persisted.length).toBeLessThanOrEqual(
      PROMPT_HISTORY_CACHE_CODE_UNIT_BUDGET,
    );
    expect(readPromptHistoryCache(storage).length).toBeLessThan(100);
    expect(readPromptHistoryCache(storage)[0]?.used_at).toBe(99);
  });

  it("changes signature when an offline host is forgotten or renamed", () => {
    const offline = [
      {
        id: "local",
        label: "This device",
        status: "error",
        baseUrl: "http://local",
      },
      {
        id: "remote",
        label: "Studio",
        status: "error",
        baseUrl: "http://studio",
      },
    ];
    expect(promptHistoryHostSignature(offline)).not.toBe(
      promptHistoryHostSignature(offline.slice(0, 1)),
    );
    expect(promptHistoryHostSignature(offline)).not.toBe(
      promptHistoryHostSignature([
        { ...offline[0]! },
        { ...offline[1]!, label: "Renamed" },
      ]),
    );
  });
});

describe("PromptHistoryCoordinator", () => {
  it("persists an accepted prompt before settlement for offline reload recall", async () => {
    const storage = new MemoryStorage();
    recordPromptHistoryCache(storage, hosts, "remote", {
      prompt: "accepted and still running",
      model: "flux",
      used_at: 50,
    });

    // New coordinator = reloaded view; neither host can be fetched now.
    const reloaded = await new PromptHistoryCoordinator().load(
      storage,
      hosts.map((host) => ({ ...host, fetchable: false, source: host.hostId })),
      async () => {
        throw new Error("must not fetch offline hosts");
      },
    );
    expect(reloaded?.map((entry) => entry.prompt)).toEqual([
      "accepted and still running",
    ]);
  });

  it("fans out exact authenticated targets and merges successful hosts chronologically", async () => {
    const storage = new MemoryStorage();
    const coordinator = new PromptHistoryCoordinator();
    const seen: Array<{ baseUrl: string; apiKey: string | null }> = [];
    const result = await coordinator.load(
      storage,
      [
        {
          hostId: "local",
          hostLabel: "This device",
          fetchable: true,
          source: { baseUrl: "http://local", apiKey: "local-secret" },
        },
        {
          hostId: "remote",
          hostLabel: "Studio",
          fetchable: true,
          source: { baseUrl: "http://studio", apiKey: "remote-secret" },
        },
      ],
      async (source) => {
        seen.push(source);
        return source.baseUrl.endsWith("studio")
          ? [{ prompt: "newest", model: "flux", used_at: 30 }]
          : [{ prompt: "oldest", model: "flux", used_at: 10 }];
      },
    );
    expect(seen).toEqual([
      { baseUrl: "http://local", apiKey: "local-secret" },
      { baseUrl: "http://studio", apiKey: "remote-secret" },
    ]);
    expect(result?.map((entry) => entry.prompt)).toEqual(["newest", "oldest"]);
    expect(storage.getItem(PROMPT_HISTORY_CACHE_KEY)).not.toContain("secret");
    expect(storage.getItem(PROMPT_HISTORY_CACHE_KEY)).not.toContain("http://");
  });

  it("retains failed offline slices, clears successful empty slices, and evicts forgotten hosts", async () => {
    const storage = new MemoryStorage();
    const coordinator = new PromptHistoryCoordinator();
    reconcilePromptHistoryCache(
      storage,
      hosts,
      [row("local", "clear me", 10), row("remote", "offline", 20)],
      ["local", "remote"],
    );
    const scoped = await coordinator.load(
      storage,
      [
        { ...hosts[0]!, fetchable: true, source: "local" },
        { ...hosts[1]!, fetchable: true, source: "remote" },
      ],
      async (source) => {
        if (source === "remote") throw new Error("offline");
        return [];
      },
    );
    expect(scoped?.map((entry) => entry.prompt)).toEqual(["offline"]);

    const forgotten = await coordinator.load(
      storage,
      [{ ...hosts[0]!, fetchable: false, source: "local" }],
      async () => [],
    );
    expect(forgotten).toEqual([]);
    expect(readPromptHistoryCache(storage)).toEqual([]);
  });

  it("discards a stale async load that resolves after a newer registry transition", async () => {
    const coordinator = new PromptHistoryCoordinator();
    const storage = new MemoryStorage();
    let release!: (
      entries: Array<{ prompt: string; model: string; used_at: number }>,
    ) => void;
    const slow = coordinator.load(
      storage,
      [{ ...hosts[0]!, fetchable: true, source: "slow" }],
      () => new Promise((resolve) => (release = resolve)),
    );
    const fresh = await coordinator.load(
      storage,
      [{ ...hosts[0]!, fetchable: true, source: "fresh" }],
      async () => [{ prompt: "fresh", model: "flux", used_at: 2 }],
    );
    release([{ prompt: "stale", model: "flux", used_at: 1 }]);
    expect(await slow).toBeNull();
    expect(fresh?.map((entry) => entry.prompt)).toEqual(["fresh"]);
    expect(
      readPromptHistoryCache(storage).map((entry) => entry.prompt),
    ).toEqual(["fresh"]);
  });
});
