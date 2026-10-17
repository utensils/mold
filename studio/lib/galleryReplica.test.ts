import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  GalleryFeedUnavailableError,
  GalleryReplica,
  galleryRowsEqual,
  type GalleryChangesPage,
  type GalleryReplicaPersistence,
  type GalleryReplicaTransport,
  type PersistedGalleryReplica,
} from "./galleryReplica";
import { parseGalleryChangesPage } from "../api/galleryChanges";

interface Row {
  filename: string;
  timestamp: number;
  trashed_at?: number | null;
  purge_at?: number | null;
  title?: string | null;
  tags?: string[];
  favorite?: boolean;
  media_version?: string | null;
}

const row = (
  filename: string,
  timestamp: number,
  extra: Partial<Row> = {},
): Row => ({
  filename,
  timestamp,
  ...extra,
});

function page(
  changes: GalleryChangesPage<Row>["changes"],
  cursor: string,
  extra: Partial<GalleryChangesPage<Row>> = {},
): GalleryChangesPage<Row> {
  return {
    protocol_version: 1,
    instance_id: "inst-1",
    cursor,
    more: false,
    reset: false,
    changes,
    ...extra,
  };
}
const up = (image: Row) => ({
  filename: image.filename,
  removed: false,
  image,
});
const gone = (filename: string) => ({ filename, removed: true });

/** A scripted feed: each `changes` call answers the next queued page and
 * records the `since` it was asked with. */
function feedTransport(pages: GalleryChangesPage<Row>[]) {
  const calls: Array<string | null> = [];
  const transport: GalleryReplicaTransport<Row> = {
    feedSupported: () => true,
    changes: vi.fn(async (since: string | null) => {
      calls.push(since);
      const next = pages.shift();
      if (!next) throw new Error("no page scripted");
      return next;
    }),
    legacyLibrary: vi.fn(async () => {
      throw new Error("legacy must not run");
    }),
  };
  return { transport, calls, pages };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

describe("GalleryReplica — feed", () => {
  it("assembles a paged snapshot and resumes from the final cursor", async () => {
    const { transport, calls } = feedTransport([
      page([up(row("a.png", 1)), up(row("b.png", 2))], "c2", { more: true }),
      page([up(row("c.png", 3, { trashed_at: 9 }))], "c3"),
    ]);
    const replica = new GalleryReplica(transport);
    const delta = await replica.pull();
    expect(calls).toEqual([null, "c2"]);
    expect(delta?.reset).toBe(true);
    expect(delta?.upserted.map((r) => r.filename).sort()).toEqual([
      "a.png",
      "b.png",
      "c.png",
    ]);
    expect(replica.cursor).toBe("c3");
    expect(replica.instanceId).toBe("inst-1");
    expect(replica.live().map((r) => r.filename)).toEqual(["b.png", "a.png"]);
    expect(replica.trash().map((r) => r.filename)).toEqual(["c.png"]);
    expect(replica.trashFresh).toBe(true);
  });

  it("applies a delta of upserts, trash moves and removals in O(k)", async () => {
    const { transport, calls, pages } = feedTransport([
      page(
        [up(row("a.png", 1)), up(row("b.png", 2)), up(row("c.png", 3))],
        "c1",
      ),
    ]);
    const replica = new GalleryReplica(transport);
    await replica.pull();
    pages.push(
      page(
        [
          up(row("a.png", 1, { title: "Renamed" })),
          up(row("b.png", 2, { trashed_at: 50 })),
          gone("c.png"),
          gone("never-seen.png"),
        ],
        "c2",
      ),
    );
    const delta = await replica.pull();
    expect(calls.at(-1)).toBe("c1");
    expect(delta).toMatchObject({
      reset: false,
      removed: ["c.png"],
      source: "feed",
    });
    expect(delta?.upserted.map((r) => r.filename).sort()).toEqual([
      "a.png",
      "b.png",
    ]);
    expect(replica.live().map((r) => r.filename)).toEqual(["a.png"]);
    expect(replica.get("a.png")?.title).toBe("Renamed");
    expect(replica.trash().map((r) => r.filename)).toEqual(["b.png"]);
  });

  it("answers null for an idle page and keeps the revision", async () => {
    const { transport, pages } = feedTransport([
      page([up(row("a.png", 1))], "c1"),
    ]);
    const replica = new GalleryReplica(transport);
    await replica.pull();
    const revision = replica.revision;
    const live = replica.live();
    pages.push(page([], "c1"));
    expect(await replica.pull()).toBeNull();
    // A re-sent identical row is no change either, and keeps its identity.
    const original = replica.get("a.png");
    pages.push(page([up(row("a.png", 1))], "c2"));
    expect(await replica.pull()).toBeNull();
    expect(replica.get("a.png")).toBe(original);
    expect(replica.revision).toBe(revision);
    expect(replica.live()).toBe(live);
  });

  it("reset discards rows the rebuilt snapshot does not re-send, without a flash", async () => {
    const { transport, pages } = feedTransport([
      page([up(row("stale.png", 1)), up(row("a.png", 2))], "c1"),
    ]);
    const replica = new GalleryReplica(transport);
    await replica.pull();
    const seen: number[] = [];
    replica.subscribe(() => seen.push(replica.size));
    pages.push(
      page([up(row("a.png", 2))], "f1", { reset: true, more: true }),
      page([up(row("d.png", 4))], "f2"),
    );
    const delta = await replica.pull();
    expect(delta).toMatchObject({ reset: true, removed: ["stale.png"] });
    expect(delta?.upserted.map((r) => r.filename)).toEqual(["d.png"]);
    expect([...replica.rows().keys()].sort()).toEqual(["a.png", "d.png"]);
    expect(seen).toEqual([2]);
    expect(replica.cursor).toBe("f2");
  });

  it("a different instance answering our cursor rebuilds from a snapshot", async () => {
    const { transport, calls, pages } = feedTransport([
      page([up(row("old.png", 1)), up(row("both.png", 2))], "c1"),
    ]);
    const replica = new GalleryReplica(transport);
    await replica.pull();
    pages.push(
      page([], "x9", { instance_id: "inst-2" }),
      page([up(row("both.png", 2)), up(row("new.png", 3))], "n1", {
        instance_id: "inst-2",
      }),
    );
    const delta = await replica.pull();
    expect(calls.slice(1)).toEqual(["c1", null]);
    expect(delta?.reset).toBe(true);
    expect(delta?.removed).toEqual(["old.png"]);
    expect(delta?.upserted.map((r) => r.filename)).toEqual(["new.png"]);
    expect(replica.instanceId).toBe("inst-2");
    expect(replica.cursor).toBe("n1");
  });

  it("a page that claims more without advancing fails without removals", async () => {
    const { transport, pages } = feedTransport([
      page([up(row("a.png", 1)), up(row("b.png", 2))], "c1"),
    ]);
    const replica = new GalleryReplica(transport);
    await replica.pull();
    pages.push(page([], "c1", { more: true, reset: true }));
    await expect(replica.pull()).rejects.toThrow(/did not advance/);
    expect(replica.cursor).toBeNull();
    expect(replica.size).toBe(2);
  });

  it("a failure mid-snapshot keeps applied rows and re-snapshots next time", async () => {
    const pages = [page([up(row("a.png", 1))], "s1", { more: true })];
    let fail = true;
    const calls: Array<string | null> = [];
    const transport: GalleryReplicaTransport<Row> = {
      feedSupported: () => true,
      changes: async (since) => {
        calls.push(since);
        const next = pages.shift();
        if (next) return next;
        if (fail) throw new Error("network");
        return page([up(row("a.png", 1)), up(row("b.png", 2))], "s2");
      },
      legacyLibrary: async () => [],
    };
    const replica = new GalleryReplica(transport);
    await expect(replica.pull()).rejects.toThrow("network");
    expect(replica.get("a.png")).toBeDefined();
    expect(replica.cursor).toBeNull();
    fail = false;
    await replica.pull();
    expect(calls).toEqual([null, "s1", null]);
    expect(replica.size).toBe(2);
  });
});

describe("GalleryReplica — single flight", () => {
  it("coalesces every request during a pull into exactly one follow-up", async () => {
    const gates: Array<ReturnType<typeof deferred<GalleryChangesPage<Row>>>> =
      [];
    const transport: GalleryReplicaTransport<Row> = {
      feedSupported: () => true,
      changes: vi.fn(() => {
        const gate = deferred<GalleryChangesPage<Row>>();
        gates.push(gate);
        return gate.promise;
      }),
      legacyLibrary: async () => [],
    };
    const replica = new GalleryReplica(transport);
    const first = replica.pull();
    await vi.waitFor(() => expect(gates).toHaveLength(1));
    const second = replica.pull();
    const third = replica.pull();
    replica.tickle();
    expect(second).toBe(third);
    gates[0]!.resolve(page([up(row("a.png", 1))], "c1"));
    await first;
    await vi.waitFor(() => expect(gates).toHaveLength(2));
    gates[1]!.resolve(page([up(row("b.png", 2))], "c2"));
    const followUp = await second;
    expect(followUp?.upserted.map((r) => r.filename)).toEqual(["b.png"]);
    expect(transport.changes).toHaveBeenCalledTimes(2);
    // Idle again: the next pull starts fresh.
    const fourth = replica.pull();
    await vi.waitFor(() => expect(gates).toHaveLength(3));
    gates[2]!.resolve(page([], "c2"));
    expect(await fourth).toBeNull();
  });

  it("join() rides the pull in flight instead of queueing another", async () => {
    const gate = deferred<GalleryChangesPage<Row>>();
    const transport: GalleryReplicaTransport<Row> = {
      feedSupported: () => true,
      changes: vi.fn(() => gate.promise),
      legacyLibrary: async () => [],
    };
    const replica = new GalleryReplica(transport);
    const first = replica.pull();
    const joined = replica.join();
    gate.resolve(page([up(row("a.png", 1))], "c1"));
    await Promise.all([first, joined]);
    expect(transport.changes).toHaveBeenCalledTimes(1);
  });

  it("notifies subscribers once per changing pull", async () => {
    const { transport, pages } = feedTransport([
      page([up(row("a.png", 1))], "c1"),
    ]);
    const replica = new GalleryReplica(transport);
    const listener = vi.fn();
    const stop = replica.subscribe(listener);
    await replica.pull();
    pages.push(page([], "c1"));
    await replica.pull();
    expect(listener).toHaveBeenCalledTimes(1);
    stop();
  });
});

describe("GalleryReplica — legacy", () => {
  function legacyTransport(library: Row[], trash: Row[] | null) {
    const state = { library, trash };
    const transport: GalleryReplicaTransport<Row> = {
      feedSupported: () => false,
      changes: vi.fn(async () => {
        throw new Error("feed must not run");
      }),
      legacyLibrary: vi.fn(async () => state.library),
      legacyTrash: vi.fn(async () => state.trash),
    };
    return { transport, state };
  }

  it("diffs listings into the same delta shape, and a 304 array is free", async () => {
    const lib = [row("a.png", 1, { media_version: "1" }), row("b.png", 2)];
    const { transport, state } = legacyTransport(lib, [
      row("t.png", 0, { trashed_at: 5 }),
    ]);
    const replica = new GalleryReplica(transport);
    const first = await replica.pull();
    expect(first).toMatchObject({ source: "legacy", reset: false });
    expect(first?.upserted).toHaveLength(3);
    expect(replica.mode).toBe("legacy");
    expect(replica.trashFresh).toBe(true);
    // 304: the conditional fetch hands back the very same arrays.
    expect(await replica.pull()).toBeNull();
    state.library = [row("a.png", 1, { media_version: "2" }), row("c.png", 3)];
    state.trash = [];
    const delta = await replica.pull();
    expect(delta?.upserted.map((r) => r.filename).sort()).toEqual([
      "a.png",
      "c.png",
    ]);
    expect(delta?.removed.sort()).toEqual(["b.png", "t.png"]);
  });

  it("a failed trash listing keeps the library and the held trash", async () => {
    const { transport, state } = legacyTransport(
      [row("a.png", 1)],
      [row("t.png", 0, { trashed_at: 5 })],
    );
    const replica = new GalleryReplica(transport);
    await replica.pull();
    transport.legacyTrash = async () => {
      throw new Error("trash down");
    };
    state.library = [row("a.png", 1), row("b.png", 2)];
    const delta = await replica.pull();
    expect(delta?.upserted.map((r) => r.filename)).toEqual(["b.png"]);
    expect(replica.trash()).toHaveLength(1);
    expect(replica.trashFresh).toBe(false);
  });

  it("partitions by trash-listing provenance, not only trashed_at", async () => {
    // Older servers (and rows organized before trash stamping) list a print
    // under ?view=trash without `trashed_at`.
    const unstamped = row("t.png", 4);
    const { transport, state } = legacyTransport(
      [row("a.png", 1)],
      [unstamped],
    );
    const replica = new GalleryReplica(transport);
    await replica.pull();
    expect(replica.live().map((r) => r.filename)).toEqual(["a.png"]);
    expect(replica.trash().map((r) => r.filename)).toEqual(["t.png"]);
    expect(replica.isTrashed("t.png")).toBe(true);
    // A failed trash listing preserves it in the trash — never the Library.
    transport.legacyTrash = async () => {
      throw new Error("trash down");
    };
    state.library = [row("a.png", 1), row("b.png", 2)];
    const delta = await replica.pull();
    expect(delta?.removed).toEqual([]);
    expect(replica.live().map((r) => r.filename)).toEqual(["b.png", "a.png"]);
    expect(replica.trash().map((r) => r.filename)).toEqual(["t.png"]);
  });

  it("a membership flip with identical content is still a change", async () => {
    const same = row("x.png", 3);
    const { transport, state } = legacyTransport([same], []);
    const replica = new GalleryReplica(transport);
    await replica.pull();
    state.library = [];
    state.trash = [row("x.png", 3)];
    const delta = await replica.pull();
    expect(delta?.upserted.map((r) => r.filename)).toEqual(["x.png"]);
    expect(replica.live()).toEqual([]);
    expect(replica.trash().map((r) => r.filename)).toEqual(["x.png"]);
  });

  it("a feed row clears a name the trash listing put in the trash", async () => {
    let feed = false;
    const transport: GalleryReplicaTransport<Row> = {
      feedSupported: () => feed,
      changes: async () => page([up(row("t.png", 4))], "c1"),
      legacyLibrary: async () => [],
      legacyTrash: async () => [row("t.png", 4)],
    };
    const replica = new GalleryReplica(transport);
    await replica.pull();
    expect(replica.trash()).toHaveLength(1);
    feed = true;
    const delta = await replica.pull();
    expect(delta?.upserted.map((r) => r.filename)).toEqual(["t.png"]);
    expect(replica.live().map((r) => r.filename)).toEqual(["t.png"]);
    expect(replica.isTrashed("t.png")).toBe(false);
  });

  it("an unlisted trash is preserved, never reported removed", async () => {
    const { transport, state } = legacyTransport(
      [row("a.png", 1)],
      [row("t.png", 0, { trashed_at: 5 })],
    );
    const replica = new GalleryReplica(transport);
    await replica.pull();
    state.trash = null;
    state.library = [row("a.png", 1), row("b.png", 2)];
    const delta = await replica.pull();
    expect(delta?.removed).toEqual([]);
    expect(replica.trash().map((r) => r.filename)).toEqual(["t.png"]);
    expect(replica.trashFresh).toBe(false);
  });

  it("falls back to the listing when the feed is not ready, then snapshots", async () => {
    let ready = false;
    const transport: GalleryReplicaTransport<Row> = {
      feedSupported: () => true,
      changes: vi.fn(async (since) => {
        if (!ready) throw new GalleryFeedUnavailableError("not_ready");
        expect(since).toBeNull();
        return page([up(row("a.png", 1))], "c1");
      }),
      legacyLibrary: vi.fn(async () => [row("a.png", 1)]),
    };
    const replica = new GalleryReplica(transport);
    expect((await replica.pull())?.source).toBe("legacy");
    ready = true;
    // The snapshot re-sends the same row: no visible change at all.
    expect(await replica.pull()).toBeNull();
    expect(replica.mode).toBe("feed");
    expect(replica.cursor).toBe("c1");
  });

  it("feed and legacy end in the same state for the same gallery", async () => {
    const library = [row("a.png", 3, { tags: ["x"] }), row("b.png", 2)];
    const trash = [row("t.png", 1, { trashed_at: 7, purge_at: 99 })];
    const legacy = new GalleryReplica(
      legacyTransport(library, trash).transport,
    );
    await legacy.pull();
    const feed = new GalleryReplica(
      feedTransport([
        page([up(trash[0]!), up(library[1]!)], "p1", { more: true }),
        page([up(library[0]!)], "p2"),
      ]).transport,
    );
    await feed.pull();
    expect(feed.live()).toEqual(legacy.live());
    expect(feed.trash()).toEqual(legacy.trash());
  });

  it("rejects a pull when the listing itself fails", async () => {
    const transport: GalleryReplicaTransport<Row> = {
      feedSupported: () => false,
      changes: async () => page([], "x"),
      legacyLibrary: async () => {
        throw new Error("offline");
      },
    };
    await expect(new GalleryReplica(transport).pull()).rejects.toThrow(
      "offline",
    );
  });
});

describe("GalleryReplica — persistence", () => {
  function memoryStore(initial: PersistedGalleryReplica<Row> | null = null) {
    let saved = initial;
    const store: GalleryReplicaPersistence<Row> & { saves: number } = {
      saves: 0,
      load: async () => saved,
      save: async (state) => {
        store.saves += 1;
        saved = JSON.parse(
          JSON.stringify(state),
        ) as PersistedGalleryReplica<Row>;
      },
    };
    return { store, read: () => saved };
  }

  it("round-trips cursor and rows, and writes only when something changed", async () => {
    const { store, read } = memoryStore();
    const first = feedTransport([page([up(row("a.png", 1))], "c1")]);
    const writer = new GalleryReplica(first.transport, { persistence: store });
    await writer.pull();
    expect(read()).toEqual({
      version: 1,
      instanceId: "inst-1",
      cursor: "c1",
      rows: [row("a.png", 1)],
    });
    first.pages.push(page([], "c1"));
    await writer.pull();
    expect(store.saves).toBe(1);

    const second = feedTransport([page([up(row("b.png", 2))], "c2")]);
    const reader = new GalleryReplica(second.transport, { persistence: store });
    const hydrated = await reader.hydrate();
    expect(hydrated).toMatchObject({ source: "hydrate", reset: true });
    expect(reader.live().map((r) => r.filename)).toEqual(["a.png"]);
    await reader.pull();
    expect(second.calls).toEqual(["c1"]);
    expect(reader.live().map((r) => r.filename)).toEqual(["b.png", "a.png"]);
    expect(read()?.cursor).toBe("c2");
  });

  it("a failed save never fails the pull", async () => {
    const replica = new GalleryReplica(
      feedTransport([page([up(row("a.png", 1))], "c1")]).transport,
      {
        persistence: {
          load: async () => null,
          save: async () => {
            throw new Error("quota");
          },
        },
      },
    );
    await expect(replica.pull()).resolves.not.toBeNull();
  });
});

describe("galleryRowsEqual", () => {
  it("treats undefined and absent keys alike and compares deeply", () => {
    expect(galleryRowsEqual({ a: 1, b: undefined }, { a: 1 })).toBe(true);
    expect(galleryRowsEqual({ a: [1, { b: 2 }] }, { a: [1, { b: 2 }] })).toBe(
      true,
    );
    expect(galleryRowsEqual({ a: [1, 2] }, { a: [2, 1] })).toBe(false);
    expect(galleryRowsEqual({ a: null }, { a: undefined })).toBe(false);
  });
});

// Shared golden fixtures (docs/contracts/gallery-changes-v1.json) — the same
// scenarios the Rust and Swift replicas replay.
/* Found by walking up from the working directory: `import.meta.url` is not
 * a file URL under happy-dom, and both studio's and desktop's runners
 * collect this file. */
function findRepoFile(relative: string): string | null {
  let directory = process.cwd();
  for (;;) {
    const candidate = resolve(directory, relative);
    if (existsSync(candidate)) return candidate;
    const parent = dirname(directory);
    if (parent === directory) return null;
    directory = parent;
  }
}
const fixturePath = findRepoFile("docs/contracts/gallery-changes-v1.json");
interface FixtureScenario {
  name: string;
  initial: string[];
  since: string | null;
  pages: GalleryChangesPage<Row>[];
  expected: Record<string, Partial<Row>>;
  final_cursor: string;
}
const fixtures: { scenarios: FixtureScenario[] } | null = fixturePath
  ? (JSON.parse(readFileSync(fixturePath, "utf8")) as {
      scenarios: FixtureScenario[];
    })
  : null;

describe.skipIf(fixtures === null)(
  "gallery-changes-v1 contract fixtures",
  () => {
    for (const scenario of fixtures?.scenarios ?? []) {
      it(scenario.name, async () => {
        const pages = [...scenario.pages];
        const calls: Array<string | null> = [];
        const transport: GalleryReplicaTransport<Row> = {
          feedSupported: () => true,
          changes: async (since) => {
            calls.push(since);
            const next = pages.shift();
            if (!next) throw new Error("fixture exhausted");
            // Through the wire parser, exactly as the HTTP transport reads it.
            return parseGalleryChangesPage<Row>(next);
          },
          legacyLibrary: async () => [],
        };
        const replica = new GalleryReplica(transport, {
          persistence: {
            load: async () => ({
              version: 1,
              instanceId: scenario.pages[0]?.instance_id ?? null,
              cursor: scenario.since,
              rows: scenario.initial.map((filename) => row(filename, 0)),
            }),
            save: async () => undefined,
          },
        });
        await replica.pull();
        expect(calls[0]).toBe(scenario.since);
        expect(pages).toHaveLength(0);
        expect([...replica.rows().keys()].sort()).toEqual(
          Object.keys(scenario.expected).sort(),
        );
        for (const [filename, fields] of Object.entries(scenario.expected))
          expect(replica.get(filename)).toMatchObject(fields);
        expect(replica.cursor).toBe(scenario.final_cursor);
      });
    }
  },
);
