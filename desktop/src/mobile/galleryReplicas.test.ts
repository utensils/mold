import { IDBFactory, IDBObjectStore } from "fake-indexeddb";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { loadCachedGallery, loadCachedGalleryReplica } from "./galleryCache";
import {
  clearMobileGalleryHosts,
  hydratedMobileGalleryReplica,
  pullMobileGalleryReplica,
} from "./galleryReplicas";

const row = (filename: string, timestamp: number, extra: Record<string, unknown> = {}) => ({
  filename,
  timestamp,
  metadata: { prompt: "", model: "m", seed: 1, steps: 1, guidance: 1, width: 8, height: 8 },
  ...extra,
});

let answers: unknown[] = [];
let urls: string[] = [];

beforeEach(() => {
  Object.defineProperty(globalThis, "indexedDB", { configurable: true, value: new IDBFactory() });
  answers = [];
  urls = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      urls.push(String(input));
      const next = answers.shift();
      return new Response(JSON.stringify(next ?? {}), {
        status: next === undefined ? 500 : 200,
        headers: { "content-type": "application/json" },
      });
    }),
  );
});
afterEach(() => vi.unstubAllGlobals());

const feedCaps = { gallery: { can_delete: true, changes_feed: { protocol_version: 1 } } } as never;
function route(cacheKey: string, capabilities: unknown = feedCaps) {
  return {
    cacheKey,
    target: { baseUrl: "https://workstation.example", apiKey: null },
    capabilities: () => capabilities as never,
    signal: new AbortController().signal,
  };
}
const feedPage = (changes: unknown[], cursor: string) => ({
  protocol_version: 1,
  instance_id: "inst",
  cursor,
  more: false,
  reset: false,
  changes,
});

describe("phone gallery replicas", () => {
  it("pulls the feed, persists rows + cursor, and resumes from IndexedDB", async () => {
    answers.push(
      feedPage(
        [
          { filename: "a.png", removed: false, image: row("a.png", 1) },
          { filename: "t.png", removed: false, image: row("t.png", 0, { trashed_at: 4 }) },
        ],
        "c1",
      ),
    );
    const replica = await pullMobileGalleryReplica(route("inst-a"));
    expect(urls).toEqual(["https://workstation.example/api/gallery/changes?limit=500"]);
    expect(replica.live().map((p) => p.filename)).toEqual(["a.png"]);
    expect((await loadCachedGallery("inst-a")).map((p) => p.filename)).toEqual(["a.png"]);
    expect((await loadCachedGalleryReplica("inst-a"))?.cursor).toBe("c1");

    // The next pull resumes from that cursor.
    answers.push(feedPage([], "c1"));
    await pullMobileGalleryReplica(route("inst-a"));
    expect(urls[1]).toContain("since=c1");
  });

  it("never rewrites the record for an idle pull", async () => {
    answers.push(feedPage([{ filename: "a.png", removed: false, image: row("a.png", 1) }], "c1"));
    await pullMobileGalleryReplica(route("inst-b"));
    const put = vi.spyOn(IDBObjectStore.prototype, "put");
    answers.push(feedPage([], "c1"));
    await pullMobileGalleryReplica(route("inst-b"));
    expect(put).not.toHaveBeenCalled();
    // …while a change does write (the spy is not vacuous).
    answers.push(feedPage([{ filename: "b.png", removed: false, image: row("b.png", 2) }], "c2"));
    await pullMobileGalleryReplica(route("inst-b"));
    expect(put).toHaveBeenCalledTimes(1);
    put.mockRestore();
  });

  it("keeps the plain listing for a host without the feed and paints saved rows first", async () => {
    answers.push([row("x.png", 3)]);
    await pullMobileGalleryReplica(route("inst-c", { gallery: { can_delete: true } }));
    expect(urls).toEqual(["https://workstation.example/api/gallery"]);
    const hydrated = await hydratedMobileGalleryReplica(route("inst-c"));
    expect(hydrated.live().map((p) => p.filename)).toEqual(["x.png"]);
  });

  it("forgets a cleared host's replica and record", async () => {
    answers.push(feedPage([{ filename: "a.png", removed: false, image: row("a.png", 1) }], "c1"));
    await pullMobileGalleryReplica(route("inst-d"));
    await clearMobileGalleryHosts(["inst-d"]);
    expect(await loadCachedGalleryReplica("inst-d")).toBeNull();
    const fresh = await hydratedMobileGalleryReplica(route("inst-d"));
    expect(fresh.size).toBe(0);
  });
});
