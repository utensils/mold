import { afterEach, describe, expect, it, vi } from "vitest";
import {
  fetchGalleryChanges,
  galleryChangesFeedAdvertised,
  galleryChangesPath,
  parseGalleryChangesPage,
  httpGalleryReplicaTransport,
} from "./galleryChanges";
import {
  GalleryFeedUnavailableError,
  GalleryReplica,
} from "../lib/galleryReplica";

afterEach(() => vi.unstubAllGlobals());

const json = (value: unknown, init: ResponseInit = {}) =>
  new Response(JSON.stringify(value), {
    ...init,
    headers: { "content-type": "application/json", ...(init.headers ?? {}) },
  });

describe("gallery changes client", () => {
  it("reads the capability strictly", () => {
    expect(
      galleryChangesFeedAdvertised({
        gallery: { changes_feed: { protocol_version: 1 } },
      }),
    ).toBe(true);
    expect(
      galleryChangesFeedAdvertised({
        gallery: { changes_feed: { protocol_version: 2 } },
      }),
    ).toBe(false);
    expect(galleryChangesFeedAdvertised({ gallery: {} })).toBe(false);
    expect(galleryChangesFeedAdvertised(null)).toBe(false);
  });

  it("reads pages the way mold_core's serde defaults do", () => {
    // `changes` and `removed` are `#[serde(default)]` in GalleryChangesPage /
    // GalleryChange: an absent field is an empty list / false.
    const page = parseGalleryChangesPage({
      protocol_version: 1,
      instance_id: "i",
      cursor: "c",
      more: false,
      reset: true,
    });
    expect(page.changes).toEqual([]);
    const withChange = parseGalleryChangesPage<{
      filename: string;
      timestamp: number;
    }>({
      protocol_version: 1,
      instance_id: "i",
      cursor: "c",
      more: false,
      reset: false,
      changes: [
        { filename: "a.png", image: { filename: "a.png", timestamp: 1 } },
      ],
    });
    expect(withChange.changes[0]?.removed).toBe(false);
    expect(() => parseGalleryChangesPage({ protocol_version: 2 })).toThrow(
      GalleryFeedUnavailableError,
    );
  });

  it("omits since for a snapshot and encodes an opaque cursor", () => {
    expect(galleryChangesPath(null, 500)).toBe(
      "/api/gallery/changes?limit=500",
    );
    expect(galleryChangesPath("a+b/c", 10)).toBe(
      "/api/gallery/changes?since=a%2Bb%2Fc&limit=10",
    );
  });

  it("maps 503 NOT_READY and 404 to the fallback signal, with the key in headers", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        json(
          { error: "warming", code: "GALLERY_FEED_NOT_READY" },
          { status: 503 },
        ),
      )
      .mockResolvedValueOnce(
        json(
          { error: "nope", code: "GALLERY_FEED_UNSUPPORTED" },
          { status: 404 },
        ),
      )
      .mockResolvedValueOnce(json({ error: "boom" }, { status: 500 }));
    vi.stubGlobal("fetch", fetchMock);
    const target = { baseUrl: "http://feed-test:7680", apiKey: "secret" };
    await expect(fetchGalleryChanges(target, null, 500)).rejects.toMatchObject({
      reason: "not_ready",
    });
    await expect(fetchGalleryChanges(target, "c1", 500)).rejects.toMatchObject({
      reason: "unsupported",
    });
    const failure = fetchGalleryChanges(target, "c1", 500);
    await expect(failure).rejects.not.toBeInstanceOf(
      GalleryFeedUnavailableError,
    );
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://feed-test:7680/api/gallery/changes?limit=500");
    expect(url).not.toContain("secret");
    expect((init.headers as Headers).get("x-api-key")).toBe("secret");
  });

  it("drives a replica: feed when advertised, conditional listings otherwise", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/gallery/changes"))
        return json({
          protocol_version: 1,
          instance_id: "i",
          cursor: "c1",
          more: false,
          reset: false,
          changes: [
            {
              filename: "a.png",
              removed: false,
              image: { filename: "a.png", timestamp: 1 },
            },
          ],
        });
      if (url.endsWith("view=trash"))
        return json([{ filename: "t.png", timestamp: 0, trashed_at: 3 }]);
      return json([{ filename: "b.png", timestamp: 2 }]);
    });
    vi.stubGlobal("fetch", fetchMock);
    let advertised = false;
    const replica = new GalleryReplica(
      httpGalleryReplicaTransport<{
        filename: string;
        timestamp: number;
        trashed_at?: number;
      }>(
        { baseUrl: "http://drive-test:7680", apiKey: null },
        { feedSupported: () => advertised, listTrash: () => true },
      ),
    );
    await replica.pull();
    expect(replica.live().map((r) => r.filename)).toEqual(["b.png"]);
    expect(replica.trash().map((r) => r.filename)).toEqual(["t.png"]);
    advertised = true;
    await replica.pull();
    expect([...replica.rows().keys()]).toEqual(["a.png"]);
    expect(replica.mode).toBe("feed");
  });
});
