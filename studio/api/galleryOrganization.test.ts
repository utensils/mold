import { afterEach, describe, expect, it, vi } from "vitest";
import type { ApiTarget } from "./client";
import { ApiError } from "./client";
import {
  createCollection,
  deleteCollection,
  deleteGalleryImageForever,
  deleteTag,
  emptyTrash,
  listCollections,
  listTags,
  listTrash,
  organizeGallery,
  patchGalleryImage,
  renameTag,
  restoreTrashed,
  setCollectionItems,
  sweepTrash,
  trashGalleryImage,
  trashMany,
  updateCollection,
  updateCollectionHidden,
} from "./galleryOrganization";

const target: ApiTarget = { baseUrl: "http://plato:7680", apiKey: "secret" };

interface Captured {
  url: string;
  method: string;
  headers: Headers;
  body: unknown;
}

function stub(
  respond: () => Response = () => new Response(null, { status: 204 }),
): () => Captured {
  let captured: Captured | null = null;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const headers = new Headers(init?.headers);
      captured = {
        url: String(input),
        method: init?.method ?? "GET",
        headers,
        body:
          typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
      };
      return respond();
    }),
  );
  return () => {
    if (!captured) throw new Error("fetch was not called");
    return captured;
  };
}

afterEach(() => vi.unstubAllGlobals());

describe("gallery organization API", () => {
  it("PATCHes one print through the explicit target with the API key header", async () => {
    const captured = stub(() =>
      Response.json({ filename: "a b.png", title: "New", favorite: true }),
    );
    const image = await patchGalleryImage(target, "a b.png", {
      title: "New",
      favorite: true,
      add_tags: ["x"],
    });
    const call = captured();
    expect(call.url).toBe("http://plato:7680/api/gallery/image/a%20b.png");
    expect(call.method).toBe("PATCH");
    expect(call.headers.get("x-api-key")).toBe("secret");
    expect(call.headers.get("content-type")).toBe("application/json");
    expect(call.body).toEqual({
      title: "New",
      favorite: true,
      add_tags: ["x"],
    });
    expect(image).toEqual({
      filename: "a b.png",
      title: "New",
      favorite: true,
    });
  });

  it("returns null when the PATCH answers with no body", async () => {
    stub();
    expect(await patchGalleryImage(target, "a.png", { title: "" })).toBeNull();
  });

  it("POSTs a bulk organize body verbatim", async () => {
    const captured = stub();
    await organizeGallery(target, {
      filenames: ["a.png", "b.png"],
      favorite: true,
      add_tags: ["t"],
      remove_tags: [],
      add_to_collections: ["c1"],
      remove_from_collections: ["c2"],
    });
    const call = captured();
    expect(call.url).toBe("http://plato:7680/api/gallery/organize");
    expect(call.method).toBe("POST");
    expect(call.body).toEqual({
      filenames: ["a.png", "b.png"],
      favorite: true,
      add_tags: ["t"],
      remove_tags: [],
      add_to_collections: ["c1"],
      remove_from_collections: ["c2"],
    });
  });

  it("lists collections from either a bare array or an enveloped list", async () => {
    const row = {
      id: "c1",
      name: "Studio",
      slug: "studio",
      description: null,
      cover_filename: null,
      count: 2,
      created_at: 1,
      updated_at: 2,
    };
    let captured = stub(() => Response.json([row]));
    expect(await listCollections(target)).toEqual([row]);
    expect(captured().url).toBe("http://plato:7680/api/gallery/collections");
    expect(captured().method).toBe("GET");

    captured = stub(() => Response.json({ collections: [row] }));
    expect(await listCollections(target)).toEqual([row]);
  });

  it("rejects a collections listing it cannot read", async () => {
    stub(() => Response.json({ nope: true }));
    await expect(listCollections(target)).rejects.toThrow(/collections/);
  });

  it("creates, updates, and deletes a collection on the right paths", async () => {
    let captured = stub(() => Response.json({ id: "c1" }));
    await createCollection(target, { name: "Studio", description: "d" });
    expect(captured()).toMatchObject({
      url: "http://plato:7680/api/gallery/collections",
      method: "POST",
      body: { name: "Studio", description: "d" },
    });

    captured = stub(() => Response.json({ id: "c/1" }));
    await updateCollection(target, "c/1", {
      name: "Renamed",
      cover_filename: "x.png",
    });
    expect(captured()).toMatchObject({
      url: "http://plato:7680/api/gallery/collections/c%2F1",
      method: "PATCH",
      body: { name: "Renamed", cover_filename: "x.png" },
    });

    captured = stub();
    await deleteCollection(target, "c1");
    expect(captured()).toMatchObject({
      url: "http://plato:7680/api/gallery/collections/c1",
      method: "DELETE",
    });
    expect(captured().headers.get("x-api-key")).toBe("secret");
  });

  it("rejects an older server that silently ignores hidden collection updates", async () => {
    stub(() => Response.json({ id: "c1", name: "Studio" }));
    await expect(updateCollectionHidden(target, "c1", true)).rejects.toThrow(
      /does not support hidden collections/,
    );
  });

  it("accepts a hidden collection update only when the echoed state matches", async () => {
    let captured = stub(() => Response.json({ id: "c1", hidden: true }));
    await expect(
      updateCollectionHidden(target, "c1", true),
    ).resolves.toMatchObject({
      hidden: true,
    });
    expect(captured().body).toEqual({ hidden: true });

    captured = stub(() => Response.json({ id: "c1", hidden: false }));
    await expect(
      updateCollectionHidden(target, "c1", false),
    ).resolves.toMatchObject({
      hidden: false,
    });
  });

  it("PUTs collection item changes and hands back the collection when returned", async () => {
    const captured = stub(() => Response.json({ id: "c1", count: 3 }));
    const result = await setCollectionItems(target, "c1", {
      add: ["a.png"],
      remove: ["b.png"],
    });
    expect(captured()).toMatchObject({
      url: "http://plato:7680/api/gallery/collections/c1/items",
      method: "PUT",
      body: { add: ["a.png"], remove: ["b.png"] },
    });
    expect(result).toEqual({ id: "c1", count: 3 });
  });

  it("lists, renames, and deletes tags", async () => {
    let captured = stub(() =>
      Response.json({ tags: [{ name: "a", count: 1 }] }),
    );
    expect(await listTags(target)).toEqual([{ name: "a", count: 1 }]);
    expect(captured().url).toBe("http://plato:7680/api/gallery/tags");

    captured = stub(() => Response.json([{ name: "a", count: 1 }]));
    expect(await listTags(target)).toEqual([{ name: "a", count: 1 }]);

    captured = stub();
    await renameTag(target, "b&w", "mono");
    expect(captured()).toMatchObject({
      url: "http://plato:7680/api/gallery/tags/b%26w",
      method: "PATCH",
      body: { name: "mono" },
    });

    captured = stub();
    await deleteTag(target, "b&w");
    expect(captured()).toMatchObject({
      url: "http://plato:7680/api/gallery/tags/b%26w",
      method: "DELETE",
    });
  });

  it("trashes through the plain DELETE and purges with ?permanent=true", async () => {
    let captured = stub();
    await trashGalleryImage(target, "a b.png");
    expect(captured()).toMatchObject({
      url: "http://plato:7680/api/gallery/image/a%20b.png",
      method: "DELETE",
    });

    captured = stub();
    await deleteGalleryImageForever(target, "a b.png");
    expect(captured()).toMatchObject({
      url: "http://plato:7680/api/gallery/image/a%20b.png?permanent=true",
      method: "DELETE",
    });
    expect(captured().headers.get("x-api-key")).toBe("secret");
  });

  it("posts bulk trash and restore filename lists", async () => {
    let captured = stub();
    await trashMany(target, ["a.png", "b.png"]);
    expect(captured()).toMatchObject({
      url: "http://plato:7680/api/gallery/trash",
      method: "POST",
      body: { filenames: ["a.png", "b.png"] },
    });

    captured = stub();
    await restoreTrashed(target, ["a.png"]);
    expect(captured()).toMatchObject({
      url: "http://plato:7680/api/gallery/trash/restore",
      method: "POST",
      body: { filenames: ["a.png"] },
    });
  });

  it("empties and sweeps the trash, returning the server's counts", async () => {
    let captured = stub(() => Response.json({ purged: 4 }));
    expect(await emptyTrash(target)).toEqual({ purged: 4 });
    expect(captured()).toMatchObject({
      url: "http://plato:7680/api/gallery/trash",
      method: "DELETE",
    });

    captured = stub(() => Response.json({ purged: 1, remaining: 2 }));
    expect(await sweepTrash(target)).toEqual({ purged: 1, remaining: 2 });
    expect(captured()).toMatchObject({
      url: "http://plato:7680/api/gallery/trash/sweep",
      method: "POST",
    });
  });

  it("lists the trash view and rejects a non-list body", async () => {
    const captured = stub(() =>
      Response.json([{ filename: "a.png", trashed_at: 1, purge_at: 2 }]),
    );
    expect(await listTrash(target)).toEqual([
      { filename: "a.png", trashed_at: 1, purge_at: 2 },
    ]);
    expect(captured()).toMatchObject({
      url: "http://plato:7680/api/gallery?view=trash",
      method: "GET",
    });

    stub(() => Response.json({ error: "nope" }));
    await expect(listTrash(target)).rejects.toThrow(/trash/);
  });

  it("surfaces HTTP failures as ApiError with the server's message", async () => {
    stub(
      () =>
        new Response(JSON.stringify({ error: "restore target exists" }), {
          status: 409,
          headers: { "content-type": "application/json" },
        }),
    );
    await expect(restoreTrashed(target, ["a.png"])).rejects.toMatchObject({
      name: "ApiError",
      status: 409,
      message: "restore target exists",
    });
    await expect(restoreTrashed(target, ["a.png"])).rejects.toBeInstanceOf(
      ApiError,
    );
  });

  it("sends no API key header when the target has none", async () => {
    const captured = stub();
    await trashGalleryImage(
      { baseUrl: "http://local:7680", apiKey: null },
      "a.png",
    );
    expect(captured().headers.has("x-api-key")).toBe(false);
  });
});
