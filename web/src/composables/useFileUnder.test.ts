import { beforeEach, describe, expect, it, vi } from "vitest";
import { addTag, pickCollection } from "@studio/lib/fileUnder";
import { useFileUnder } from "./useFileUnder";
import { addHost, ORIGIN_HOST_ID } from "../lib/hostRegistry";
import { autoTagTitle, reloadAutoTagTitle } from "../lib/fileUnder";
import type { OutputMetadata } from "../types";

const hostCapabilitiesMock = vi.hoisted(() =>
  vi.fn(async (_host: { id: string }): Promise<Record<string, unknown>> => ({
    gallery: { organize: true },
  })),
);

vi.mock("../components/machines/hostClient", () => ({
  hostCapabilities: hostCapabilitiesMock,
  hostGallery: vi.fn(async () => []),
  hostApiTarget: (host: { url: string }) => ({ baseUrl: host.url }),
}));

const listCollectionsMock = vi.hoisted(() =>
  vi.fn(async (): Promise<import("../types").Collection[]> => [
    {
      id: "c1",
      name: "Smurfs",
      slug: "smurfs",
      description: null,
      cover_filename: null,
      count: 12,
      created_at: 1,
      updated_at: 1,
    },
  ]),
);
const listTagsMock = vi.hoisted(() =>
  vi.fn(async () => [
    { name: "blue", count: 9 },
    { name: "dusk", count: 2 },
  ]),
);

vi.mock("@studio/api/galleryOrganization", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@studio/api/galleryOrganization")>()),
  listCollections: listCollectionsMock,
  listTags: listTagsMock,
}));

function controller(
  title = "Smurfs",
  targetHostId: string | null = ORIGIN_HOST_ID,
) {
  return useFileUnder({
    title: () => title,
    targetHostId: () => targetHostId,
  });
}

describe("useFileUnder", () => {
  beforeEach(() => {
    localStorage.clear();
    reloadAutoTagTitle();
    hostCapabilitiesMock.mockClear();
    hostCapabilitiesMock.mockResolvedValue({ gallery: { organize: true } });
    listCollectionsMock.mockClear();
    listTagsMock.mockClear();
  });

  it("stays unavailable until a host has positively reported organize", () => {
    const fileUnder = controller();
    expect(fileUnder.available.value).toBe(false);
    expect(fileUnder.requestFields()).toEqual({});
  });

  it("becomes available once the pinned host reports gallery.organize", async () => {
    const fileUnder = controller();
    await fileUnder.refresh();
    expect(fileUnder.available.value).toBe(true);
  });

  it("stays hidden when the pinned host cannot organize", async () => {
    hostCapabilitiesMock.mockResolvedValue({ gallery: { can_delete: true } });
    const fileUnder = controller();
    await fileUnder.refresh();
    expect(fileUnder.available.value).toBe(false);
  });

  it("stays hidden when the capability probe failed outright", async () => {
    hostCapabilitiesMock.mockRejectedValue(new Error("offline"));
    const fileUnder = controller();
    await fileUnder.refresh();
    expect(fileUnder.available.value).toBe(false);
  });

  it("offers the group under automatic routing when any machine can file", async () => {
    const plato = addHost({ name: "plato", url: "http://plato:7680" });
    hostCapabilitiesMock.mockImplementation(async (host: { id: string }) =>
      host.id === plato.id ? { gallery: { organize: true } } : {},
    );
    const fileUnder = controller("Smurfs", null);
    await fileUnder.refresh();
    expect(fileUnder.available.value).toBe(true);
    // …and a pinned machine that cannot file still hides it.
    const pinned = useFileUnder({
      title: () => "Smurfs",
      targetHostId: () => ORIGIN_HOST_ID,
    });
    await pinned.refresh();
    expect(pinned.available.value).toBe(false);
  });

  it("merges the fleet's tags and collections for the pickers", async () => {
    const fileUnder = controller();
    await fileUnder.refresh();
    expect(fileUnder.suggestions.value.map((tag) => tag.name)).toEqual([
      "blue",
      "dusk",
    ]);
    expect(fileUnder.collections.value.map((c) => c.slug)).toEqual(["smurfs"]);
  });

  it("builds the ghost tag and the title match into the request fields", async () => {
    const fileUnder = controller();
    await fileUnder.refresh();
    fileUnder.state.value = addTag(fileUnder.state.value, "blue");
    expect(fileUnder.requestFields()).toEqual({
      tags: ["smurfs", "blue"],
      collection: { name: "Smurfs" },
    });
  });

  it("drops the ghost tag when the preference is off", async () => {
    const fileUnder = controller();
    await fileUnder.refresh();
    autoTagTitle.value = false;
    fileUnder.state.value = addTag(fileUnder.state.value, "blue");
    expect(fileUnder.requestFields()).toEqual({
      tags: ["blue"],
      collection: { name: "Smurfs" },
    });
  });

  it("sends a picked collection by name, never by host-local id", async () => {
    const fileUnder = controller();
    await fileUnder.refresh();
    fileUnder.state.value = pickCollection(fileUnder.state.value, {
      id: "c1",
      name: "River studies",
    });
    expect(fileUnder.requestFields().collection).toEqual({
      name: "River studies",
    });
  });

  it("resets to a fresh draft", async () => {
    const fileUnder = controller();
    await fileUnder.refresh();
    fileUnder.state.value = addTag(fileUnder.state.value, "blue");
    fileUnder.reset();
    expect(fileUnder.requestFields()).toEqual({
      tags: ["smurfs"],
      collection: { name: "Smurfs" },
    });
  });

  it("restores the filing a print actually landed with", async () => {
    const fileUnder = controller();
    await fileUnder.refresh();
    fileUnder.restoreFromMetadata({
      title: "Smurfs",
      tags: ["smurfs", "blue"],
      collection: "River studies",
    } as OutputMetadata);
    expect(fileUnder.requestFields()).toEqual({
      tags: ["smurfs", "blue"],
      collection: { name: "River studies" },
    });
  });

  it("keeps the ghost retired for a filed print that opted out of it", async () => {
    const fileUnder = controller();
    await fileUnder.refresh();
    fileUnder.restoreFromMetadata({
      title: "Smurfs",
      tags: ["blue"],
    } as OutputMetadata);
    expect(fileUnder.requestFields().tags).toEqual(["blue"]);
  });

  it("leaves an unfiled print's defaults free to apply", async () => {
    const fileUnder = controller();
    await fileUnder.refresh();
    fileUnder.restoreFromMetadata({ title: "Smurfs" } as OutputMetadata);
    expect(fileUnder.requestFields()).toEqual({
      tags: ["smurfs"],
      collection: { name: "Smurfs" },
    });
  });
});

it("filters hidden collection members from Recent and reacts to visibility changes", async () => {
  const fileUnder = controller();
  const collection = {
    id: "c1",
    name: "Smurfs",
    slug: "smurfs",
    hidden: true,
    description: null,
    cover_filename: null,
    count: 1,
    created_at: 1,
    updated_at: 1,
  };
  listCollectionsMock.mockResolvedValue([collection]);
  const privatePrint = {
    filename: "private.png",
    timestamp: 1,
    metadata: { collection: "Smurfs" },
    collections: ["c1"],
  } as unknown as import("../types").GalleryImage;
  const publicPrint = {
    ...privatePrint,
    filename: "public.png",
    metadata: privatePrint.metadata,
    collections: [],
  } as import("../types").GalleryImage;
  await fileUnder.refreshRecentVisibility();
  expect(fileUnder.visibleRecent([privatePrint, publicPrint])).toEqual([
    publicPrint,
  ]);
  listCollectionsMock.mockResolvedValue([{ ...collection, hidden: false }]);
  await fileUnder.refreshRecentVisibility();
  expect(fileUnder.visibleRecent([privatePrint, publicPrint])).toEqual([
    privatePrint,
    publicPrint,
  ]);
});

it("keeps hidden visibility on failed refresh and withholds unresolved memberships", async () => {
  const fileUnder = controller();
  const print = {
    filename: "private.png",
    timestamp: 1,
    metadata: {},
    collections: ["private"],
  } as unknown as import("../types").GalleryImage;
  listCollectionsMock.mockRejectedValueOnce(new Error("offline"));
  await fileUnder.refreshRecentVisibility();
  expect(fileUnder.visibleRecent([print])).toEqual([]);
  listCollectionsMock.mockResolvedValueOnce([
    {
      id: "private",
      slug: "private",
      name: "Private",
      hidden: true,
      description: null,
      cover_filename: null,
      count: 1,
      created_at: 1,
      updated_at: 1,
    },
  ]);
  await fileUnder.refreshRecentVisibility();
  expect(fileUnder.visibleRecent([print])).toEqual([]);
  listCollectionsMock.mockRejectedValueOnce(new Error("offline"));
  await fileUnder.refreshRecentVisibility();
  expect(fileUnder.visibleRecent([print])).toEqual([]);
});
