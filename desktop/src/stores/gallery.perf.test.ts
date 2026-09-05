/**
 * Library store performance guards. Every assertion is an OPERATION COUNT
 * against a data-size budget — see `@studio/lib/galleryPerfBudget`. A
 * wall-clock number would be flaky in CI; a count is exact and names the hot
 * path that regressed.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { toRaw } from "vue";

const counters = vi.hoisted(() => ({
  unionOrganization: 0,
  sameLogicalGalleryPrint: 0,
  reset() {
    counters.unionOrganization = 0;
    counters.sameLogicalGalleryPrint = 0;
  },
}));

vi.mock("@studio/lib/libraryOrganization", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@studio/lib/libraryOrganization")>();
  return {
    ...actual,
    unionOrganization: (...args: Parameters<typeof actual.unionOrganization>) => {
      counters.unionOrganization += 1;
      return actual.unionOrganization(...args);
    },
  };
});
vi.mock("@studio/lib/galleryPrintIdentity", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@studio/lib/galleryPrintIdentity")>();
  return {
    ...actual,
    sameLogicalGalleryPrint: (...args: Parameters<typeof actual.sameLogicalGalleryPrint>) => {
      counters.sameLogicalGalleryPrint += 1;
      return actual.sameLogicalGalleryPrint(...args);
    },
  };
});
vi.mock("../lib/ipc", () => ({
  ipc: { localGalleryList: vi.fn() },
}));
vi.mock("../lib/api/client", () => ({
  conditionalApiJsonTo: vi.fn(),
  apiFetchTo: vi.fn(),
}));

import { expectOpsUnder } from "@studio/lib/galleryPerfBudget";
import { useConnectionStore } from "./connection";
import { useGalleryStore, type GalleryBucket } from "./gallery";
import { useHostsStore } from "./hosts";
import type { GalleryImage, ServerCapabilities } from "../lib/api/types";

const LOCAL = 2_000;
const REMOTE_SHARED = 1_000; // remote rows mirrored locally under the same filename
const REMOTE_ONLY = 1_000;
const LEGACY_COPIES = 200; // same seed+size+model, different filename, inside the window

function row(filename: string, index: number, overrides: Partial<GalleryImage> = {}): GalleryImage {
  return {
    filename,
    timestamp: 1_800_000_000 - index * 10,
    size_bytes: 100_000 + index,
    format: index % 7 === 0 ? "mp4" : "png",
    favorite: index % 5 === 0,
    tags: index % 3 === 0 ? ["portrait", `batch-${index % 11}`] : [],
    collections: index % 4 === 0 ? ["c1"] : [],
    metadata: {
      prompt: `print ${index}`,
      model: "flux-dev:q8",
      seed: index,
      steps: 4,
      guidance: 1,
      width: 1024,
      height: index % 2 === 0 ? 768 : 1024,
    },
    ...overrides,
  } as GalleryImage;
}

const bucket = (items: GalleryImage[]): GalleryBucket => ({
  items,
  loading: false,
  error: null,
  loaded: true,
  authorityTarget: null,
  authorityResolved: false,
});

function seedFleet() {
  const conn = useConnectionStore();
  conn.info = { mode: "local", baseUrl: "http://127.0.0.1:49152", apiKey: "k" };
  conn.status = "ready";
  const hosts = useHostsStore();
  hosts.extras.push({
    id: "plato-7680",
    label: "plato",
    url: "http://plato:7680",
    apiKey: "pk",
    status: "ready",
    error: null,
    instanceId: null,
  });
  const caps = {
    gallery: { can_delete: true, organize: true, trash: { enabled: true, retention_days: 30 } },
  } as unknown as ServerCapabilities;
  hosts.capabilities["local"] = caps;
  hosts.capabilities["plato-7680"] = caps;

  const local: GalleryImage[] = [];
  for (let i = 0; i < LOCAL; i++) local.push(row(`local-${i}.png`, i));
  const remote: GalleryImage[] = [];
  for (let i = 0; i < REMOTE_SHARED; i++) remote.push(row(`local-${i}.png`, i));
  for (let i = 0; i < REMOTE_ONLY; i++) remote.push(row(`remote-${i}.png`, LOCAL + i));
  // Legacy auto-saves: identical seed/size/model under a minted name.
  for (let i = 0; i < LEGACY_COPIES; i++) {
    const origin = LOCAL - 1 - i;
    remote.push(
      row(`mold-flux-legacy-${i}.png`, origin, { timestamp: local[origin]!.timestamp + 5 }),
    );
  }
  const gallery = useGalleryStore();
  gallery.buckets["local"] = bucket(local);
  gallery.buckets["plato-7680"] = bucket(remote);
  const collections = [{ id: "c1", name: "Keepers", slug: "keepers", hidden: false }];
  gallery.collectionsByHost["local"] = { items: collections, loaded: true } as never;
  gallery.collectionsByHost["plato-7680"] = { items: collections, loaded: true } as never;
  return gallery;
}

beforeEach(() => {
  setActivePinia(createPinia());
  counters.reset();
});

describe("gallery store derived work is indexed once per data change", () => {
  it("runs unionOrganization at most once per logical print across every getter", () => {
    const gallery = seedFleet();
    const logical = gallery.merged.length;
    expect(logical).toBe(LOCAL + REMOTE_ONLY);
    counters.reset();

    // Everything the Library header, chips, shelf and grid read on one change.
    void gallery.basePrints;
    void gallery.filterChipTags;
    void gallery.collectionCounts("keepers");
    void gallery.kindCounts;
    void gallery.chipCounts;
    void gallery.filtered;
    for (const entry of gallery.merged) void gallery.organizationOf(entry);
    for (const entry of gallery.merged) void gallery.organizationOf(entry);

    expectOpsUnder("unionOrganization per derived pass", counters.unionOrganization, logical);
  });

  it("re-narrowing by favorites, tags and a collection performs no new union passes", () => {
    const gallery = seedFleet();
    void gallery.filtered;
    counters.reset();
    gallery.scope = "favorites";
    void gallery.filtered;
    gallery.tagFilter = ["portrait"];
    void gallery.filtered;
    gallery.scope = "collections";
    gallery.collectionSlug = "keepers";
    void gallery.filtered;
    expectOpsUnder("unionOrganization on filter change", counters.unionOrganization, 0);
  });

  it("resolves every copy of a print without scanning the buckets", () => {
    const gallery = seedFleet();
    const entries = gallery.merged;
    // Instrument the bucket arrays themselves: every element read is
    // counted, so an inline `.find()`/`.some()` scan (which would not call
    // `sameLogicalGalleryPrint`) still shows up as N² reads.
    let elementReads = 0;
    const counted = (items: GalleryImage[]) =>
      new Proxy(items, {
        get(target, prop, receiver) {
          if (typeof prop === "string" && /^\d+$/.test(prop)) elementReads += 1;
          return Reflect.get(target, prop, receiver);
        },
      });
    for (const key of Object.keys(gallery.buckets)) {
      gallery.buckets[key]!.items = counted(toRaw(gallery.buckets[key]!.items));
    }
    counters.reset();
    let copies = 0;
    for (const entry of entries) {
      copies += gallery.allLocationsOf(entry).length;
      void gallery.locationsOf(entry);
      void gallery.trashLocationsOf(entry);
    }
    expect(copies).toBe(LOCAL + REMOTE_SHARED + REMOTE_ONLY + LEGACY_COPIES);
    expectOpsUnder("sameLogicalGalleryPrint scans", counters.sameLogicalGalleryPrint, 0);
    // Rebuilding the per-bucket indexes reads each row a bounded number of
    // times; 4 200 lookups over 4 200 rows must stay far from 17 million.
    const totalRows = LOCAL + REMOTE_SHARED + REMOTE_ONLY + LEGACY_COPIES;
    expectOpsUnder("bucket element reads during copy lookups", elementReads, 6 * totalRows);
  });

  it("returns a print's copies in bucket order, legacy twin before canonical row", () => {
    const gallery = seedFleet();
    // Plato lists a legacy-named twin ABOVE the canonical row for one print.
    const canonical = row("local-7.png", 7);
    const twin = row("mold-flux-twin.png", 7, { timestamp: canonical.timestamp + 3 });
    gallery.buckets["plato-7680"] = bucket([twin, canonical]);
    const entry = gallery.merged.find((e) => e.item.filename === "local-7.png")!;
    expect(gallery.locationsOf(entry).filter((l) => l.sourceKey === "plato-7680")).toEqual([
      { sourceKey: "plato-7680", filename: "mold-flux-twin.png" },
      { sourceKey: "plato-7680", filename: "local-7.png" },
    ]);
  });

  it("finds a legacy copy (different filename, same identity) through the index", () => {
    const gallery = seedFleet();
    const origin = gallery.merged.find((e) => e.item.filename === `local-${LOCAL - 1}.png`)!;
    expect(gallery.locationsOf(origin)).toEqual(
      expect.arrayContaining([
        { sourceKey: "local", filename: `local-${LOCAL - 1}.png` },
        { sourceKey: "plato-7680", filename: "mold-flux-legacy-0.png" },
      ]),
    );
  });

  it("answers existsLocally and rowOf without scanning", () => {
    const gallery = seedFleet();
    const remoteOnly = gallery.merged.filter((e) => e.sourceKey !== "local");
    expect(remoteOnly.length).toBe(REMOTE_ONLY);
    counters.reset();
    for (const entry of remoteOnly) expect(gallery.existsLocally(entry)).toBe(false);
    for (const entry of gallery.merged) {
      expect(gallery.rowOf(entry.sourceKey, entry.item.filename)).toBe(entry.item);
    }
    expectOpsUnder(
      "sameLogicalGalleryPrint in existsLocally/rowOf",
      counters.sameLogicalGalleryPrint,
      0,
    );
  });
});
