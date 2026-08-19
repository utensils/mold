import { describe, expect, it } from "vitest";
import {
  GALLERY_ORGANIZATION_EVENT_TYPES,
  isGalleryOrganizationEvent,
  type GalleryOrganizationEvent,
} from "./galleryOrganization";

describe("gallery organization SSE envelopes", () => {
  it("names exactly the four additive event types", () => {
    expect([...GALLERY_ORGANIZATION_EVENT_TYPES]).toEqual([
      "gallery_updated",
      "gallery_trashed",
      "gallery_restored",
      "gallery_collections_changed",
    ]);
  });

  it("recognises every variant and narrows the union", () => {
    const events: unknown[] = [
      { type: "gallery_updated", filename: "a.png", image: null },
      {
        type: "gallery_updated",
        filename: "a.png",
        image: { filename: "a.png", title: "T", favorite: true },
      },
      { type: "gallery_trashed", filename: "a.png" },
      { type: "gallery_restored", filename: "a.png", image: null },
      { type: "gallery_collections_changed" },
    ];
    for (const event of events) {
      expect(isGalleryOrganizationEvent(event)).toBe(true);
    }
    const narrowed = events.filter(isGalleryOrganizationEvent);
    const types = narrowed.map((event: GalleryOrganizationEvent) => event.type);
    expect(types).toHaveLength(5);
  });

  it("rejects other server events, missing filenames, and junk", () => {
    expect(
      isGalleryOrganizationEvent({ type: "gallery_added", filename: "a" }),
    ).toBe(false);
    expect(
      isGalleryOrganizationEvent({ type: "gallery_removed", filename: "a" }),
    ).toBe(false);
    expect(isGalleryOrganizationEvent({ type: "gallery_updated" })).toBe(false);
    expect(
      isGalleryOrganizationEvent({ type: "gallery_trashed", filename: 3 }),
    ).toBe(false);
    expect(isGalleryOrganizationEvent(null)).toBe(false);
    expect(isGalleryOrganizationEvent("gallery_updated")).toBe(false);
    expect(isGalleryOrganizationEvent(undefined)).toBe(false);
  });
});
