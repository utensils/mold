import {
  collectionVisibilityRevision,
  collectionHiddenAfterRevision,
  serializeCollectionVisibility,
} from "../lib/collectionVisibility";
import { apiJsonTo, type ApiTarget } from "./client";
import {
  listCollections,
  createCollection,
  updateCollection,
  updateCollectionHidden,
  patchGalleryImage,
  setCollectionItems,
} from "./galleryOrganization";
import type {
  Collection,
  GalleryOrganizationFields,
} from "../lib/api/galleryOrganization";
import {
  collectionSlug,
  normalizeTagName,
  tagKey,
} from "../lib/libraryOrganization";
export interface MirrorOrganization {
  item: GalleryOrganizationFields;
  collections: Collection[];
  visibilityRevision?: number;
}
/** Capture before transfer so a read failure cannot silently claim a complete copy. */
export async function captureMirrorOrganization(
  target: ApiTarget,
  filename: string,
): Promise<MirrorOrganization> {
  const visibilityRevision = collectionVisibilityRevision();
  const [rows, collections] = await Promise.all([
    apiJsonTo<Array<GalleryOrganizationFields & { filename: string }>>(
      target,
      "/api/gallery",
    ),
    listCollections(target),
  ]);
  const item = rows.find((r) => r.filename === filename);
  if (!item)
    throw new Error("The source print changed. Refresh before copying it.");
  return {
    item,
    visibilityRevision,
    collections: collections.filter((c) => item.collections?.includes(c.id)),
  };
}
/** Repair existing copies as well as new ones. Keep destination-only organization. */
export async function applyMirrorOrganization(
  target: ApiTarget,
  filename: string,
  snapshot: MirrorOrganization,
): Promise<void> {
  const [rows, collections] = await Promise.all([
    apiJsonTo<Array<GalleryOrganizationFields & { filename: string }>>(
      target,
      "/api/gallery",
    ),
    listCollections(target),
  ]);
  const item = rows.find((r) => r.filename === filename);
  if (!item)
    throw new Error(
      "The destination print is unavailable; organization was not synchronized.",
    );
  const tags = new Map<string, string>();
  for (const tag of [...(item.tags ?? []), ...(snapshot.item.tags ?? [])])
    tags.set(tagKey(tag), normalizeTagName(tag));
  await patchGalleryImage(target, filename, {
    ...(item.title || snapshot.item.title
      ? { title: item.title || snapshot.item.title! }
      : {}),
    favorite: item.favorite === true || snapshot.item.favorite === true,
    tags: [...tags.values()],
  });
  for (const source of snapshot.collections) {
    const slug = source.slug || collectionSlug(source.name);
    let dest = collections.find(
      (c) => (c.slug || collectionSlug(c.name)) === slug,
    );
    if (!dest) {
      dest = await createCollection(target, {
        name: source.name,
        description: source.description,
      });
      collections.push(dest);
    }
    const destination = dest;
    await serializeCollectionVisibility(async () => {
      const captured = snapshot.visibilityRevision ?? 0;
      const fallback = source.hidden === true || destination.hidden === true;
      let desired = collectionHiddenAfterRevision(slug, fallback, captured);
      while ((destination.hidden === true) !== desired) {
        Object.assign(
          destination,
          await updateCollectionHidden(target, destination.id, desired),
        );
        desired = collectionHiddenAfterRevision(slug, fallback, captured);
      }
    });
    if (!dest.description && source.description)
      await updateCollection(target, dest.id, {
        description: source.description,
      });
    await setCollectionItems(target, dest.id, { add: [filename], remove: [] });
  }
}
