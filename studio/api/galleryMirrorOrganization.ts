import {
  reconcileMirrorOrganization,
  recoverMirrorOrganization,
  type MirrorOrganizationState,
  type MirrorOrganizationIntent,
} from "../lib/mirrorOrganizationPolicy";
export interface MirrorReconciliation {
  prior?: MirrorOrganizationState;
  contributors?: MirrorOrganizationState[];
  deliberate?: boolean;
  pending?: MirrorOrganizationIntent;
  origin?: string;
  scalarAuthority?: boolean;
  scalarFields?: string[];
  observeOnly?: boolean;
  intent(state: MirrorOrganizationIntent): Promise<void>;
  current?(): boolean;
}
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
  item: GalleryOrganizationFields & {
    metadata_synthetic?: boolean;
    timestamp?: number;
  };
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
    apiJsonTo<Array<MirrorOrganization["item"] & { filename: string }>>(
      target,
      "/api/gallery",
    ),
    listCollections(target),
  ]);
  const item = rows.find((r) => r.filename === filename);
  if (!item)
    throw new Error("The source print changed. Refresh before copying it.");
  return {
    item: {
      ...(item.title == null ? {} : { title: item.title }),
      favorite: item.favorite === true,
      tags: item.tags ?? [],
      collections: item.collections ?? [],
      metadata_synthetic: item.metadata_synthetic ?? false,
      ...(Number.isSafeInteger(item.timestamp)
        ? { timestamp: item.timestamp! }
        : {}),
    },
    visibilityRevision,
    collections: collections.filter((c) => item.collections?.includes(c.id)),
  };
}
/** Repair existing copies as well as new ones. Keep destination-only organization. */
export async function applyMirrorOrganization(
  target: ApiTarget,
  filename: string,
  snapshot: MirrorOrganization,
  reconciliation?: MirrorReconciliation,
): Promise<MirrorOrganizationState | void> {
  const [rows, collections] = await Promise.all([
    apiJsonTo<Array<MirrorOrganization["item"] & { filename: string }>>(
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
  const check = () => {
    if (reconciliation?.current && !reconciliation.current())
      throw new Error(
        "The Library connection changed during organization synchronization.",
      );
  };
  check();
  const slugOf = (collection: Collection) =>
    collection.slug || collectionSlug(collection.name);
  const destinationSlugs = collections
    .filter((collection) => item.collections?.includes(collection.id))
    .map(slugOf);
  const destinationValues = {
    title: item.title ?? null,
    favorite: item.favorite === true,
    tags: item.tags ?? [],
    collections: destinationSlugs,
  };
  const prior = reconciliation?.pending
    ? recoverMirrorOrganization(
        reconciliation.prior,
        reconciliation.pending,
        destinationValues,
      )
    : reconciliation?.prior;
  const plan = reconciliation
    ? reconcileMirrorOrganization(
        {
          title: snapshot.item.title ?? null,
          favorite: snapshot.item.favorite === true,
          tags: snapshot.item.tags ?? [],
          collections: snapshot.collections.map(slugOf),
          hidden: snapshot.collections
            .filter((collection) => collection.hidden)
            .map(slugOf),
        },
        destinationValues,
        prior,
        reconciliation,
      )
    : null;
  if (plan) {
    await reconciliation!.intent({
      state: plan.state,
      baseline: destinationValues,
      desired: plan.desired,
    });
    check();
  }
  const tags = new Map<string, string>();
  for (const tag of [...(item.tags ?? []), ...(snapshot.item.tags ?? [])])
    tags.set(tagKey(tag), normalizeTagName(tag));
  check();
  await patchGalleryImage(
    target,
    filename,
    plan
      ? {
          title: plan.desired.title,
          favorite: plan.desired.favorite,
          tags: plan.desired.tags,
        }
      : {
          ...(item.title || snapshot.item.title
            ? { title: item.title || snapshot.item.title! }
            : {}),
          favorite: item.favorite === true || snapshot.item.favorite === true,
          tags: [...tags.values()],
        },
  );
  check();
  if (plan) {
    for (const collection of collections) {
      if (
        destinationSlugs.includes(slugOf(collection)) &&
        !plan.desired.collections.includes(slugOf(collection))
      ) {
        check();
        await setCollectionItems(target, collection.id, {
          add: [],
          remove: [filename],
        });
      }
    }
  }
  for (const source of snapshot.collections) {
    if (plan && !plan.desired.collections.includes(slugOf(source))) continue;
    check();
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
      check();
      const captured = snapshot.visibilityRevision ?? 0;
      const fallback = source.hidden === true || destination.hidden === true;
      let desired = collectionHiddenAfterRevision(slug, fallback, captured);
      while ((destination.hidden === true) !== desired) {
        check();
        Object.assign(
          destination,
          await updateCollectionHidden(target, destination.id, desired),
        );
        desired = collectionHiddenAfterRevision(slug, fallback, captured);
      }
    });
    check();
    if (!dest.description && source.description)
      await updateCollection(target, dest.id, {
        description: source.description,
      });
    check();
    await setCollectionItems(target, dest.id, { add: [filename], remove: [] });
  }
  if (plan) {
    check();
    const [confirmedRows, confirmedCollections] = await Promise.all([
      apiJsonTo<Array<MirrorOrganization["item"] & { filename: string }>>(
        target,
        "/api/gallery",
      ),
      listCollections(target),
    ]);
    check();
    const confirmed = confirmedRows.find((row) => row.filename === filename);
    const same = (left: string[], right: string[]) =>
      JSON.stringify([...left].sort()) === JSON.stringify([...right].sort());
    const hiddenConfirmed = snapshot.collections
      .filter((collection) => collection.hidden === true)
      .every((source) =>
        confirmedCollections.some(
          (collection) =>
            slugOf(collection) === slugOf(source) && collection.hidden === true,
        ),
      );
    if (
      !hiddenConfirmed ||
      !confirmed ||
      (confirmed.title ?? null) !== plan.desired.title ||
      (confirmed.favorite === true) !== plan.desired.favorite ||
      !same(
        (confirmed.tags ?? []).map(tagKey),
        plan.desired.tags.map(tagKey),
      ) ||
      !same(
        confirmedCollections
          .filter((collection) =>
            confirmed.collections?.includes(collection.id),
          )
          .map(slugOf),
        plan.desired.collections,
      )
    )
      throw new Error(
        "Destination organization changed or could not be confirmed. Copy again to repair it.",
      );
    return plan.state;
  }
}
