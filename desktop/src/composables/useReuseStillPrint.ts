import {
  inspectLegacyVideoSettings,
  legacyAutomaticRequest,
} from "@studio/api/legacyVideoSettings";
import { captureChainOwner } from "@studio/api/chainJobs";
import { pendingLegacyVideoSettings } from "@studio/lib/videoLibraryExtensionHandoff";
import { useAppPrefsStore } from "../stores/appPrefs";
import { videoSettingsForReuse } from "@studio/lib/videoSettingsReuse";
import {
  retainedSourceMediaDisclosable,
  retainedSourceMediaDisclosure,
  retainedSourceMediaInventory,
} from "@studio/api/gallerySourceMedia";
import { useComposerStore, type RetainedSourceReuseHandoff } from "../stores/composer";
import { useGalleryStore, type MergedPrint } from "../stores/gallery";
import { useToastStore } from "../stores/toasts";

/**
 * "Use these settings again" for a print: the full metadata prefill plus
 * the print's own retained-source authority, asked of the producing host.
 *
 * The Lightbox and the Create view's Recent tab promise the same thing, so
 * they run the same routine — the only per-surface part is what happens after
 * (the Lightbox navigates; Recent is already on the canvas).
 */
export function useReuseStillPrint() {
  const composer = useComposerStore();
  const gallery = useGalleryStore();
  const toasts = useToastStore();

  let epoch = 0;
  return async function reuseStillPrint(entry: MergedPrint) {
    const attempt = ++epoch;
    let legacy: Awaited<ReturnType<typeof inspectLegacyVideoSettings>> | null = null;
    try {
      if (entry.item.metadata.chain && !entry.item.metadata.video_job) {
        const target = gallery.targetOfOrNull(entry.sourceKey);
        if (!target)
          throw new Error("Reconnect the video's original machine before restoring its clips.");
        const owner = await captureChainOwner({
          hostId: entry.sourceKey,
          label: entry.hostLabel,
          instanceId: "",
          target,
        });
        legacy = await inspectLegacyVideoSettings(
          owner,
          entry.item.filename,
          entry.item.metadata,
          () => attempt === epoch,
        );
        if (attempt !== epoch) return false;
        await useAppPrefsStore().update({
          generateTargetHost: entry.sourceKey === "local" ? null : entry.sourceKey,
        });
        if (attempt !== epoch) return false;
      }
      videoSettingsForReuse(entry.item.metadata);
    } catch (error) {
      toasts.push(error instanceof Error ? error.message : String(error), "error");
      return false;
    }
    pendingLegacyVideoSettings.value = null;
    // The bucket's authority may be unresolved (this device before its engine
    // answers); the recipe still restores, and the canvas says the media
    // cannot load rather than the whole reuse refusing.
    const target = gallery.targetOfOrNull(entry.sourceKey);
    // The settings AND the picture they made: the prefill names the print so
    // the canvas shows it once the recipe has landed in the form.
    const restoreMetadata = legacy
      ? {
          ...entry.item.metadata,
          seed: legacy.request.seed!,
          frames: legacy.request.stages[0]!.frames,
          prompt: legacy.request.stages[0]!.prompt,
        }
      : entry.item.metadata;
    const retainedVersion = composer.beginRetainedSourceReuse({
      ...(legacy?.automaticDuration
        ? {
            request: {
              ...legacyAutomaticRequest(legacy),
              title: entry.item.title ?? entry.item.metadata.title ?? "",
              ...(entry.item.metadata.tags ? { tags: entry.item.metadata.tags } : {}),
              ...(entry.item.metadata.collection
                ? { collection: { name: entry.item.metadata.collection } }
                : {}),
            },
          }
        : { metadata: restoreMetadata }),
      print: {
        filename: entry.item.filename,
        metadata: entry.item.metadata,
        hostId: entry.sourceKey === "local" ? null : entry.sourceKey,
        hostLabel: entry.hostLabel,
        target,
        settledAtMs: entry.item.timestamp * 1000,
      },
    });
    if (legacy) {
      pendingLegacyVideoSettings.value = legacy;
      composer.declineRetainedSource(retainedVersion);
      return true;
    }
    // The merged Library prefers a local output mirror, but mirroring the
    // output does not copy the producing host's private source-media archive.
    // Ask every known copy before disclosing an unavailable inventory, and
    // retain the exact filename and host that actually own the source bytes.
    const locations = [
      { sourceKey: entry.sourceKey, filename: entry.item.filename },
      ...gallery.locationsOf(entry),
    ].filter(
      (location, index, all) =>
        all.findIndex(
          (other) => other.sourceKey === location.sourceKey && other.filename === location.filename,
        ) === index,
    );
    void (async () => {
      let unavailable: RetainedSourceReuseHandoff | null = null;
      for (const location of locations) {
        if (!composer.isRetainedSourceCurrent(retainedVersion)) return;
        const origin = gallery.targetOfOrNull(location.sourceKey);
        if (!origin) continue;
        try {
          const inventory = await retainedSourceMediaInventory(location.filename, origin);
          if (!composer.isRetainedSourceCurrent(retainedVersion)) return;
          const handoff = {
            filename: location.filename,
            origin,
            inventory,
            metadata: entry.item.metadata,
          };
          if (inventory.availability === "available") {
            composer.setRetainedSourceIfCurrent(retainedVersion, handoff);
            return;
          }
          // Prefer a concrete archive/auth failure over a mirror's lack of
          // private pins when no reachable copy can restore the media.
          if (!unavailable || unavailable.inventory.availability === "unavailable_legacy") {
            unavailable = handoff;
          }
        } catch {
          // One unreachable copy must not hide a reachable source archive.
          // The established local stash/gallery-name restore stays live.
        }
      }
      if (!unavailable) {
        const first = locations[0];
        const origin = first && gallery.targetOfOrNull(first.sourceKey);
        if (origin)
          unavailable = {
            filename: first.filename,
            origin,
            inventory: { availability: "unavailable_missing_or_corrupt", members: [] },
            metadata: entry.item.metadata,
          };
      }
      if (!unavailable) {
        // No copy could be asked at all; Create must not wait for an answer.
        composer.declineRetainedSource(retainedVersion);
        return;
      }
      if (!composer.setRetainedSourceIfCurrent(retainedVersion, unavailable)) return;
      // Only the missing key is this door's to say. An unretained archive is
      // not the end of the story: Create falls back to the local stash and the
      // gallery by name, and names whatever it still cannot find.
      const disclosure =
        unavailable.inventory.availability === "unavailable_auth" &&
        retainedSourceMediaDisclosable(entry.item.metadata)
          ? retainedSourceMediaDisclosure(unavailable.inventory.availability)
          : null;
      if (disclosure) toasts.push(disclosure, "error");
    })();
    return true;
  };
}
