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

  return function reuseStillPrint(entry: MergedPrint) {
    // The bucket's authority may be unresolved (this device before its engine
    // answers); the recipe still restores, and the canvas says the media
    // cannot load rather than the whole reuse refusing.
    const target = gallery.targetOfOrNull(entry.sourceKey);
    // The settings AND the picture they made: the prefill names the print so
    // the canvas shows it once the recipe has landed in the form.
    const retainedVersion = composer.beginRetainedSourceReuse({
      metadata: entry.item.metadata,
      print: {
        filename: entry.item.filename,
        metadata: entry.item.metadata,
        hostId: entry.sourceKey === "local" ? null : entry.sourceKey,
        hostLabel: entry.hostLabel,
        target,
        settledAtMs: entry.item.timestamp * 1000,
      },
    });
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
          const handoff = { filename: location.filename, origin, inventory };
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
      if (!unavailable || !composer.setRetainedSourceIfCurrent(retainedVersion, unavailable))
        return;
      const disclosure = retainedSourceMediaDisclosable(entry.item.metadata)
        ? retainedSourceMediaDisclosure(unavailable.inventory.availability)
        : null;
      if (disclosure) toasts.push(disclosure, "error");
    })();
  };
}
