import {
  retainedSourceMediaDisclosable,
  retainedSourceMediaDisclosure,
  retainedSourceMediaInventory,
} from "@studio/api/gallerySourceMedia";
import { useComposerStore } from "../stores/composer";
import { useGalleryStore, type MergedPrint } from "../stores/gallery";
import { useToastStore } from "../stores/toasts";

/**
 * "Use these settings again" for a STILL print: the full metadata prefill plus
 * the print's own retained-source authority, asked of the producing host.
 *
 * The Lightbox and the Create view's Recent tab promise the same thing, so
 * they run the same routine — the only per-surface part is what happens after
 * (the Lightbox navigates; Recent is already on the canvas). A sequence print
 * belongs to `composer.setSequence` and never reaches here.
 */
export function useReuseStillPrint() {
  const composer = useComposerStore();
  const gallery = useGalleryStore();
  const toasts = useToastStore();

  return function reuseStillPrint(entry: MergedPrint) {
    const retainedVersion = composer.beginRetainedSourceReuse({ metadata: entry.item.metadata });
    const target = gallery.targetOf(entry.sourceKey);
    if (!target) return;
    // Always ask — the host is the only authority on what it retained, and the
    // metadata under-reports inline video/audio/mask bytes. But a text-to-image
    // print's archive entry resolves with no pins, which the server can only
    // report as `unavailable_legacy`, so an UNAVAILABLE answer is toasted only
    // when the print's own metadata says conditioning bytes were shipped.
    void retainedSourceMediaInventory(entry.item.filename, target)
      .then((inventory) => {
        if (
          !composer.setRetainedSourceIfCurrent(retainedVersion, {
            filename: entry.item.filename,
            origin: target,
            inventory,
          })
        ) {
          return;
        }
        const disclosure = retainedSourceMediaDisclosable(entry.item.metadata)
          ? retainedSourceMediaDisclosure(inventory.availability)
          : null;
        if (disclosure) toasts.push(disclosure, "error");
      })
      // The established local stash/gallery-name restore stays live. A
      // transport failure inspecting the additive endpoint must not turn a
      // previously working reuse into a dead end.
      .catch(() => {});
  };
}
