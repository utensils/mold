import {
  groupLogicalGalleryPrints,
  type GalleryPrintIdentityInput,
} from "./galleryPrintIdentity";
import type { LibraryUnreadLedger } from "./libraryUnreadLedger";

/** Preserve the maintained mobile shell's existing per-host timestamp cutoffs.
 * Only the first observation of each legacy host migrates; later media is read
 * individually, so a Library visit can no longer advance an entire cutoff. */
export function observeTimestampViewingHistory<
  T extends GalleryPrintIdentityInput & { hostId: string },
>(
  ledger: LibraryUnreadLedger,
  prints: readonly T[],
  loadedHosts: readonly string[],
  cutoffs: Readonly<Record<string, number>>,
): string[][] {
  const migrationHosts = new Set(
    prints
      .map((print) => print.hostId)
      .filter((host) => !ledger.known.has(host) && cutoffs[host] != null),
  );
  ledger.observe([], [...migrationHosts]);
  const groups = groupLogicalGalleryPrints(prints);
  const keys = groups.map((group) =>
    group.copies.map((print) => `${print.hostId}|${print.filename}`),
  );
  ledger.observe(keys, loadedHosts);
  groups.forEach((group, index) => {
    if (
      group.copies.some(
        (print) =>
          migrationHosts.has(print.hostId) &&
          print.timestamp <= cutoffs[print.hostId]!,
      )
    )
      ledger.view(keys[index]!);
  });
  return keys;
}
