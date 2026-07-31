import { useHostModelsStore } from "../stores/hostModels";
import { useModelStore } from "../stores/models";
import type { HostView } from "../stores/hosts";

/**
 * Have we actually read this machine's installed set?
 *
 * This is the `inventoryKnown` gate for `planModelInstall`. A machine whose
 * `/api/models` has not landed is neither an owner nor known to be missing a
 * model — offering to install onto it would be a guess, and labelling a row
 * from that guess makes the action flicker as snapshots arrive.
 *
 * Reactive by construction: both reads hit Pinia state, so a caller that
 * invokes the returned predicate during render re-renders when a snapshot
 * lands.
 */
export function useInventoryKnown(): (host: HostView) => boolean {
  const models = useModelStore();
  const hostModels = useHostModelsStore();
  return (host) =>
    (hostModels.byHost[host.id]?.fetchedAt ?? 0) > 0 ||
    // The primary's canonical list lives in its own store, not the per-host map.
    (host.primary && models.all.length > 0);
}
