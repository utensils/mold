/**
 * Where a sequence renders. A sequence routes by its model — the shared
 * generation-host policy (sticky pick / Auto / Most capable) applies, with
 * one twist: CUDA-only families (LTX-2) escalate Auto to "Most capable" so
 * they land on a CUDA box instead of a Metal Mac that can't run them.
 * Extracted from the retired ChainsView so the unified Create submit path
 * keeps the exact routing behavior.
 */
import { useAppPrefsStore } from "../stores/appPrefs";
import { useHostsStore, type HostRoute } from "../stores/hosts";
import { normalizeTargetHost } from "./hosts";
import type { ModelEntry } from "./api/types";

export function routeForModel(model: Pick<ModelEntry, "name" | "family">): HostRoute | null {
  const hosts = useHostsStore();
  // The header chip's sticky pick wins; a ghost id reads as Auto so a stale
  // persisted host can never wedge every Generate click.
  const sticky = normalizeTargetHost(
    useAppPrefsStore().settings?.generateTargetHost ?? null,
    hosts.all,
  );
  const cudaOnly = model.family === "ltx2" || model.family === "ltx-2";
  return hosts.resolveRoute(sticky ?? (cudaOnly ? "capable" : null), model.name);
}
