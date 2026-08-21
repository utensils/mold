/*
 * Create-time filing state for the web Create workspace ("File under").
 *
 * Binds the shared `@studio/lib/fileUnder` contract to web's host registry:
 * the same per-host organization snapshot the Library builds supplies the
 * capability gate, the merged tag vocabulary, and the merged collection
 * shelf, so Create and Library can never disagree about what exists.
 *
 * Nothing here creates a collection. The group only records an intent; the
 * request carries `{ name }` and the routed host get-or-creates by slug at
 * develop time, which is what makes one request file correctly anywhere in
 * the fleet.
 */
import { computed, ref, shallowRef, type Ref } from "vue";
import {
  buildFileUnderRequestFields,
  emptyFileUnderState,
  type FileUnderRequestFields,
  type FileUnderState,
} from "@studio/lib/fileUnder";
import type { TagCount } from "../types";
import {
  anyHostOrganizes,
  fetchOrganization,
  hostOrganizes,
  mergedCollections,
  mergedTags,
  type HostOrganizationSnapshot,
} from "../lib/libraryOrganization";
import { listHosts } from "../lib/hostRegistry";
import { autoTagTitle, titleTagWasApplied } from "../lib/fileUnder";
import type { OutputMetadata } from "../types";

export interface UseFileUnderOptions {
  /** The live Create title. */
  title: () => string | null | undefined;
  /**
   * The concrete machine this print is routed to, or `null` under an
   * automatic policy (Auto / Most capable), where the landing host is not
   * known until admission.
   */
  targetHostId: () => string | null;
}

export interface FileUnderController {
  /** Whether the group renders at all — positive capability knowledge only. */
  available: Ref<boolean>;
  state: Ref<FileUnderState>;
  /** Merged, count-sorted tag vocabulary for the `Add tag…` suggestions. */
  suggestions: Ref<TagCount[]>;
  /** Collections merged across hosts by slug. */
  collections: Ref<ReturnType<typeof mergedCollections>>;
  /** The additive `tags` / `collection` slice of a request. */
  requestFields: () => FileUnderRequestFields;
  /** Re-probe every registered host's organization state. */
  refresh: () => Promise<void>;
  /** ⌘N / "new print": back to a fresh draft. */
  reset: () => void;
  /** Reuse settings: restore what a print was actually filed under. */
  restoreFromMetadata: (metadata: OutputMetadata) => void;
}

export function useFileUnder(
  options: UseFileUnderOptions,
): FileUnderController {
  const snapshots = shallowRef<HostOrganizationSnapshot[]>([]);
  const state = ref<FileUnderState>(emptyFileUnderState());

  const available = computed(() => {
    const hostId = options.targetHostId();
    // A pinned machine answers for itself; under Auto / Most capable the
    // print lands on one of several, so any filing-capable machine in the
    // fleet is enough to offer the group.
    return hostId
      ? hostOrganizes(snapshots.value, hostId)
      : anyHostOrganizes(snapshots.value);
  });

  const collections = computed(() => mergedCollections(snapshots.value));
  const suggestions = computed(() => mergedTags(snapshots.value));

  async function refresh(): Promise<void> {
    snapshots.value = await fetchOrganization(listHosts());
  }

  function requestFields(): FileUnderRequestFields {
    // A host that cannot file must never be sent filing it would reject.
    if (!available.value) return {};
    return buildFileUnderRequestFields(
      state.value,
      options.title(),
      autoTagTitle.value,
      collections.value,
    );
  }

  function reset(): void {
    state.value = emptyFileUnderState();
  }

  function restoreFromMetadata(metadata: OutputMetadata): void {
    const tags = metadata.tags ?? null;
    const collection = metadata.collection?.trim() ?? "";
    const restored = emptyFileUnderState();
    if (tags) {
      restored.manualTags = [...tags];
      // The print was filed, and its title's tag is not among them — so the
      // original explicitly opted out. Restoring the ghost would silently
      // re-add a tag the user removed.
      restored.ghostRemoved = !titleTagWasApplied(metadata.title, tags);
    }
    if (collection) {
      restored.picked = { name: collection };
      restored.pickedExplicitly = true;
    }
    state.value = restored;
  }

  return {
    available,
    state,
    suggestions,
    collections,
    requestFields,
    refresh,
    reset,
    restoreFromMetadata,
  };
}
