import { onScopeDispose, ref, watch, type Ref } from "vue";
import {
  restoreRetainedDraftMedia,
  type RetainedDraftLayout,
} from "@studio/lib/retainedDraftMedia";
import {
  automatedReuseApply,
  clearRetainedSourceReuseIntent,
  isAutomatedReuseApply,
  retainedSourceReuseState,
  retainedSourceReuseSnapshot,
  retainedSourceReuseIsCurrent,
  setRetainedSourceReuseIntentIfCurrent,
} from "../lib/retainedSourceReuse";

/** Materializes one Library handover after the model contract is known.
 * Owns cancellation and status; the composer owns its typed form and canvas. */
export function useRetainedSourceRestore(options: {
  loaded: Ref<boolean>;
  read: () => object;
  layout: () => RetainedDraftLayout;
  apply: (patch: Record<string, unknown>) => void;
  onError: (message: string) => void;
  /** How many inputs the draft holds. A failed restore blocks until
   * deliberate recovery, and attaching an input by hand is that recovery. */
  attachedMedia?: () => number;
}) {
  const error = ref<string | null>(null);
  const pending = ref(false);
  let disposed = false;
  let materializingVersion: number | null = null;
  const abort = new AbortController();
  onScopeDispose(() => {
    disposed = true;
    abort.abort();
  });
  watch(
    [retainedSourceReuseState, options.loaded],
    async ([intent, loaded]) => {
      if (!intent) {
        error.value = null;
        pending.value = false;
        return;
      }
      // Nothing retained is not a refusal: the Library door restores an
      // unpinned print from the local stash and the gallery by name and names
      // what it cannot find. Blocking here kept Generate disabled after that
      // fallback had already filled the well.
      if (intent.inventory.availability !== "available") {
        error.value = null;
        pending.value = false;
        return;
      }
      if (!loaded) {
        pending.value = true;
        return;
      }
      const snapshot = retainedSourceReuseSnapshot();
      if (!snapshot || materializingVersion === snapshot.version) return;
      materializingVersion = snapshot.version;
      const isCurrent = () =>
        !disposed && retainedSourceReuseIsCurrent(snapshot.version);
      error.value = null;
      pending.value = true;
      try {
        const result = await restoreRetainedDraftMedia({
          ...intent,
          read: options.read,
          layout: options.layout(),
          signal: abort.signal,
          isCurrent,
        });
        if (!result || !isCurrent()) return;
        automatedReuseApply(() => options.apply(result.patch));
        setRetainedSourceReuseIntentIfCurrent(snapshot.version, {
          ...intent,
          inventory: result.inventory,
        });
      } catch (cause) {
        if (isCurrent()) {
          error.value =
            cause instanceof Error
              ? cause.message
              : "The original inputs could not be restored. Reuse this print again or reattach them.";
          options.onError(error.value);
        }
      } finally {
        if (isCurrent()) pending.value = false;
      }
    },
    { immediate: true },
  );
  if (options.attachedMedia)
    watch(
      options.attachedMedia,
      (count, previous) => {
        // A hand attach after a failure gives up the archive with the error,
        // so no archived role (a mask paired with the old source) is relayed
        // beside the input the person chose instead.
        if (
          count > previous &&
          error.value !== null &&
          !isAutomatedReuseApply()
        ) {
          error.value = null;
          clearRetainedSourceReuseIntent();
        }
      },
      // Synchronous, so the automated marker is still set when it reads it.
      { flush: "sync" },
    );
  return { error, pending };
}
