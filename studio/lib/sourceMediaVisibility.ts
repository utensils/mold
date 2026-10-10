import { onMounted, onBeforeUnmount, watch } from "vue";
import {
  sourceOwnerIdentity,
  type SourceMediaOwner,
} from "./sourceMediaBrowser";
import { invalidateSourceOwner } from "./sourceMediaResources";
/** Keep private caches fenced even while the source browser itself is closed. */
export function useSourceMediaVisibility(owners: () => SourceMediaOwner[]) {
  let previous = new Set<string>();
  onMounted(() =>
    watch(
      () => owners().map(sourceOwnerIdentity),
      (keys) => {
        const current = new Set(keys);
        for (const key of previous)
          if (!current.has(key)) invalidateSourceOwner(key);
        previous = current;
      },
      { immediate: true, flush: "sync" },
    ),
  );
  onBeforeUnmount(() => {
    for (const key of previous) invalidateSourceOwner(key);
  });
}
