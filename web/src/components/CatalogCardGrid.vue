<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useCatalog } from "../composables/useCatalog";
import { useModelInstallTargets } from "../composables/useModelInstallTargets";
import { toast } from "../lib/toasts";
import type { CatalogEntryWire } from "../types";
import CatalogCard from "./CatalogCard.vue";

const cat = useCatalog();
const installTargets = useModelInstallTargets();
const sentinel = ref<HTMLElement | null>(null);
let observer: IntersectionObserver | null = null;

function detach() {
  observer?.disconnect();
  observer = null;
}

// rootMargin lets us start loading the next page ~one screenful before the
// sentinel actually scrolls into view, so the user rarely sees an empty
// stretch waiting for the fetch to complete.
function attach() {
  detach();
  if (!sentinel.value) return;
  if (typeof IntersectionObserver === "undefined") return;
  observer = new IntersectionObserver(
    (entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        void cat.loadMore();
      }
    },
    { rootMargin: "400px" },
  );
  observer.observe(sentinel.value);
}

onMounted(attach);
onBeforeUnmount(detach);
// Re-attach when the sentinel ref changes (e.g. transitions from
// hasMore=false → true after a filter change loads a fresh result set).
watch(sentinel, attach);

function openCard(id: string) {
  void cat.openDetail(id);
}

// A model can be installed here and missing on the next machine, so the pull
// resolves a target first: one candidate acts straight away, several ask.
async function pullCard(entry: CatalogEntryWire) {
  const choice = await installTargets.chooseInstallTarget({
    modelId: entry.id,
    displayName: entry.name,
    ownedByOrigin: entry.installed,
  });
  if (choice.kind === "cancelled") return;
  try {
    await installTargets.startDownloadOn(choice.target, entry.id);
    toast("success", installTargets.queuedMessage(choice.target));
  } catch (error) {
    toast("error", error instanceof Error ? error.message : String(error));
  }
}
</script>

<template>
  <div class="min-w-0 flex-1">
    <!-- Loading state -->
    <div
      v-if="cat.loading.value && cat.visibleEntries.value.length === 0"
      class="flex items-center justify-center py-12 text-sm text-ink-3"
    >
      Loading…
    </div>

    <!-- Error state -->
    <div
      v-if="cat.errorMsg.value || cat.providerErrors.value.length"
      data-test="catalog-provider-warning"
      class="bg-bench border border-edge flex items-start gap-3 rounded-2xl px-4 py-3 text-sm text-rose-200"
      role="alert"
    >
      <span class="mt-0.5">⚠</span>
      <div class="min-w-0 flex-1">
        <p class="font-medium text-rose-100">
          {{
            cat.errorMsg.value
              ? "Couldn't refresh the catalog."
              : "Part of the catalog is unavailable."
          }}
        </p>
        <p class="text-rose-200/80">
          {{
            cat.errorMsg.value ??
            cat.providerErrors.value.map((item) => item.message).join(" ")
          }}
          <span v-if="cat.providerErrors.value.length">
            Showing available models.</span
          >
        </p>
      </div>
      <button
        type="button"
        data-test="catalog-retry"
        class="border border-rose-200/40 shrink-0 rounded-lg px-3 py-1.5 font-medium text-rose-100 hover:bg-rose-100/10"
        :disabled="cat.loading.value"
        @click="cat.refresh()"
      >
        {{ cat.loading.value ? "Retrying…" : "Retry" }}
      </button>
    </div>

    <!-- Empty state -->
    <div
      v-if="!cat.loading.value && cat.visibleEntries.value.length === 0"
      class="flex flex-col items-center justify-center gap-2 py-16 text-ink-3"
    >
      <p class="text-sm">No models found.</p>
      <p class="text-xs">
        No catalog entry matches every filter you've set — try widening one.
      </p>
    </div>

    <!-- Grid -->
    <template v-else-if="cat.visibleEntries.value.length > 0">
      <p
        data-testid="catalog-result-count"
        class="mb-3 text-[12px] text-ink-3"
        aria-live="polite"
      >
        {{ cat.resultCount.value.toLocaleString() }}
        {{ cat.resultCount.value === 1 ? "result" : "results" }}
      </p>

      <div
        data-test="catalog-results"
        :data-layout="cat.layout.value"
        :class="
          cat.layout.value === 'grid'
            ? 'grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3'
            : 'flex flex-col gap-2'
        "
      >
        <CatalogCard
          v-for="entry in cat.visibleEntries.value"
          :key="entry.id"
          :entry="entry"
          :layout="cat.layout.value"
          @open="openCard(entry.id)"
          @pull="pullCard(entry)"
        />
      </div>

      <div
        v-if="cat.loadingMore.value"
        class="py-6 text-center text-sm text-ink-3"
      >
        Loading more…
      </div>

      <!-- The sentinel triggers loadMore() when scrolled into view.
           We only render it while there's actually more to fetch so a
           stale observer callback can't double-fetch the final page. -->
      <div
        v-if="cat.hasMore.value"
        ref="sentinel"
        data-testid="catalog-load-more-sentinel"
        class="h-1 w-full"
        aria-hidden="true"
      />
    </template>
  </div>
</template>
