<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import { useCatalog } from "../composables/useCatalog";
import CatalogCard from "./CatalogCard.vue";

const cat = useCatalog();
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

function pullCard(id: string) {
  void cat.startDownload(id);
}
</script>

<template>
  <div class="min-w-0 flex-1">
    <!-- Loading state -->
    <div
      v-if="cat.loading.value"
      class="flex items-center justify-center py-12 text-sm text-ink-3"
    >
      Loading…
    </div>

    <!-- Error state -->
    <div
      v-else-if="cat.errorMsg.value"
      class="bg-bench border border-edge flex items-start gap-3 rounded-2xl px-4 py-3 text-sm text-rose-200"
    >
      <span class="mt-0.5">⚠</span>
      <div>
        <p class="font-medium text-rose-100">Couldn't load the catalog.</p>
        <p class="text-rose-200/80">{{ cat.errorMsg.value }}</p>
      </div>
    </div>

    <!-- Empty state -->
    <div
      v-else-if="cat.visibleEntries.value.length === 0"
      class="flex flex-col items-center justify-center gap-2 py-16 text-ink-3"
    >
      <p class="text-sm">No models found.</p>
      <p class="text-xs">
        No catalog entry matches every filter you've set — try widening one.
      </p>
    </div>

    <!-- Grid -->
    <template v-else>
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
          @pull="pullCard(entry.id)"
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
