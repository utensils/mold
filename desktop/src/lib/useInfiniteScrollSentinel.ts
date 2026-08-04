import { onBeforeUnmount, ref, watch, type Ref } from "vue";

/**
 * Loads more results while an end-of-list sentinel is visible. A bounded
 * follow-up loop handles pages that do not move the sentinel because every
 * returned row was filtered out or deduplicated.
 */
export function useInfiniteScrollSentinel(
  sentinel: Ref<HTMLElement | null>,
  loading: Ref<boolean>,
  hasMore: Ref<boolean>,
  loadMore: () => void,
  maxChainedFetches: number,
) {
  const sentinelVisible = ref(false);
  let chainedFetches = 0;
  let observer: IntersectionObserver | null = null;

  watch(sentinel, (element) => {
    observer?.disconnect();
    observer = null;
    sentinelVisible.value = false;
    if (!element) return;
    observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) sentinelVisible.value = entry.isIntersecting;
        if (sentinelVisible.value) {
          chainedFetches = 0;
          loadMore();
        }
      },
      { rootMargin: "600px 0px" },
    );
    observer.observe(element);
  });

  watch(loading, (isLoading) => {
    if (isLoading || !sentinelVisible.value || !hasMore.value) return;
    if (chainedFetches >= maxChainedFetches) return;
    chainedFetches += 1;
    loadMore();
  });

  onBeforeUnmount(() => observer?.disconnect());
}
