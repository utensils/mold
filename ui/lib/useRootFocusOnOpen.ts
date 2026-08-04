import { nextTick, onMounted, watch, type Ref } from "vue";

/** Focus an overlay root after it is initially rendered or reopened. */
export function useRootFocusOnOpen(
  root: Ref<HTMLElement | null>,
  open: () => boolean,
) {
  async function focusRoot() {
    await nextTick();
    root.value?.focus();
  }

  onMounted(() => {
    if (open()) void focusRoot();
  });
  watch(open, (isOpen) => {
    if (isOpen) void focusRoot();
  });
}
