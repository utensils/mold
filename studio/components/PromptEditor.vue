<script setup lang="ts">
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
  useSlots,
} from "vue";
import { overlayDepth, subscribeOverlayDepth } from "@ui/lib/overlayStack";
import ModalPanel from "@ui/components/ModalPanel.vue";
import type { PromptAuthoringSource } from "../lib/promptProvenance";
const props = withDefaults(
  defineProps<{
    open: boolean;
    prompt: string;
    history?: string[];
    historyLoading?: boolean;
    historyError?: string;
  }>(),
  { history: () => [] },
);
const emit = defineEmits<{
  authored: [text: string, source: PromptAuthoringSource];
  close: [];
}>();
const slots = useSlots();
const textarea = ref<HTMLTextAreaElement | null>(null);
const recent = ref(false);
const viewportHeight = ref(0);
const viewportTop = ref(0);
function measureViewport() {
  viewportHeight.value = window.visualViewport?.height ?? window.innerHeight;
  viewportTop.value = window.visualViewport?.offsetTop ?? 0;
}
onMounted(() => {
  measureViewport();
  window.visualViewport?.addEventListener("resize", measureViewport);
  window.visualViewport?.addEventListener("scroll", measureViewport);
});
onBeforeUnmount(() => {
  window.visualViewport?.removeEventListener("resize", measureViewport);
  window.visualViewport?.removeEventListener("scroll", measureViewport);
});
const search = ref("");
const cleared = ref<string | null>(null);
const results = computed(() =>
  [...new Set(props.history)].filter((text) =>
    text.toLocaleLowerCase().includes(search.value.toLocaleLowerCase()),
  ),
);
let opener: HTMLElement | null = null;
let previousOverflow = "";
let locked = false;
let unsubscribeDepth: (() => void) | null = null;
let observedDepth = 0;
function unlock() {
  if (!locked) return;
  document.body.style.overflow = previousOverflow;
  locked = false;
  unsubscribeDepth?.();
  unsubscribeDepth = null;
}
watch(
  () => props.open,
  async (open) => {
    if (open) {
      opener =
        document.activeElement instanceof HTMLElement
          ? document.activeElement
          : null;
      previousOverflow = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      locked = true;
      cleared.value = null;
      recent.value = false;
      search.value = "";
      observedDepth = overlayDepth();
      unsubscribeDepth = subscribeOverlayDepth((depth) => {
        const previous = observedDepth;
        observedDepth = depth;
        if (depth < previous && depth === 1 && props.open)
          void nextTick(() => textarea.value?.focus());
      });
      await nextTick();
      await nextTick();
      textarea.value?.focus();
    } else {
      unlock();
      await nextTick();
      if (opener?.isConnected && overlayDepth() === 0) opener.focus();
    }
  },
  { immediate: true },
);
onBeforeUnmount(unlock);
watch(
  () => props.prompt,
  (value) => {
    if (value !== "") cleared.value = null;
  },
);
function author(text: string, source: PromptAuthoringSource = "typed") {
  cleared.value = null;
  emit("authored", text, source);
}
function clear() {
  cleared.value = props.prompt;
  emit("authored", "", "clear");
  textarea.value?.focus();
}
function undoClear() {
  const text = cleared.value;
  cleared.value = null;
  if (text !== null) emit("authored", text, "undo-clear");
  textarea.value?.focus();
}
function choose(text: string) {
  author(text, "recalled");
  recent.value = false;
  void nextTick(() => textarea.value?.focus());
}
function keydown(event: KeyboardEvent) {
  // Editing never dispatches the shell's generation or expansion shortcuts.
  if (
    (event.metaKey || event.ctrlKey) &&
    ["Enter", "e", "E"].includes(event.key)
  )
    event.preventDefault();
  if (event.key !== "Escape" && event.key !== "Tab") event.stopPropagation();
}
</script>
<template>
  <Teleport to="body">
    <ModalPanel
      :open="open"
      title="Prompt"
      :width="960"
      class="prompt-editor"
      :style="
        viewportHeight
          ? {
              '--prompt-editor-height': `${viewportHeight}px`,
              '--prompt-editor-top': `${viewportTop}px`,
            }
          : undefined
      "
      @close="emit('close')"
      @keydown="keydown"
    >
      <div class="prompt-editor__content" @keydown="keydown">
        <div class="prompt-editor__toolbar">
          <button
            type="button"
            data-test="prompt-recent-toggle"
            :aria-expanded="recent"
            @click="recent = !recent"
          >
            {{ recent ? "Back to prompt" : "Recent prompts" }}
          </button>
          <button
            type="button"
            v-if="cleared === null"
            data-test="prompt-clear"
            :disabled="!prompt"
            @click="clear"
          >
            Clear
          </button>
          <button
            v-if="cleared !== null"
            type="button"
            data-test="prompt-undo-clear"
            @click="undoClear"
          >
            Undo clear
          </button>
          <details v-if="slots.tools" class="prompt-editor__rewrite">
            <summary>Rewrite</summary>
            <div
              class="prompt-editor__rewrite-menu"
              @click="
                (
                  $event.currentTarget as HTMLElement
                ).parentElement?.removeAttribute('open')
              "
            >
              <slot name="tools" />
            </div>
          </details>
        </div>
        <div v-if="recent" class="prompt-editor__recent">
          <input
            v-model="search"
            type="search"
            placeholder="Search recent prompts"
            aria-label="Search recent prompts"
          />
          <div class="prompt-editor__results">
            <button
              v-for="text in results"
              :key="text"
              type="button"
              data-test="prompt-history-item"
              @click="choose(text)"
            >
              {{ text }}
            </button>
            <p v-if="historyLoading">Loading recent prompts…</p>
            <p v-else-if="historyError" role="status">{{ historyError }}</p>
            <p v-else-if="!results.length">No recent prompts</p>
          </div>
        </div>
        <textarea
          v-show="!recent"
          ref="textarea"
          placeholder="Write your prompt…"
          :value="prompt"
          aria-label="Prompt text"
          spellcheck="true"
          @input="author(($event.target as HTMLTextAreaElement).value)"
        />
      </div>
      <template #footer
        ><button type="button" data-test="prompt-done" @click="emit('close')">
          Done
        </button></template
      >
    </ModalPanel>
  </Teleport>
</template>
<style scoped>
.prompt-editor {
  font-family: var(--mold-font-sans);
  font-size: var(--mold-fs-sm);
  box-sizing: border-box;
  position: fixed;
  top: var(--prompt-editor-top, 0px);
  bottom: auto;
  height: var(--prompt-editor-height, 100dvh);
  z-index: calc(var(--mold-z-modal, 100) - 1);
  padding: 32px;
}
.prompt-editor :deep(.ms-modal__panel) {
  max-width: 960px;
  width: 100% !important;
  height: min(800px, calc(var(--prompt-editor-height, 100dvh) - 64px));
  max-height: calc(var(--prompt-editor-height, 100dvh) - 64px);
}
.prompt-editor :deep(.ms-modal__body) {
  display: flex;
  overflow: hidden;
}
.prompt-editor__content {
  display: flex;
  flex-direction: column;
  gap: 16px;
  flex: 1;
  min-height: 0;
  min-width: 0;
}
.prompt-editor__toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  flex-shrink: 0;
}
.prompt-editor :deep(button),
.prompt-editor input,
.prompt-editor summary {
  font: inherit;
  color: var(--mold-text);
  background: var(--mold-panel-raised, var(--mold-bg));
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-3);
  padding: 8px 12px;
  min-height: 44px;
  box-sizing: border-box;
}
.prompt-editor :deep(button:disabled) {
  opacity: 0.45;
}
.prompt-editor textarea {
  flex: 1;
  min-height: 120px;
  width: 100%;
  resize: none;
  overflow: auto;
  padding: 16px;
  font: inherit;
  line-height: 1.65;
  font-size: max(1rem, var(--mold-fs-md));
  color: var(--mold-text);
  background: var(--mold-bg);
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-3);
}
.prompt-editor__recent {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  gap: 10px;
}
.prompt-editor__results {
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.prompt-editor__results button {
  text-align: left;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  flex-shrink: 0;
}
.prompt-editor :deep(:is(button, input, textarea, summary):focus-visible) {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}
@media (max-width: 600px) {
  .prompt-editor {
    padding: 0;
  }
  .prompt-editor :deep(.ms-modal__panel) {
    height: var(--prompt-editor-height, 100dvh);
    max-height: var(--prompt-editor-height, 100dvh);
    max-width: 100%;
    border-radius: 0;
  }
  .prompt-editor textarea {
    min-height: 0;
  }
  .prompt-editor :deep(.ms-modal__footer) {
    padding-bottom: max(14px, env(safe-area-inset-bottom));
  }
}

.prompt-editor__rewrite {
  position: relative;
}
.prompt-editor summary {
  cursor: pointer;
  list-style: none;
}
.prompt-editor summary::-webkit-details-marker {
  display: none;
}
.prompt-editor__rewrite-menu {
  position: absolute;
  z-index: 1;
  right: 0;
  top: calc(100% + 6px);
  min-width: 180px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
  background: var(--mold-panel-raised, var(--mold-bg));
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-3);
  box-shadow: var(--mold-shadow-md);
}
.prompt-editor input {
  font-size: max(1rem, var(--mold-fs-md));
}
@media (max-width: 600px) {
  .prompt-editor__toolbar {
    gap: 8px;
  }
  .prompt-editor :deep(.ms-modal__head) {
    padding: 12px 16px;
  }
  .prompt-editor :deep(.ms-modal__footer) {
    padding-top: 8px;
    padding-bottom: max(8px, env(safe-area-inset-bottom));
  }
}
</style>
