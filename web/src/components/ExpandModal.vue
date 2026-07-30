<script setup lang="ts">
/*
 * Prompt-expansion overlay (spec §06 Create). Previews LLM rewrites of the
 * composer prompt and applies the one the user picks.
 *
 * Mold Studio migration: was a hand-rolled `fixed inset-0` panel with an emoji
 * title, Tailwind utility colors and no Escape handling. Now @ui ModalPanel +
 * SegmentedControl + Icon, with `useOverlayFocus` supplying the Tab trap and
 * opener restoration the kit leaves to the host.
 */
import { computed, ref } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import Icon from "@ui/components/Icon.vue";
import { expandPrompt } from "../api";
import type { StreamTarget } from "../api";
import { useOverlayFocus } from "../composables/useOverlayFocus";
import type { ExpandFormState, ModelInfoExtended } from "../types";

const props = defineProps<{
  open: boolean;
  prompt: string;
  expand: ExpandFormState;
  currentModel: ModelInfoExtended | null;
  /**
   * The composer's active style as a natural-language directive (`styleHint`)
   * the server weaves into the expander's system message. Null for a chain
   * stage — the style row belongs to the single-print composer, not to stage
   * text — and never appended to the prompt itself.
   */
  styleDirective?: string | null;
  /** Exact generation route. Expansion must use the same host as the print. */
  target?: StreamTarget;
}>();
const emit = defineEmits<{
  (e: "update:expand", v: ExpandFormState): void;
  (e: "apply-prompt", v: string): void;
  (e: "close"): void;
}>();

const previewing = ref(false);
const previewError = ref<string | null>(null);
const previewResults = ref<string[]>([]);

const host = ref<HTMLElement | null>(null);
const isOpen = computed(() => props.open);
const { onKeydown } = useOverlayFocus(isOpen, host, () => emit("close"));

const effectiveFamily = computed(
  () => props.expand.familyOverride ?? props.currentModel?.family ?? "flux",
);

type Variations = ExpandFormState["variations"];

// Typed to the literal union so SegmentedControl's generic resolves to
// `1 | 3 | 5` rather than widening to `number` at the update handler.
const variationsOptions: { value: Variations; label: string }[] = [
  { value: 1, label: "1" },
  { value: 3, label: "3" },
  { value: 5, label: "5" },
];

function patch<K extends keyof ExpandFormState>(key: K, v: ExpandFormState[K]) {
  emit("update:expand", { ...props.expand, [key]: v });
}

async function preview() {
  previewing.value = true;
  previewError.value = null;
  try {
    const style = props.styleDirective?.trim();
    const request = {
      prompt: props.prompt,
      model_family: effectiveFamily.value,
      variations: props.expand.variations,
      ...(style ? { style } : {}),
    };
    const res = props.target
      ? await expandPrompt(request, undefined, props.target)
      : await expandPrompt(request);
    previewResults.value = res.expanded;
  } catch (e) {
    previewError.value = e instanceof Error ? e.message : String(e);
  } finally {
    previewing.value = false;
  }
}

function pick(text: string) {
  emit("apply-prompt", text);
  emit("close");
}
</script>

<template>
  <!-- Viewport-fixed host: ModalPanel is `position: absolute; inset: 0` and
       Create is a long scrolling column, so mounting inline would draw the
       panel at the top of the document. Same host pattern as
       ModelDetailDrawer. -->
  <div
    v-if="open"
    ref="host"
    class="xm-host"
    data-test="expand-modal-host"
    @keydown="onKeydown"
  >
    <ModalPanel
      :open="open"
      :width="560"
      label="Prompt expansion"
      @close="emit('close')"
    >
      <div class="xm">
        <header class="xm__head">
          <h2 class="xm__title">
            <Icon name="sparkle" :size="17" />
            <span>Prompt expansion</span>
          </h2>
          <button
            type="button"
            class="xm__close"
            aria-label="Close"
            data-test="expand-close"
            @click="emit('close')"
          >
            <Icon name="close" :size="15" />
          </button>
        </header>

        <div class="xm__body">
          <label class="xm__check">
            <input
              type="checkbox"
              :checked="expand.enabled"
              @change="
                patch('enabled', ($event.target as HTMLInputElement).checked)
              "
            />
            <span>Enable expansion before submit</span>
          </label>

          <div class="xm__field">
            <span class="xm__label">Variations</span>
            <SegmentedControl
              :model-value="expand.variations"
              :options="variationsOptions"
              label="Variations"
              @update:model-value="(v: Variations) => patch('variations', v)"
            />
          </div>

          <details class="xm__adv">
            <summary>Advanced: model family override</summary>
            <input
              type="text"
              :value="expand.familyOverride ?? ''"
              placeholder="auto"
              class="xm__input"
              @change="
                patch(
                  'familyOverride',
                  ($event.target as HTMLInputElement).value || null,
                )
              "
            />
          </details>

          <div class="xm__actions">
            <button
              type="button"
              class="xm__go"
              :disabled="previewing || !prompt.trim()"
              data-test="expand-preview"
              @click="preview"
            >
              {{ previewing ? "Expanding…" : "Preview" }}
            </button>
          </div>

          <p v-if="previewError" class="xm__error">{{ previewError }}</p>

          <ul v-if="previewResults.length" class="xm__results">
            <li v-for="(text, i) in previewResults" :key="i">
              <button type="button" class="xm__result" @click="pick(text)">
                {{ text }}
              </button>
            </li>
          </ul>

          <p class="xm__foot">
            Backend model, temperature, and thinking mode are controlled by the
            server config. Ask your operator to adjust them.
          </p>
        </div>
      </div>
    </ModalPanel>
  </div>
</template>

<style scoped>
.xm-host {
  position: fixed;
  inset: 0;
  z-index: 40;
}

.xm {
  display: flex;
  flex-direction: column;
  gap: 14px;
  color: var(--rebate);
}

.xm__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.xm__title {
  display: flex;
  align-items: center;
  gap: 9px;
  margin: 0;
  font-family: var(--f-display);
  font-size: 17px;
  font-weight: 700;
  letter-spacing: -0.01em;
}

.xm__title svg {
  color: var(--safelight);
}

.xm__close {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  background: transparent;
  color: var(--ink-3);
  cursor: pointer;
  transition:
    color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease);
}

.xm__close:hover {
  color: var(--rebate);
  background: var(--surface);
}

.xm__body {
  display: flex;
  flex-direction: column;
  gap: 14px;
  max-height: min(62vh, 520px);
  overflow-y: auto;
}

.xm__check {
  display: flex;
  align-items: center;
  gap: 9px;
  font-size: 13px;
  color: var(--ink-2);
  cursor: pointer;
}

.xm__check input {
  accent-color: var(--safelight);
}

.xm__field {
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.xm__label {
  font-family: var(--f-mono);
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--ink-3);
}

.xm__adv {
  font-size: 13px;
  color: var(--ink-2);
}

.xm__adv summary {
  cursor: pointer;
  color: var(--ink-3);
  font-size: 12px;
}

.xm__input {
  width: 100%;
  margin-top: 8px;
  padding: 9px 12px;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  color: var(--rebate);
  font-family: var(--f-mono);
  font-size: 13px;
  outline: none;
}

.xm__input:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.xm__actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.xm__go {
  border: 0;
  background: var(--safelight);
  color: var(--on-accent);
  padding: 9px 18px;
  border-radius: var(--radius-control-lg);
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
}

.xm__go:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.xm__note,
.xm__error {
  margin: 0;
  font-family: var(--f-mono);
  font-size: 11px;
  letter-spacing: 0.02em;
  color: var(--ink-3);
}

.xm__error {
  color: var(--stop);
}

.xm__results {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.xm__result {
  width: 100%;
  padding: 12px;
  text-align: left;
  border: 1px solid var(--edge);
  border-radius: var(--radius-control-lg);
  background: var(--bath);
  color: var(--rebate);
  font-family: var(--f-body);
  font-size: 13px;
  line-height: 1.5;
  cursor: pointer;
  transition:
    background var(--dur-quick) var(--ease),
    border-color var(--dur-quick) var(--ease);
}

.xm__result:hover {
  background: var(--surface);
  border-color: var(--safelight);
}

.xm__foot {
  margin: 0;
  font-size: 12px;
  line-height: 1.5;
  color: var(--ink-3);
}
</style>
