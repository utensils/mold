<script setup lang="ts">
/**
 * One server-queue row, in full, on every surface.
 *
 * The model is resolved by `queueEntryDetailModel` so web, desktop, and
 * iPhone cannot disagree about what a queued job says about itself; this file
 * only paints it. Actions are emitted rather than performed, because each
 * shell owns its own authenticated target for the exact selected host.
 */
import { computed, ref } from "vue";
import type { QueueEntryDetailModel } from "../lib/queueEntryDetail";
import { copyTextToClipboard } from "../lib/notificationClipboard";

const props = withDefaults(
  defineProps<{
    model: QueueEntryDetailModel;
    /** Live denoise snapshot for a running row, when the host emits one. */
    /** The host's folded progress snapshot: a denoise image is optional,
     * and a host with previews disabled still reports the step counter. */
    preview?: {
      preview_image: string | null;
      step: number | null;
      total: number | null;
    } | null;
    cancelling?: boolean;
    retrying?: boolean;
    /** Failure from the last action, shown inline and never as a toast. */
    error?: string | null;
    /** `inline` runs the phone's own two-step confirm inside this component;
     * `delegate` emits immediately so desktop and web can raise their shared
     * ConfirmDialog. */
    confirm?: "inline" | "delegate";
    /** Phone layout: single column, 44pt controls, 16px editable text. */
    compact?: boolean;
  }>(),
  {
    preview: null,
    cancelling: false,
    retrying: false,
    error: null,
    confirm: "delegate",
    compact: false,
  },
);

const emit = defineEmits<{
  (e: "close"): void;
  (e: "reuse"): void;
  (e: "cancel"): void;
  (e: "retry"): void;
}>();

const cancelArmed = ref(false);
const copied = ref(false);

const cancelLabel = computed(() => {
  if (props.cancelling) return "Cancelling…";
  if (props.confirm === "inline" && cancelArmed.value) return "Cancel job?";
  return props.model.running ? "Stop job" : "Cancel job";
});

function onCancel(): void {
  if (props.cancelling) return;
  if (props.confirm === "inline" && !cancelArmed.value) {
    cancelArmed.value = true;
    return;
  }
  cancelArmed.value = false;
  emit("cancel");
}

async function copyDetail(): Promise<void> {
  copied.value = await copyTextToClipboard(props.model.copyText);
  if (copied.value) setTimeout(() => (copied.value = false), 2000);
}
</script>

<template>
  <div
    class="qed"
    :class="{ 'qed--compact': compact }"
    data-test="queue-entry-detail"
  >
    <header class="qed__head">
      <span class="qed__state" :data-state="model.stateLabel.toLowerCase()">
        {{ model.stateCode }}
      </span>
      <h2 class="qed__title" :title="model.modelId">{{ model.modelLabel }}</h2>
      <button
        type="button"
        class="qed__close"
        aria-label="Close job detail"
        data-test="queue-detail-close"
        @click="emit('close')"
      >
        ✕
      </button>
    </header>

    <div class="qed__body">
      <figure
        v-if="preview && (preview.preview_image || preview.step !== null)"
        class="qed__preview"
        data-test="queue-detail-preview"
      >
        <img
          v-if="preview.preview_image"
          :src="`data:image/png;base64,${preview.preview_image}`"
          alt="Latest denoise step for this job"
        />
        <figcaption v-if="preview.step !== null && preview.total !== null">
          Step {{ preview.step }} of {{ preview.total }}
        </figcaption>
      </figure>

      <p
        v-if="model.problem"
        class="qed__problem"
        role="alert"
        data-test="queue-detail-problem"
      >
        <strong>{{ model.problem.title }}</strong>
        <span>{{ model.problem.detail }}</span>
      </p>

      <p v-if="model.title" class="qed__printtitle">{{ model.title }}</p>

      <section v-if="model.prompt" class="qed__prompt">
        <span class="qed__legend">PROMPT</span>
        <p data-test="queue-detail-prompt">{{ model.prompt }}</p>
        <p v-if="model.negativePrompt" class="qed__negative">
          −&nbsp;{{ model.negativePrompt }}
        </p>
      </section>

      <p
        v-if="model.settingsNotice"
        class="qed__notice"
        data-test="queue-detail-settings-notice"
      >
        {{ model.settingsNotice }}
      </p>

      <section
        v-for="group in model.groups"
        :key="group.title"
        class="qed__group"
        :data-test="`queue-detail-group-${group.title.toLowerCase().replace(/\s+/g, '-')}`"
      >
        <span class="qed__legend">{{ group.title.toUpperCase() }}</span>
        <dl>
          <div v-for="field in group.fields" :key="field.label">
            <dt>{{ field.label }}</dt>
            <dd :class="{ qed__mono: field.mono }">{{ field.value }}</dd>
          </div>
        </dl>
      </section>

      <section class="qed__group" data-test="queue-detail-facts">
        <span class="qed__legend">QUEUE</span>
        <dl>
          <div>
            <dt>Waiting</dt>
            <dd>{{ model.waitLabel }}</dd>
          </div>
          <div v-for="field in model.facts" :key="field.label">
            <dt>{{ field.label }}</dt>
            <dd :class="{ qed__mono: field.mono }">{{ field.value }}</dd>
          </div>
        </dl>
      </section>
    </div>

    <footer class="qed__foot">
      <p
        v-if="error"
        class="qed__error"
        role="alert"
        data-test="queue-detail-error"
      >
        {{ error }}
      </p>
      <p
        v-if="!model.reuse.available && model.reuse.blockedReason"
        class="qed__hint"
      >
        {{ model.reuse.blockedReason }}
      </p>
      <p
        v-if="model.retry.applicable && !model.retry.available"
        class="qed__hint"
        data-test="queue-detail-retry-hint"
      >
        {{ model.retry.blockedReason }}
      </p>
      <p v-if="model.cancel.blockedReason" class="qed__hint">
        {{ model.cancel.blockedReason }}
      </p>
      <div class="qed__actions">
        <button
          type="button"
          data-test="queue-detail-reuse"
          :disabled="!model.reuse.available"
          @click="emit('reuse')"
        >
          Reuse settings
        </button>
        <button type="button" data-test="queue-detail-copy" @click="copyDetail">
          {{ copied ? "Copied" : "Copy details" }}
        </button>
        <button
          v-if="model.retry.applicable"
          type="button"
          data-test="queue-detail-retry"
          :disabled="!model.retry.available || retrying"
          @click="emit('retry')"
        >
          {{ retrying ? "Retrying…" : "Retry" }}
        </button>
        <button
          v-if="model.cancel.applicable"
          type="button"
          class="qed__danger"
          data-test="queue-detail-cancel"
          :disabled="!model.cancel.available || cancelling"
          @click="onCancel"
          @blur="cancelArmed = false"
        >
          {{ cancelLabel }}
        </button>
      </div>
    </footer>
  </div>
</template>

<style scoped>
.qed {
  display: flex;
  width: 100%;
  min-width: 0;
  max-width: 100%;
  min-height: 0;
  box-sizing: border-box;
  flex-direction: column;
  color: var(--mold-text-2, currentColor);
  font-size: 12px;
}
.qed--compact {
  font-size: 15px;
}
.qed__head {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--mold-border, var(--mold-border-control));
}
.qed__state {
  padding: 2px 6px;
  border: 1px solid var(--line, var(--mold-border-control));
  border-radius: var(--mold-radius-2, 999px);
  font-size: 10px;
  letter-spacing: 0.04em;
  white-space: nowrap;
}
.qed__state[data-state="running"] {
  border-color: var(--mold-blue);
  color: var(--mold-blue);
}
.qed__state[data-state="held"] {
  border-color: var(--mold-warning);
  color: var(--mold-warning);
}
.qed__title {
  overflow: hidden;
  min-width: 0;
  flex: 1;
  margin: 0;
  color: var(--mold-text);
  font-size: 13px;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.qed--compact .qed__title {
  font-size: 17px;
}
.qed__close {
  min-width: 32px;
  min-height: 32px;
  border: 0;
  background: none;
  color: inherit;
  cursor: pointer;
}
.qed--compact .qed__close {
  min-width: 44px;
  min-height: 44px;
}
.qed__body {
  display: flex;
  min-width: 0;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  gap: 12px;
  padding: 14px 16px;
  overflow-x: hidden;
  overflow-y: auto;
}
.qed__preview {
  margin: 0;
}
.qed__preview img {
  width: 100%;
  border-radius: var(--mold-radius-2, 12px);
}
.qed__preview figcaption {
  margin-top: 4px;
  color: var(--mold-text-dim, currentColor);
  font-size: 11px;
}
.qed__problem {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin: 0;
  padding: 10px;
  border: 1px solid var(--mold-error);
  border-radius: var(--mold-radius-2, 9px);
  color: var(--mold-error);
  white-space: pre-wrap;
  word-break: break-word;
}
.qed__printtitle {
  margin: 0;
  color: var(--mold-text);
  font-weight: 600;
}
.qed__legend {
  color: var(--mold-text-dim, currentColor);
  font-size: 10px;
  letter-spacing: 0.08em;
}
.qed__prompt p,
.qed__negative {
  margin: 4px 0 0;
  line-height: 1.5;
  /* The shells disable chrome text selection; a prompt must stay selectable. */
  user-select: text;
  word-break: break-word;
}
.qed__negative {
  color: var(--mold-text-dim, currentColor);
}
.qed__notice {
  margin: 0;
  color: var(--mold-text-dim, currentColor);
  line-height: 1.5;
}
.qed__group dl {
  display: grid;
  margin: 6px 0 0;
  gap: 6px 12px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.qed--compact .qed__group dl {
  grid-template-columns: minmax(0, 1fr);
}
.qed__group dt {
  color: var(--mold-text-dim, currentColor);
  font-size: 10px;
}
.qed--compact .qed__group dt {
  font-size: 12px;
}
.qed__group dd {
  margin: 0;
  overflow-wrap: anywhere;
}
.qed__mono {
  font-family: var(--mold-font-mono, ui-monospace, monospace);
}
.qed__foot {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 8px;
  padding: 12px 16px;
  border-top: 1px solid var(--mold-border, var(--mold-border-control));
}
.qed__error {
  margin: 0;
  color: var(--mold-error);
  word-break: break-word;
}
.qed__hint {
  margin: 0;
  color: var(--mold-text-dim, currentColor);
  line-height: 1.4;
}
.qed__actions {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  gap: 8px;
}
.qed__actions button {
  min-height: 32px;
  flex: 1 1 auto;
  padding: 0 10px;
  border: 1px solid var(--line, var(--mold-border-control));
  border-radius: var(--mold-radius-2, 9px);
  background: none;
  color: inherit;
  cursor: pointer;
}
.qed--compact .qed__actions button {
  min-width: 0;
  min-height: 44px;
  font-size: 16px;
}
.qed--compact .qed__actions {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.qed__actions button:disabled {
  cursor: default;
  opacity: 0.5;
}
.qed__danger {
  border-color: var(--mold-error) !important;
  color: var(--mold-error);
}
</style>
