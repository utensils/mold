<script setup lang="ts">
import { computed } from "vue";
import { shortcutLabel } from "../../lib/platform";

const props = defineProps<{
  prompt: string;
  batchSize: number;
  running: boolean;
  hostLabel: string | null;
  canUndo: boolean;
  blocked?: boolean;
  /**
   * Why the recipe itself refuses a prompt rewrite (`capabilities.prompt.mode:
   * "ignored"` — the family has no text encoder). Both transforms render
   * disabled with this sentence as their tooltip and a visible hint beside
   * them, and the exposed keyboard action stays silent: the view answers the
   * shortcut with the same reason.
   */
  transformBlockedReason?: string | null;
  originalAvailable?: boolean;
  remixSource?: "original" | "current";
}>();

const emit = defineEmits<{
  (e: "expand"): void;
  (e: "remix"): void;
  (e: "update:remixSource", value: "original" | "current"): void;
  (e: "restore"): void;
}>();

const isPreparedBatch = computed(() => props.batchSize > 1);
const actionLabel = computed(() =>
  isPreparedBatch.value ? `Prepare ${props.batchSize} variations` : "Write more for me",
);
const progressLabel = computed(() => {
  const machine = props.hostLabel ?? "the selected machine";
  return isPreparedBatch.value
    ? `Writing ${props.batchSize} versions on ${machine}…`
    : `Writing more on ${machine}…`;
});

function expand() {
  if (props.transformBlockedReason) return;
  if (!props.blocked && !props.running && props.prompt.trim()) emit("expand");
}

defineExpose({ expand });
</script>

<template>
  <div class="ms-expand">
    <button
      type="button"
      data-test="expand-action"
      class="ms-toolbar-button"
      :disabled="!!transformBlockedReason || blocked || running || !prompt.trim()"
      :title="
        transformBlockedReason
          ? transformBlockedReason
          : blocked
            ? 'Refresh or discard the preserved prepared batch first'
            : isPreparedBatch
              ? `Prepare ${batchSize} prompt variations`
              : 'Write more for me'
      "
      @click="expand"
    >
      {{ actionLabel }}
      <kbd v-if="!running" class="ms-expand__chord">{{ shortcutLabel("E") }}</kbd>
    </button>

    <button
      type="button"
      data-test="remix-action"
      class="ms-toolbar-button"
      :disabled="!!transformBlockedReason || blocked || running || !prompt.trim()"
      :title="
        transformBlockedReason
          ? transformBlockedReason
          : isPreparedBatch
            ? `Prepare ${batchSize} subject-preserving prompt remixes`
            : 'Remix this prompt in place'
      "
      @click="emit('remix')"
    >
      Remix
    </button>
    <label v-if="originalAvailable" class="ms-expand__source">
      Source
      <select
        data-test="remix-source-select"
        class="ms-expand__select"
        :disabled="!!transformBlockedReason"
        :value="remixSource ?? 'original'"
        @change="
          emit(
            'update:remixSource',
            ($event.target as HTMLSelectElement).value as 'original' | 'current',
          )
        "
      >
        <option value="original">Original idea</option>
        <option value="current">Current prompt</option>
      </select>
    </label>

    <button
      v-if="!isPreparedBatch && canUndo"
      type="button"
      class="ms-expand__restore"
      title="Restore original prompt"
      aria-label="Restore original prompt"
      @click="emit('restore')"
    >
      ↩
    </button>

    <span
      v-if="transformBlockedReason"
      data-test="transform-blocked-hint"
      class="ms-expand__note"
      >{{ transformBlockedReason }}</span
    >

    <span
      v-else-if="running"
      role="status"
      aria-live="polite"
      class="ms-expand__note ms-expand__note--live"
    >
      {{ progressLabel }}
    </span>
  </div>
</template>

<style scoped>
.ms-expand {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.ms-expand__chord {
  font-family: var(--mold-font-mono);
  color: var(--mold-text-dim);
}

.ms-expand__source {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}

.ms-expand__select {
  height: var(--mold-ctl-md);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  padding: 0 6px;
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-2);
}

.ms-expand__select:disabled {
  opacity: 0.5;
}

.ms-expand__restore {
  height: var(--mold-ctl-md);
  border: 0;
  background: transparent;
  padding: 0 6px;
  font-size: var(--mold-fs-sm);
  color: var(--mold-blue);
  cursor: pointer;
  transition: color var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-expand__restore:hover {
  color: var(--mold-text);
}

.ms-expand__note {
  min-width: 0;
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}

.ms-expand__note--live {
  color: var(--mold-blue);
}
</style>
