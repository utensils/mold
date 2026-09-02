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
  isPreparedBatch.value ? `Prepare ${props.batchSize} variations` : "Expand",
);
const progressLabel = computed(() => {
  const host = props.hostLabel ?? "the selected host";
  return isPreparedBatch.value
    ? `Expanding ${props.batchSize} prompts on ${host}…`
    : `Expanding on ${host}…`;
});

function expand() {
  if (props.transformBlockedReason) return;
  if (!props.blocked && !props.running && props.prompt.trim()) emit("expand");
}

defineExpose({ expand });
</script>

<template>
  <div class="flex min-w-0 flex-wrap items-center gap-2">
    <button
      type="button"
      data-test="expand-action"
      class="border-edge min-h-7 rounded-control border px-2 text-body text-ink-2 transition-colors duration-100 hover:border-safelight hover:text-ink active:translate-y-px disabled:opacity-50"
      :disabled="!!transformBlockedReason || blocked || running || !prompt.trim()"
      :title="
        transformBlockedReason
          ? transformBlockedReason
          : blocked
            ? 'Refresh or discard the preserved prepared batch first'
            : isPreparedBatch
              ? `Prepare ${batchSize} prompt variations`
              : 'Expand prompt'
      "
      @click="expand"
    >
      {{ actionLabel }}
      <kbd v-if="!running" class="kbd-hint ml-1">{{ shortcutLabel("E") }}</kbd>
    </button>

    <button
      type="button"
      data-test="remix-action"
      class="border-edge min-h-7 rounded-control border px-2 text-body text-ink-2 transition-colors duration-100 hover:border-safelight hover:text-ink active:translate-y-px disabled:opacity-50"
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
    <label v-if="originalAvailable" class="flex items-center gap-1 text-caption text-ink-3">
      Source
      <select
        data-test="remix-source-select"
        class="border-edge min-h-7 rounded-control border bg-bath px-1.5 text-caption text-ink-2 disabled:opacity-50"
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
      class="min-h-7 rounded-control px-1.5 text-body text-halide transition-colors duration-100 hover:text-ink"
      title="Restore original prompt"
      aria-label="Restore original prompt"
      @click="emit('restore')"
    >
      ↩
    </button>

    <span
      v-if="transformBlockedReason"
      data-test="transform-blocked-hint"
      class="min-w-0 text-caption text-ink-3"
      >{{ transformBlockedReason }}</span
    >

    <span
      v-else-if="running"
      role="status"
      aria-live="polite"
      class="min-w-0 text-caption text-halide"
    >
      {{ progressLabel }}
    </span>
  </div>
</template>
