<script setup lang="ts">
/**
 * Clean recovery for a generate that 404'd because the routed host doesn't
 * have the model: offer to pull it there and auto-resume the generation
 * (pullResume store) instead of surfacing a raw HTTP error.
 */
import { onMounted, ref } from "vue";

defineProps<{
  model: string;
  hostLabel: string;
  /** Known weights size, when the model exists on another host. */
  sizeGb?: number | null;
}>();
const emit = defineEmits<{ (e: "confirm"): void; (e: "close"): void }>();

// Move focus into the teleported overlay so Esc works immediately
// (house pattern — matches ImagePickerModal).
const cancelBtn = ref<HTMLButtonElement | null>(null);
onMounted(() => cancelBtn.value?.focus());
</script>

<template>
  <Teleport to="body">
    <div
      class="fixed inset-0 z-40 flex items-center justify-center bg-black/70 p-4"
      @click.self="emit('close')"
      @keydown.esc="emit('close')"
    >
      <div
        class="border-edge w-full max-w-md rounded-chrome border bg-bench p-5"
        role="dialog"
        aria-modal="true"
        :aria-label="`Model not on ${hostLabel}`"
        data-test="missing-model-dialog"
      >
        <h2 class="text-body-lg font-semibold text-ink">Model not on {{ hostLabel }}</h2>
        <p class="mt-2 text-body text-ink-2">
          <span class="data-mono text-ink">{{ model }}</span>
          isn't installed on {{ hostLabel
          }}<template v-if="sizeGb"> ({{ sizeGb.toFixed(1) }} GB)</template>. Download it there and
          run this generation automatically once it's ready?
        </p>
        <div class="mt-4 flex justify-end gap-2">
          <button
            ref="cancelBtn"
            type="button"
            class="border-edge h-8 rounded-control border px-3 text-body text-ink-2 hover:text-ink"
            data-test="missing-model-cancel"
            @click="emit('close')"
          >
            Cancel
          </button>
          <button
            type="button"
            class="h-8 rounded-control bg-safelight px-3 text-body font-semibold text-on-accent active:translate-y-px"
            data-test="missing-model-pull"
            @click="emit('confirm')"
          >
            Download and generate
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
