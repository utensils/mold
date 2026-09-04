<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import type { ModelInstallTarget } from "@studio/lib/modelInstallTargets";
import type { HostView } from "../../stores/hosts";

const props = defineProps<{
  modelName: string;
  /** Every machine this model can be sent to, and what sending it there does.
   *  A mixed list is normal: some machines lack the model, others already
   *  have it and can only be repaired. */
  targets: ModelInstallTarget<HostView>[];
}>();
const emit = defineEmits<{
  (e: "select", host: HostView): void;
  (e: "close"): void;
}>();
/** One verb for the whole dialog: the style is on offer, or it is a repair. */
const action = computed<"install" | "repair">(() =>
  props.targets.some((target) => target.action === "install") ? "install" : "repair",
);
const title = computed(() =>
  action.value === "repair"
    ? `Which machine should repair ${props.modelName}?`
    : `Where should ${props.modelName} go?`,
);
/** Fresh and repair targets in the same list — the copy has to cover both. */
const mixed = computed(
  () =>
    props.targets.some((target) => target.action === "install") &&
    props.targets.some((target) => target.action === "repair"),
);
const restoreFocusEl = ref<HTMLElement | null>(null);

onMounted(() => {
  restoreFocusEl.value = document.activeElement as HTMLElement | null;
});
onBeforeUnmount(() => restoreFocusEl.value?.focus?.());
</script>

<template>
  <ModalPanel
    :open="true"
    :width="480"
    :title="title"
    data-test="download-target-dialog"
    @close="emit('close')"
  >
    <template #description>
      <template v-if="action === 'repair'">
        Only the missing or damaged files are fetched on the machine you pick.
      </template>
      <!-- A mixed list must not promise a fresh copy for a machine that can
           only be repaired. -->
      <template v-else-if="mixed">
        The style and everything it needs are kept on the machine you pick; machines that already
        have it are repaired instead.
      </template>
      <template v-else>
        The style and everything it needs are kept on the machine you pick.
      </template>
    </template>

    <div class="flex flex-col gap-1" role="list">
      <button
        v-for="target in targets"
        :key="target.host.id"
        type="button"
        role="listitem"
        :data-test="`download-target-${target.host.id}`"
        class="border-border flex min-h-12 items-center gap-3 rounded-control border px-3 py-2 text-left transition-colors duration-150 hover:border-accent hover:bg-bg-deep"
        @click="emit('select', target.host)"
      >
        <span
          class="h-2 w-2 shrink-0 rounded-full"
          :class="target.host.status === 'ready' ? 'bg-success' : 'bg-error'"
          aria-hidden="true"
        />
        <span class="min-w-0 flex-1">
          <span class="block truncate text-sm font-medium text-fg">
            {{ target.host.label }}
          </span>
          <span class="block truncate text-micro text-fg-dim">
            {{ target.host.kind === "local" ? "This device" : target.host.baseUrl }}
          </span>
        </span>
        <!-- Says what picking this machine actually does, so a mixed list
             never leaves the user guessing which one is the fresh copy. -->
        <span
          class="font-mono text-micro whitespace-nowrap shrink-0"
          :class="target.action === 'install' ? 'text-accent' : 'text-fg-dim'"
        >
          {{ target.action === "install" ? "Get it" : "Already here · repair" }}
        </span>
        <span v-if="target.host.queueDepth != null" class="font-mono text-micro text-fg-dim">
          {{ target.host.queueDepth }} queued
        </span>
      </button>
    </div>

    <template #footer>
      <button
        type="button"
        data-test="download-target-cancel"
        class="min-h-8 rounded-control border border-border px-3.5 py-1.5 text-xs text-fg-2 transition-colors duration-100 hover:border-border-focus hover:text-fg"
        @click="emit('close')"
      >
        Cancel
      </button>
    </template>
  </ModalPanel>
</template>
