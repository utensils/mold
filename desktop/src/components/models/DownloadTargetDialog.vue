<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, onUnmounted, ref } from "vue";
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
const closeBtn = ref<HTMLButtonElement | null>(null);
/** One verb for the whole dialog: an install is on offer, or it is a repair. */
const action = computed<"install" | "repair">(() =>
  props.targets.some((target) => target.action === "install") ? "install" : "repair",
);
const title = computed(() => `Choose where to ${action.value} ${props.modelName}`);
let restoreFocusEl: HTMLElement | null = null;

function onKeydown(event: KeyboardEvent) {
  if (event.key === "Escape") emit("close");
}

onMounted(() => {
  restoreFocusEl = document.activeElement as HTMLElement | null;
  window.addEventListener("keydown", onKeydown);
  void nextTick(() => closeBtn.value?.focus());
});
onUnmounted(() => window.removeEventListener("keydown", onKeydown));
onBeforeUnmount(() => restoreFocusEl?.focus?.());
</script>

<template>
  <Teleport to="body">
    <div
      class="fixed inset-0 z-40 flex items-center justify-center bg-black/55 p-5"
      @click.self="emit('close')"
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="download-target-title"
        data-test="download-target-dialog"
        class="border-edge z-50 w-full max-w-md rounded-chrome border bg-bench p-4 shadow-raised"
      >
        <div class="mb-3 flex items-start justify-between gap-4">
          <div>
            <h2 id="download-target-title" class="text-body-lg font-semibold text-ink">
              {{ title }}
            </h2>
            <p class="mt-1 text-caption text-ink-2">
              <template v-if="action === 'repair'">
                Only missing or damaged files will be fetched on the selected host.
              </template>
              <template v-else>
                The model and its required components will be stored on the machine you pick.
              </template>
            </p>
          </div>
          <button
            ref="closeBtn"
            type="button"
            class="h-7 rounded-control px-2 text-ink-2 hover:bg-bath hover:text-ink"
            :aria-label="`Close ${action} target picker`"
            @click="emit('close')"
          >
            ✕
          </button>
        </div>

        <div class="flex flex-col gap-1" role="list">
          <button
            v-for="target in targets"
            :key="target.host.id"
            type="button"
            role="listitem"
            :data-test="`download-target-${target.host.id}`"
            class="border-edge flex min-h-12 items-center gap-3 rounded-control border px-3 py-2 text-left transition-colors duration-150 hover:border-safelight hover:bg-bath"
            @click="emit('select', target.host)"
          >
            <span
              class="h-2 w-2 shrink-0 rounded-full"
              :class="target.host.status === 'ready' ? 'bg-halide' : 'bg-stop'"
              aria-hidden="true"
            />
            <span class="min-w-0 flex-1">
              <span class="block truncate text-body font-medium text-ink">
                {{ target.host.label }}
              </span>
              <span class="block truncate text-caption text-ink-3">
                {{ target.host.kind === "local" ? "This device" : target.host.baseUrl }}
              </span>
            </span>
            <!-- Says what picking this machine actually does, so a mixed list
                 never leaves the user guessing which one is the fresh copy. -->
            <span
              class="edge-code shrink-0"
              :class="target.action === 'install' ? 'text-safelight' : 'text-ink-3'"
            >
              {{ target.action === "install" ? "Install" : "Already installed · repair" }}
            </span>
            <span v-if="target.host.queueDepth != null" class="data-mono text-caption text-ink-3">
              {{ target.host.queueDepth }} queued
            </span>
          </button>
        </div>
      </section>
    </div>
  </Teleport>
</template>
