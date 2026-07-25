<script setup lang="ts">
import { onMounted, onUnmounted } from "vue";
import type { EnrichedQueueEntry } from "../../stores/jobs";
import type { OutputMetadata } from "../../lib/api/types";

/**
 * Right-side info panel for one server-queue row — the queue counterpart of
 * the model detail drawer. Shows the job's state and, when the host shares
 * them (additive `metadata` on `GET /api/queue` entries), the submitted
 * settings with a one-click path back into the Generate composer. Seed 0 on
 * the wire means "not pinned" — the parent maps it back to a random seed.
 */
defineProps<{
  entry: EnrichedQueueEntry;
  hostLabel: string;
  stateCode: string;
  elapsed: string | null;
  modelLabel: string;
}>();
const emit = defineEmits<{
  (e: "close"): void;
  (e: "load", metadata: OutputMetadata): void;
}>();

function onKeydown(event: KeyboardEvent): void {
  if (event.key === "Escape") emit("close");
}
onMounted(() => window.addEventListener("keydown", onKeydown));
onUnmounted(() => window.removeEventListener("keydown", onKeydown));

function seedLabel(entry: EnrichedQueueEntry): string {
  if (!entry.metadata) return "";
  // Newer servers say exactly whether the seed was pinned; older wire data
  // falls back to "0 means unpinned".
  const pinned = entry.seed_pinned ?? entry.metadata.seed !== 0;
  return pinned ? String(entry.metadata.seed) : "Random";
}
</script>

<template>
  <aside
    class="border-edge fixed inset-y-0 right-0 z-40 flex w-96 max-w-full flex-col border-l bg-bench shadow-raised"
    role="dialog"
    aria-modal="false"
    :aria-label="`Queued job — ${modelLabel}`"
    data-test="queue-entry-drawer"
  >
    <!-- Header -->
    <div class="border-edge flex items-center gap-2 border-b px-4 py-3">
      <span
        class="h-1.5 w-1.5 shrink-0 rounded-full"
        :class="entry.state === 'running' ? 'bg-safelight' : 'bg-halide'"
        aria-hidden="true"
      />
      <h2 class="min-w-0 flex-1 truncate text-body font-semibold text-ink" :title="modelLabel">
        {{ modelLabel }}
      </h2>
      <button
        type="button"
        class="h-7 shrink-0 rounded-control px-2 text-ink-2 hover:bg-bath hover:text-ink"
        aria-label="Close job detail"
        data-test="drawer-close"
        @click="emit('close')"
      >
        ✕
      </button>
    </div>

    <div class="min-h-0 flex-1 overflow-y-auto">
      <div class="flex flex-col gap-3 p-4">
        <!-- Job state -->
        <dl class="grid grid-cols-2 gap-x-3 gap-y-2">
          <div>
            <dt class="text-caption text-ink-3">State</dt>
            <dd class="data-mono text-body text-ink-2">{{ stateCode }}</dd>
          </div>
          <div>
            <dt class="text-caption text-ink-3">Host</dt>
            <dd class="truncate text-body text-ink-2">{{ hostLabel }}</dd>
          </div>
          <div v-if="elapsed">
            <dt class="text-caption text-ink-3">Elapsed</dt>
            <dd class="data-mono text-body text-ink-2">{{ elapsed }}</dd>
          </div>
          <div>
            <dt class="text-caption text-ink-3">Owner</dt>
            <dd class="text-body text-ink-2">{{ entry.mine ? "This app" : "Other client" }}</dd>
          </div>
        </dl>

        <!-- Submitted settings (additive wire field; absent on older hosts) -->
        <template v-if="entry.metadata">
          <section class="border-edge border-t pt-3" data-test="queue-settings">
            <span class="edge-code">SETTINGS</span>
            <p
              class="mt-1.5 text-caption leading-relaxed text-ink-2"
              data-test="queue-prompt"
              :title="entry.metadata.prompt"
            >
              {{ entry.metadata.prompt }}
            </p>
            <p
              v-if="entry.metadata.negative_prompt"
              class="mt-1 text-caption leading-relaxed text-ink-3"
            >
              −&nbsp;{{ entry.metadata.negative_prompt }}
            </p>
            <dl class="mt-2 grid grid-cols-2 gap-x-3 gap-y-2">
              <div>
                <dt class="text-caption text-ink-3">Size</dt>
                <dd class="data-mono text-body text-ink-2">
                  {{ entry.metadata.width }}×{{ entry.metadata.height }}
                </dd>
              </div>
              <div>
                <dt class="text-caption text-ink-3">Steps</dt>
                <dd class="data-mono text-body text-ink-2">{{ entry.metadata.steps }}</dd>
              </div>
              <div>
                <dt class="text-caption text-ink-3">Guidance</dt>
                <dd class="data-mono text-body text-ink-2">{{ entry.metadata.guidance }}</dd>
              </div>
              <div>
                <dt class="text-caption text-ink-3">Seed</dt>
                <dd class="data-mono text-body text-ink-2">{{ seedLabel(entry) }}</dd>
              </div>
              <div v-if="entry.metadata.frames">
                <dt class="text-caption text-ink-3">Frames</dt>
                <dd class="data-mono text-body text-ink-2">
                  {{ entry.metadata.frames
                  }}<template v-if="entry.metadata.fps"> @ {{ entry.metadata.fps }} fps</template>
                </dd>
              </div>
            </dl>
            <div v-if="entry.metadata.loras?.length" class="mt-2 flex flex-wrap gap-1">
              <span
                v-for="lora in entry.metadata.loras"
                :key="lora.path"
                class="border-edge data-mono rounded-full border px-1.5 py-0.5 text-caption text-ink-3"
                :title="lora.path"
              >
                {{ lora.path.split("/").pop() }} · {{ lora.scale }}
              </span>
            </div>
          </section>
        </template>
        <p v-else class="border-edge border-t pt-3 text-caption text-ink-3">
          This host doesn't share queued-job settings — upgrade its server to inspect and reuse
          them.
        </p>
      </div>
    </div>

    <!-- Action -->
    <div class="border-edge border-t p-4">
      <button
        type="button"
        data-test="queue-load-settings"
        class="border-edge h-8 w-full rounded-control border text-body text-safelight transition-colors duration-150 hover:border-safelight active:translate-y-px disabled:opacity-50"
        :disabled="!entry.metadata"
        :title="
          entry.metadata
            ? 'Open Generate with this job\'s settings'
            : 'This host does not share queued-job settings'
        "
        @click="entry.metadata && emit('load', entry.metadata)"
      >
        Load settings in Generate
      </button>
    </div>
  </aside>
</template>
