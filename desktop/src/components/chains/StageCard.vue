<script setup lang="ts">
import { computed, ref, watch } from "vue";
import DevelopCanvas from "../../lib/develop/DevelopCanvas.vue";
import type { DevelopPhase } from "../../lib/develop/grain";
import type { ChainStageForm } from "../../lib/chainForm";
import { frames8n1Error, snapFrames } from "../../lib/chain";
import { authedMediaUrl } from "../../lib/gallery/media";
import type { ChainJobStageDetail } from "../../lib/api/types";

const props = defineProps<{
  stage: ChainStageForm;
  index: number;
  baseSeed: string;
  jobStage?: ChainJobStageDetail | null;
  progress?: { step: number; total: number } | null;
  jobId?: string | null;
  canMoveLeft: boolean;
  canMoveRight: boolean;
  canRemove: boolean;
}>();
const emit = defineEmits<{
  (e: "remove"): void;
  (e: "move-left"): void;
  (e: "move-right"): void;
}>();

const editing = ref(false);
const previewUrl = ref<string | null>(null);

const phase = computed<DevelopPhase>(() => {
  const state = props.jobStage?.state;
  if (state === "completed") return "fixed";
  if (state === "failed") return "stopped";
  if (state === "running") return "developing";
  return "latent";
});
const progressValue = computed(() => {
  if (props.jobStage?.state === "completed") return 1;
  if (props.progress && props.progress.total > 0) return props.progress.step / props.progress.total;
  return 0;
});
const framesError = computed(() => frames8n1Error(props.stage.frames));

// Load the stage's JPEG preview once the job reports it exists.
watch(
  () => [props.jobId, props.jobStage?.has_preview] as const,
  ([jobId, hasPreview]) => {
    if (jobId && hasPreview) {
      void authedMediaUrl(
        `/api/chain-jobs/${encodeURIComponent(jobId)}/stages/${props.index}/preview`,
      )
        .then((u) => (previewUrl.value = u))
        .catch(() => (previewUrl.value = null));
    } else {
      previewUrl.value = null;
    }
  },
  { immediate: true },
);

function snapFramesField() {
  props.stage.frames = snapFrames(props.stage.frames);
}
</script>

<template>
  <div class="border-edge flex w-40 shrink-0 flex-col gap-1 rounded-chrome border bg-bench p-2">
    <div class="edge-code flex items-center justify-between">
      <span>Stage {{ index + 1 }}</span>
      <span class="data-mono">{{ stage.frames }}f</span>
    </div>

    <!-- 96px Develop grain / preview -->
    <div
      class="relative h-24 w-full overflow-hidden rounded-media border border-[color-mix(in_srgb,var(--rebate)_18%,transparent)] bg-print-surface"
    >
      <img
        v-if="previewUrl"
        :src="previewUrl"
        alt=""
        class="absolute inset-0 h-full w-full object-cover"
      />
      <DevelopCanvas
        v-else
        :seed="`${baseSeed}·${index}·${stage.prompt}`"
        :progress="progressValue"
        :phase="phase"
        class="absolute inset-0"
      />
      <div v-if="progress && phase === 'developing'" class="edge-code absolute bottom-1 left-1.5">
        {{ progress.step }}/{{ progress.total }}
      </div>
    </div>

    <!-- prompt (click to edit) -->
    <button
      type="button"
      class="line-clamp-2 min-h-8 rounded-control px-1 text-left text-caption text-ink-2 hover:bg-bath hover:text-ink"
      :title="stage.prompt || 'Add a prompt'"
      @click="editing = !editing"
    >
      {{ stage.prompt || "Add a prompt…" }}
    </button>

    <div class="flex items-center justify-between">
      <div class="flex items-center gap-1">
        <button
          type="button"
          class="text-ink-3 hover:text-ink disabled:opacity-30"
          :disabled="!canMoveLeft"
          title="Move left"
          @click="emit('move-left')"
        >
          ◂
        </button>
        <button
          type="button"
          class="text-ink-3 hover:text-ink disabled:opacity-30"
          :disabled="!canMoveRight"
          title="Move right"
          @click="emit('move-right')"
        >
          ▸
        </button>
      </div>
      <button
        type="button"
        class="text-ink-3 hover:text-stop disabled:opacity-30"
        :disabled="!canRemove"
        title="Remove stage"
        @click="emit('remove')"
      >
        ✕
      </button>
    </div>

    <!-- inline edit popover -->
    <div v-if="editing" class="border-edge mt-1 rounded-control border bg-bath p-2">
      <label class="text-caption text-ink-2">Prompt</label>
      <textarea
        v-model="stage.prompt"
        data-selectable
        rows="2"
        class="border-edge mt-1 w-full resize-none rounded-control border bg-bench px-1.5 py-1 text-caption text-ink"
      />
      <label class="mt-2 text-caption text-ink-2">Frames</label>
      <input
        v-model.number="stage.frames"
        type="number"
        step="8"
        min="1"
        class="border-edge data-mono mt-1 h-7 w-full rounded-control border bg-bench px-1.5 text-ink"
        :class="framesError ? 'border-stop' : ''"
        @change="snapFramesField"
      />
      <p v-if="framesError" class="mt-1 text-caption text-stop">{{ framesError }}</p>
      <label class="mt-2 text-caption text-ink-2">Negative prompt</label>
      <textarea
        v-model="stage.negativePrompt"
        data-selectable
        rows="1"
        placeholder="optional"
        class="border-edge mt-1 w-full resize-none rounded-control border bg-bench px-1.5 py-1 text-caption text-ink placeholder:text-ink-3"
      />
      <button
        type="button"
        class="border-edge mt-2 h-7 w-full rounded-control border text-caption text-ink-2 hover:text-ink"
        @click="editing = false"
      >
        Done
      </button>
    </div>
  </div>
</template>
