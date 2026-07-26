<script setup lang="ts">
import { computed, ref, watch } from "vue";
import DevelopCanvas from "@ui/components/DevelopCanvas.vue";
import type { DevelopPhase } from "@ui/lib/grain";
import type { ChainStageForm } from "../../lib/chainForm";
import { frames8n1Error, snapFrames } from "../../lib/chain";
import { authedMediaUrl } from "../../lib/gallery/media";
import { base64ToDataUrl } from "../../lib/image";
import type { ApiTarget } from "../../lib/api/client";
import type { ChainJobStageDetail } from "../../lib/api/types";

const props = defineProps<{
  stage: ChainStageForm;
  index: number;
  baseSeed: string;
  jobStage?: ChainJobStageDetail | null;
  progress?: { step: number; total: number } | null;
  jobId?: string | null;
  apiTarget?: ApiTarget | null;
  hostId?: string | null;
  canMoveLeft: boolean;
  canMoveRight: boolean;
  canRemove: boolean;
}>();
const emit = defineEmits<{
  (e: "remove"): void;
  (e: "move-left"): void;
  (e: "move-right"): void;
  (e: "pick-image"): void;
  (e: "clear-image"): void;
}>();

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

// Accessible name for the develop/preview cell — its state is otherwise
// conveyed only by the grain color (temperature semantics).
const developAria = computed(() => {
  const base = props.index === 0 ? "Opening clip" : `Clip ${props.index + 1}`;
  const state = props.jobStage?.state;
  if (state === "completed") return `${base}: completed`;
  if (state === "failed") return `${base}: failed`;
  if (state === "running") {
    return props.progress
      ? `${base}: developing, step ${props.progress.step} of ${props.progress.total}`
      : `${base}: developing`;
  }
  return `${base}: not started`;
});

// Load the stage's JPEG preview once the job reports it exists.
watch(
  () => [props.jobId, props.jobStage?.has_preview] as const,
  ([jobId, hasPreview]) => {
    if (jobId && hasPreview) {
      void authedMediaUrl(
        `/api/chain-jobs/${encodeURIComponent(jobId)}/stages/${props.index}/preview`,
        {
          ...(props.apiTarget ? { target: props.apiTarget } : {}),
          ...(props.hostId ? { cacheKey: props.hostId } : {}),
        },
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
      <span>{{ index === 0 ? "Opening clip" : `Clip ${index + 1}` }}</span>
      <span class="data-mono">{{ stage.frames }}f</span>
    </div>

    <!-- 96px Develop grain / preview -->
    <div
      class="relative h-24 w-full overflow-hidden rounded-media border border-[color-mix(in_srgb,var(--rebate)_18%,transparent)] bg-print-surface"
      role="img"
      :aria-label="developAria"
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

    <!-- source image well: attach / replace / clear a starting still -->
    <div v-if="stage.sourceImage" class="relative">
      <button
        type="button"
        data-test="stage-image-thumb"
        class="border-edge block h-10 w-full overflow-hidden rounded-media border hover:brightness-110"
        title="Replace source image"
        aria-label="Replace source image"
        @click="emit('pick-image')"
      >
        <img :src="base64ToDataUrl(stage.sourceImage)" alt="" class="h-full w-full object-cover" />
      </button>
      <button
        type="button"
        data-test="stage-image-clear"
        class="border-edge absolute -top-1 -right-1 flex h-4 w-4 items-center justify-center rounded-full border bg-bath text-[10px] text-ink-2 hover:text-stop"
        title="Remove source image"
        aria-label="Remove source image"
        @click="emit('clear-image')"
      >
        ✕
      </button>
    </div>
    <button
      v-else
      type="button"
      data-test="stage-image-attach"
      class="border-edge h-6 w-full rounded-control border border-dashed text-caption text-ink-3 hover:text-ink"
      title="Attach a starting image for this stage"
      aria-label="Attach source image"
      @click="emit('pick-image')"
    >
      + Image
    </button>

    <textarea
      v-model="stage.prompt"
      data-selectable
      rows="3"
      :aria-label="index === 0 ? 'Opening clip prompt' : `Clip ${index + 1} prompt`"
      :placeholder="index === 0 ? 'How does the sequence begin?' : 'What happens next?'"
      class="border-edge min-h-16 w-full resize-none rounded-control border bg-bath px-2 py-1.5 text-caption text-ink placeholder:text-ink-3"
    />

    <div class="flex items-center justify-between">
      <div class="flex items-center gap-1">
        <button
          type="button"
          class="text-ink-3 hover:text-ink active:translate-y-px disabled:opacity-30"
          :disabled="!canMoveLeft"
          title="Move left"
          aria-label="Move stage left"
          @click="emit('move-left')"
        >
          ◂
        </button>
        <button
          type="button"
          class="text-ink-3 hover:text-ink active:translate-y-px disabled:opacity-30"
          :disabled="!canMoveRight"
          title="Move right"
          aria-label="Move stage right"
          @click="emit('move-right')"
        >
          ▸
        </button>
      </div>
      <button
        type="button"
        class="text-ink-3 hover:text-stop active:translate-y-px disabled:opacity-30"
        :disabled="!canRemove"
        title="Remove stage"
        aria-label="Remove stage"
        @click="emit('remove')"
      >
        ✕
      </button>
    </div>

    <details class="border-edge mt-1 rounded-control border bg-bath p-2">
      <summary class="cursor-pointer text-caption text-ink-2">Clip tools</summary>
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
    </details>
  </div>
</template>
