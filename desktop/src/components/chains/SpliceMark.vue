<script setup lang="ts">
import { computed } from "vue";
import { transitionLabel } from "@mold/studio";
import type { ChainStageForm } from "../../lib/chainForm";
import { cycleTransition } from "../../lib/chain";

// The splice sits before `stage` and encodes how it joins the previous clip.
const props = defineProps<{ stage: ChainStageForm; motionTail: number; fadeMax: number }>();

const label = computed(() => {
  return transitionLabel(props.stage.transition);
});

function cycle() {
  props.stage.transition = cycleTransition(props.stage.transition);
}
function stepFade(delta: number) {
  props.stage.fadeFrames = Math.min(props.fadeMax, Math.max(1, props.stage.fadeFrames + delta));
}
</script>

<template>
  <div class="flex flex-col items-center justify-center gap-1 px-1">
    <button
      type="button"
      class="flex h-16 w-8 items-center justify-center rounded-control transition-colors duration-100 hover:bg-bench active:translate-y-px"
      :title="`${label} — click to change transition`"
      :aria-label="`${label}. Click to change transition.`"
      @click="cycle"
    >
      <!-- smooth: unbroken strip · cut: hard diagonal · fade: gradient wedge -->
      <span
        v-if="stage.transition === 'smooth'"
        class="h-10 w-1.5 rounded-full bg-halide"
        aria-hidden="true"
      />
      <span
        v-else-if="stage.transition === 'cut'"
        class="h-10 w-3 bg-transparent"
        style="border-left: 2px solid var(--stop); transform: skewX(-18deg)"
        aria-hidden="true"
      />
      <span
        v-else
        class="h-10 w-3 rounded-full"
        style="background: linear-gradient(90deg, transparent, var(--safelight))"
        aria-hidden="true"
      />
    </button>
    <span class="edge-code">{{ label }}</span>
    <div v-if="stage.transition === 'fade'" class="flex items-center gap-0.5">
      <button
        type="button"
        class="text-ink-3 hover:text-ink active:translate-y-px"
        aria-label="Fewer fade frames"
        @click="stepFade(-1)"
      >
        ◂
      </button>
      <span class="data-mono text-caption text-ink">{{ stage.fadeFrames }}</span>
      <button
        type="button"
        class="text-ink-3 hover:text-ink active:translate-y-px"
        aria-label="More fade frames"
        @click="stepFade(1)"
      >
        ▸
      </button>
    </div>
  </div>
</template>
