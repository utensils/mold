<script setup lang="ts">
import { computed } from "vue";
import type { ChainStageForm } from "../../lib/chainForm";
import { cycleTransition } from "../../lib/chain";

// The splice sits before `stage` and encodes how it joins the previous clip.
const props = defineProps<{ stage: ChainStageForm; motionTail: number; fadeMax: number }>();

const label = computed(() => {
  if (props.stage.transition === "smooth") return `${props.motionTail}f tail`;
  if (props.stage.transition === "cut") return "cut";
  return "fade";
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
      class="flex h-16 w-8 items-center justify-center rounded-control hover:bg-bench"
      :title="`Transition: ${stage.transition} — click to cycle`"
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
      <button type="button" class="text-ink-3 hover:text-ink" @click="stepFade(-1)">◂</button>
      <span class="data-mono text-caption text-ink">{{ stage.fadeFrames }}</span>
      <button type="button" class="text-ink-3 hover:text-ink" @click="stepFade(1)">▸</button>
    </div>
  </div>
</template>
