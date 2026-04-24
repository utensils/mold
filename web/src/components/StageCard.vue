<script setup lang="ts">
import { computed } from "vue";
import type { ChainStageToml } from "../lib/chainToml";

const props = defineProps<{
  index: number;
  isFirst: boolean;
  stage: ChainStageToml;
  framesPerClipCap: number;
  fadeFramesMax: number;
}>();

const emit = defineEmits<{
  (e: "update:stage", v: ChainStageToml): void;
  (e: "delete"): void;
  (e: "move-up"): void;
  (e: "move-down"): void;
  (e: "duplicate"): void;
  (e: "expand"): void;
  (e: "pick-image"): void;
  (e: "clear-image"): void;
}>();

function updatePrompt(v: string) {
  emit("update:stage", { ...props.stage, prompt: v });
}
function updateFrames(v: number) {
  emit("update:stage", { ...props.stage, frames: v });
}
function updateTransition(v: "smooth" | "cut" | "fade") {
  emit("update:stage", { ...props.stage, transition: v });
}

const durationSec = computed(() => (props.stage.frames / 24).toFixed(2));
const frameOptions = computed(() => {
  const out: number[] = [];
  for (let n = 9; n <= props.framesPerClipCap; n += 8) out.push(n);
  return out;
});

const hasSourceImage = computed(() => Boolean(props.stage.source_image_b64));
const sourceImageLabel = computed(() => {
  const name = props.stage.source_image;
  if (name && name.length > 0) return name;
  return "source image";
});
</script>

<template>
  <div class="glass rounded-2xl p-3 space-y-2">
    <div class="flex items-center gap-2 text-xs text-slate-400">
      <span class="drag-handle cursor-grab select-none">⋮⋮</span>
      <span>{{ index + 1 }}</span>
      <template v-if="!isFirst">
        <div class="inline-flex rounded-full bg-slate-900/60 p-0.5">
          <button
            type="button"
            :class="
              (stage.transition ?? 'smooth') === 'smooth'
                ? 'bg-brand-500/60'
                : ''
            "
            class="rounded-full px-2 py-0.5"
            @click="updateTransition('smooth')"
          >
            smooth
          </button>
          <button
            type="button"
            :class="stage.transition === 'cut' ? 'bg-brand-500/60' : ''"
            class="rounded-full px-2 py-0.5"
            @click="updateTransition('cut')"
          >
            cut
          </button>
          <button
            type="button"
            :class="stage.transition === 'fade' ? 'bg-brand-500/60' : ''"
            class="rounded-full px-2 py-0.5"
            @click="updateTransition('fade')"
          >
            fade
          </button>
        </div>
      </template>
      <span v-else class="italic opacity-60">Opening frame</span>
      <div class="ml-auto flex items-center gap-2">
        <select
          class="rounded-full bg-slate-900/60 px-2 py-0.5 text-xs"
          :value="stage.frames"
          @change="
            updateFrames(Number(($event.target as HTMLSelectElement).value))
          "
        >
          <option v-for="n in frameOptions" :key="n" :value="n">
            {{ n }}f ({{ (n / 24).toFixed(2) }}s)
          </option>
        </select>
        <button
          class="icon-btn"
          aria-label="Duplicate"
          @click="emit('duplicate')"
        >
          ⎘
        </button>
        <button class="icon-btn" aria-label="Move up" @click="emit('move-up')">
          ↑
        </button>
        <button
          class="icon-btn"
          aria-label="Move down"
          @click="emit('move-down')"
        >
          ↓
        </button>
        <button class="icon-btn" aria-label="Delete" @click="emit('delete')">
          ✕
        </button>
      </div>
    </div>

    <div class="flex items-start gap-2">
      <div v-if="hasSourceImage" class="relative flex-shrink-0">
        <img
          :src="`data:image/png;base64,${stage.source_image_b64}`"
          :alt="sourceImageLabel"
          class="h-12 w-12 rounded-xl object-cover"
        />
        <button
          type="button"
          class="absolute -right-1 -top-1 h-5 w-5 rounded-full bg-slate-900/90 text-xs text-slate-100 hover:bg-slate-800"
          aria-label="Remove source image"
          title="Remove source image"
          @click="emit('clear-image')"
        >
          ✕
        </button>
      </div>
      <button
        v-else
        type="button"
        class="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-xl border border-dashed border-slate-700 bg-slate-900/40 text-lg text-slate-400 hover:border-slate-500 hover:text-slate-200"
        aria-label="Attach source image"
        title="Attach source image"
        @click="emit('pick-image')"
      >
        🖼️
      </button>

      <textarea
        class="min-h-[2.5rem] flex-1 resize-none bg-transparent text-base text-slate-100 placeholder:text-slate-500 focus:outline-none"
        :value="stage.prompt"
        placeholder="Describe this stage…"
        @input="updatePrompt(($event.target as HTMLTextAreaElement).value)"
      />
    </div>

    <div class="flex flex-wrap items-center gap-2 text-xs text-slate-500">
      <button class="hover:text-slate-200" @click="emit('expand')">
        ✨ Expand
      </button>
      <button
        v-if="hasSourceImage"
        class="hover:text-slate-200"
        @click="emit('pick-image')"
      >
        🖼️ Replace image
      </button>
      <button v-else class="hover:text-slate-200" @click="emit('pick-image')">
        🖼️ Attach image
      </button>
      <span>{{ durationSec }}s</span>
      <span
        v-if="
          !isFirst &&
          (stage.transition ?? 'smooth') === 'smooth' &&
          stage.source_image_b64
        "
        class="text-amber-400"
        title="Smooth transitions use the prior clip's motion tail; this image still anchors character/scene identity across the stage boundary"
      >
        smooth transition — image used as identity anchor
      </span>
    </div>
  </div>
</template>

<style scoped>
.icon-btn {
  display: inline-flex;
  height: 1.5rem;
  width: 1.5rem;
  align-items: center;
  justify-content: center;
  border-radius: 9999px;
  color: rgb(148 163 184);
  transition: color 150ms ease;
}
.icon-btn:hover {
  color: rgb(226 232 240);
}
</style>
