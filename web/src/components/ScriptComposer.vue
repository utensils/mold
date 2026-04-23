<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import StageCard from "./StageCard.vue";
import {
  readChainScript,
  writeChainScript,
  type ChainScriptToml,
  type ChainStageToml,
} from "../lib/chainToml";
import { fetchChainLimits, type ChainLimits } from "../api";

const props = defineProps<{
  model: string;
  width: number;
  height: number;
  fps: number;
}>();

const emit = defineEmits<{
  (e: "submit", script: ChainScriptToml): void;
  (e: "expand", stageIndex: number, prompt: string): void;
}>();

const DRAFT_KEY = "mold.chain.draft.v2";

function blankStage(transition?: "smooth" | "cut" | "fade"): ChainStageToml {
  return { prompt: "", frames: 97, transition };
}

function newScript(): ChainScriptToml {
  return {
    schema: "mold.chain.v1",
    chain: {
      model: props.model,
      width: props.width,
      height: props.height,
      fps: props.fps,
      steps: 8,
      guidance: 3.0,
      strength: 1.0,
      motion_tail_frames: 25,
      output_format: "mp4",
    },
    stage: [blankStage()],
  };
}

const script = ref<ChainScriptToml>(newScript());
const limits = ref<ChainLimits | null>(null);
const importFileInput = ref<HTMLInputElement | null>(null);

onMounted(async () => {
  const draft = localStorage.getItem(DRAFT_KEY);
  if (draft) {
    try {
      script.value = JSON.parse(draft);
    } catch {
      /* ignore corrupt draft */
    }
  }
  limits.value = await fetchChainLimits(props.model).catch(() => null);
});

watch(script, (s) => localStorage.setItem(DRAFT_KEY, JSON.stringify(s)), {
  deep: true,
});

watch(
  () => props.model,
  async (m) => {
    script.value.chain.model = m;
    limits.value = await fetchChainLimits(m).catch(() => null);
  },
);

function addStage() {
  script.value.stage.push(blankStage("smooth"));
}
function updateStage(i: number, next: ChainStageToml) {
  script.value.stage[i] = next;
}
function deleteStage(i: number) {
  if (script.value.stage.length <= 1) return;
  script.value.stage.splice(i, 1);
}
function moveUp(i: number) {
  if (i === 0) return;
  const a = script.value.stage;
  [a[i - 1], a[i]] = [a[i], a[i - 1]];
}
function moveDown(i: number) {
  const a = script.value.stage;
  if (i >= a.length - 1) return;
  [a[i], a[i + 1]] = [a[i + 1], a[i]];
}
function duplicate(i: number) {
  script.value.stage.splice(i + 1, 0, { ...script.value.stage[i] });
}

const totalFrames = computed(() => {
  const s = script.value;
  const mt = s.chain.motion_tail_frames;
  let total = 0;
  s.stage.forEach((stage, i) => {
    if (i === 0) {
      total += stage.frames;
      return;
    }
    switch (stage.transition ?? "smooth") {
      case "smooth":
        total += Math.max(0, stage.frames - mt);
        break;
      case "cut":
        total += stage.frames;
        break;
      case "fade":
        total += Math.max(0, stage.frames - (stage.fade_frames ?? 8));
        break;
    }
  });
  return total;
});

function exportToml(): string {
  return writeChainScript(script.value);
}

function copyToml() {
  void navigator.clipboard.writeText(exportToml());
}

function downloadToml() {
  const toml = exportToml();
  const blob = new Blob([toml], { type: "application/toml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "chain.toml";
  a.click();
  URL.revokeObjectURL(url);
}

async function importToml(file: File) {
  const text = await file.text();
  try {
    script.value = readChainScript(text);
  } catch (err) {
    alert(String(err));
  }
}

function handleImportChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) importToml(file);
  input.value = "";
}

const framesPerClipCap = computed(
  () => limits.value?.frames_per_clip_cap ?? 97,
);
const fadeFramesMax = computed(() => limits.value?.fade_frames_max ?? 32);
const maxStages = computed(() => limits.value?.max_stages ?? 16);
const maxTotalFrames = computed(() => limits.value?.max_total_frames ?? 1552);

const overCap = computed(
  () =>
    totalFrames.value > maxTotalFrames.value ||
    script.value.stage.length > maxStages.value,
);

const canAddStage = computed(() => script.value.stage.length < maxStages.value);

function submit() {
  emit("submit", script.value);
}

function getStagePrompt(i: number): string {
  return script.value.stage[i]?.prompt ?? "";
}

function setStagePrompt(i: number, v: string) {
  if (script.value.stage[i]) {
    script.value.stage[i].prompt = v;
  }
}

defineExpose({ getStagePrompt, setStagePrompt });
</script>

<template>
  <div class="flex flex-col gap-3">
    <div class="flex items-center gap-2 text-sm">
      <span class="font-semibold text-slate-100">Script mode</span>
      <div class="ml-auto flex gap-2">
        <button
          class="rounded-lg bg-slate-900/60 px-3 py-1 text-xs text-slate-200 hover:bg-slate-800/80"
          @click="importFileInput?.click()"
        >
          Import
        </button>
        <input
          ref="importFileInput"
          type="file"
          accept=".toml"
          class="hidden"
          @change="handleImportChange"
        />
        <button
          class="rounded-lg bg-slate-900/60 px-3 py-1 text-xs text-slate-200 hover:bg-slate-800/80"
          @click="downloadToml()"
        >
          Export
        </button>
        <button
          class="rounded-lg bg-slate-900/60 px-3 py-1 text-xs text-slate-200 hover:bg-slate-800/80"
          @click="copyToml()"
        >
          Copy TOML
        </button>
      </div>
    </div>

    <StageCard
      v-for="(stage, i) in script.stage"
      :key="i"
      :index="i"
      :is-first="i === 0"
      :stage="stage"
      :frames-per-clip-cap="framesPerClipCap"
      :fade-frames-max="fadeFramesMax"
      @update:stage="updateStage(i, $event)"
      @delete="deleteStage(i)"
      @move-up="moveUp(i)"
      @move-down="moveDown(i)"
      @duplicate="duplicate(i)"
      @expand="emit('expand', i, stage.prompt)"
    />

    <div
      class="flex items-center justify-between text-xs"
      :class="overCap ? 'text-red-400' : 'text-slate-400'"
      :title="
        overCap
          ? 'Reduce frames or stages — server will reject this script'
          : undefined
      "
    >
      <span>
        {{ script.stage.length }} stages · {{ totalFrames }} frames ·
        {{ (totalFrames / script.chain.fps).toFixed(1) }}s @
        {{ script.chain.fps }}fps
      </span>
      <button
        v-if="canAddStage"
        class="rounded-lg bg-slate-900/60 px-3 py-1 text-slate-200 hover:bg-slate-800/80"
        @click="addStage"
      >
        + Add stage
      </button>
      <span v-else class="text-slate-500">Max stages reached</span>
    </div>

    <button
      class="w-full rounded-xl bg-brand-500 py-2 text-sm font-semibold text-white hover:bg-brand-600 disabled:opacity-50"
      :disabled="overCap || script.stage.every((s) => !s.prompt.trim())"
      @click="submit"
    >
      Generate
    </button>
  </div>
</template>
