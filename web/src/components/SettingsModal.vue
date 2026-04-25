<script setup lang="ts">
import { computed, ref } from "vue";
import type { GenerateFormState, ModelInfoExtended, Scheduler } from "../types";
import {
  NO_CFG_FAMILIES,
  UNET_SCHEDULER_FAMILIES,
  VIDEO_FAMILIES,
} from "../types";
import ModelPicker from "./ModelPicker.vue";
import { outputFormatsForFamily } from "../composables/useGenerateForm";

// ── Model Discovery (catalog auth + NSFW) ────────────────────────────────────
const hfToken = ref("");
const civitaiToken = ref("");
const showNsfw = ref(false);

async function saveSetting(key: string, value: string): Promise<void> {
  await fetch("/api/settings/set", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ key, value }),
  });
}

const props = defineProps<{
  open: boolean;
  modelValue: GenerateFormState;
  models: ModelInfoExtended[];
}>();
const emit = defineEmits<{
  (e: "update:modelValue", v: GenerateFormState): void;
  (e: "close"): void;
}>();

function patch<K extends keyof GenerateFormState>(
  key: K,
  value: GenerateFormState[K],
) {
  emit("update:modelValue", { ...props.modelValue, [key]: value });
}

const currentModel = computed(
  () => props.models.find((m) => m.name === props.modelValue.model) ?? null,
);
const family = computed(() => currentModel.value?.family ?? "");

const showNegative = computed(() => !NO_CFG_FAMILIES.includes(family.value));
const showScheduler = computed(() =>
  UNET_SCHEDULER_FAMILIES.includes(family.value),
);
const showVideo = computed(() => VIDEO_FAMILIES.includes(family.value));
const showStrength = computed(() => !!props.modelValue.sourceImage);

function selectModel(m: ModelInfoExtended) {
  const next: GenerateFormState = {
    ...props.modelValue,
    model: m.name,
    width: m.default_width,
    height: m.default_height,
    steps: m.default_steps,
    guidance: m.default_guidance,
  };
  if (VIDEO_FAMILIES.includes(m.family)) {
    next.frames ??= 25;
    next.fps ??= 24;
  } else {
    next.frames = null;
    next.fps = null;
  }
  // Reset output format to the family's default when it's no longer valid
  // (e.g. switching from FLUX → LTX-2 leaves `png` selected, which the
  // server would reject).
  const formats = outputFormatsForFamily(m.family);
  if (!formats.includes(next.outputFormat)) {
    next.outputFormat = formats[0];
  }
  emit("update:modelValue", next);
}

// frames must be 8n+1 (9, 17, 25, 33, ...)
function clampFrames(n: number): number {
  if (!Number.isFinite(n)) return 25;
  const rounded = Math.max(9, Math.round((n - 1) / 8) * 8 + 1);
  return rounded;
}

// Video length is a pure UI convenience — the backend only consumes frames.
// We derive it from frames / fps and let the user edit either side; the
// other recomputes (see onChangeLength / onChangeFps).
const videoLength = computed(() => {
  const frames = props.modelValue.frames ?? 25;
  const fps = props.modelValue.fps ?? 24;
  return fps > 0 ? frames / fps : 0;
});

function onChangeFrames(raw: string) {
  const n = clampFrames(Number(raw));
  patch("frames", n);
}

function onChangeLength(raw: string) {
  const secs = Number(raw);
  if (!Number.isFinite(secs) || secs <= 0) return;
  const fps = props.modelValue.fps ?? 24;
  patch("frames", clampFrames(secs * fps));
}

function onChangeFps(raw: string) {
  const nextFps = Number(raw);
  if (!Number.isFinite(nextFps) || nextFps <= 0) return;
  // Changing fps keeps length steady and adjusts frames.
  const length = videoLength.value;
  emit("update:modelValue", {
    ...props.modelValue,
    fps: nextFps,
    frames: clampFrames(length * nextFps),
  });
}

const advancedOpen = ref(false);

const sizePresets = [512, 768, 1024] as const;
const batchChips = [1, 2, 3, 4] as const;

const schedulerOptions: Scheduler[] = [
  "default",
  "ddim",
  "euler-ancestral",
  "unipc",
];
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-40 flex items-center justify-center bg-slate-950/70 backdrop-blur-sm"
      @click.self="emit('close')"
    >
      <div
        class="glass max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-3xl p-6 sm:p-8"
      >
        <div class="flex items-center justify-between">
          <h2 class="text-lg font-semibold text-slate-100">Settings</h2>
          <button
            type="button"
            class="text-slate-400 hover:text-slate-100"
            @click="emit('close')"
          >
            ✕
          </button>
        </div>

        <section class="mt-4">
          <ModelPicker
            :models="models"
            :model-value="modelValue.model"
            @update:model-value="(v: string) => patch('model', v)"
            @select="selectModel"
          />
        </section>

        <section class="mt-4">
          <label class="text-xs uppercase text-slate-400">Size</label>
          <div class="mt-1 flex flex-wrap gap-2">
            <button
              v-for="n in sizePresets"
              :key="n"
              type="button"
              class="rounded-full px-3 py-1 text-sm"
              :class="
                modelValue.width === n && modelValue.height === n
                  ? 'bg-brand-500 text-white'
                  : 'bg-slate-900/60 text-slate-200'
              "
              @click="
                emit('update:modelValue', {
                  ...modelValue,
                  width: n,
                  height: n,
                })
              "
            >
              {{ n }}×{{ n }}
            </button>
            <div class="flex items-center gap-2 text-sm">
              <input
                type="number"
                :value="modelValue.width"
                class="w-20 rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
                @input="
                  patch(
                    'width',
                    Number(($event.target as HTMLInputElement).value) ||
                      modelValue.width,
                  )
                "
              />
              ×
              <input
                type="number"
                :value="modelValue.height"
                class="w-20 rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
                @input="
                  patch(
                    'height',
                    Number(($event.target as HTMLInputElement).value) ||
                      modelValue.height,
                  )
                "
              />
            </div>
          </div>
        </section>

        <section class="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <label class="text-xs uppercase text-slate-400"
              >Steps — {{ modelValue.steps }}</label
            >
            <input
              type="range"
              min="1"
              max="100"
              :value="modelValue.steps"
              class="w-full"
              @input="
                patch(
                  'steps',
                  Number(($event.target as HTMLInputElement).value),
                )
              "
            />
          </div>
          <div>
            <label class="text-xs uppercase text-slate-400"
              >Guidance — {{ modelValue.guidance.toFixed(1) }}</label
            >
            <input
              type="range"
              min="0"
              max="20"
              step="0.1"
              :value="modelValue.guidance"
              class="w-full"
              @input="
                patch(
                  'guidance',
                  Number(($event.target as HTMLInputElement).value),
                )
              "
            />
          </div>
        </section>

        <section class="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <label class="text-xs uppercase text-slate-400">Seed</label>
            <div class="mt-1 flex items-center gap-2">
              <input
                type="number"
                :value="modelValue.seed ?? ''"
                placeholder="random"
                class="flex-1 rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
                @input="
                  patch(
                    'seed',
                    ($event.target as HTMLInputElement).value === ''
                      ? null
                      : Number(($event.target as HTMLInputElement).value),
                  )
                "
              />
              <button
                type="button"
                class="rounded-lg bg-slate-900/60 px-3 py-1 text-sm"
                @click="patch('seed', null)"
              >
                🎲
              </button>
            </div>
          </div>
          <div>
            <label class="text-xs uppercase text-slate-400">Batch</label>
            <div class="mt-1 flex gap-1">
              <button
                v-for="n in batchChips"
                :key="n"
                type="button"
                class="rounded-full px-3 py-1 text-sm"
                :class="
                  modelValue.batchSize === n
                    ? 'bg-brand-500 text-white'
                    : 'bg-slate-900/60 text-slate-200'
                "
                @click="patch('batchSize', n)"
              >
                {{ n }}
              </button>
            </div>
          </div>
        </section>

        <section v-if="showNegative" class="mt-4">
          <label class="text-xs uppercase text-slate-400"
            >Negative prompt</label
          >
          <textarea
            :value="modelValue.negativePrompt"
            rows="2"
            class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
            placeholder="e.g. blurry, low quality, watermark"
            @input="
              patch(
                'negativePrompt',
                ($event.target as HTMLTextAreaElement).value,
              )
            "
          />
        </section>

        <section v-if="showStrength" class="mt-4">
          <label class="text-xs uppercase text-slate-400"
            >Strength — {{ modelValue.strength.toFixed(2) }}</label
          >
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            :value="modelValue.strength"
            class="w-full"
            @input="
              patch(
                'strength',
                Number(($event.target as HTMLInputElement).value),
              )
            "
          />
        </section>

        <section
          v-if="showVideo"
          class="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-3"
        >
          <div>
            <label class="text-xs uppercase text-slate-400"
              >Frames (8n+1)</label
            >
            <input
              type="number"
              :value="modelValue.frames ?? 25"
              class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
              @change="
                onChangeFrames(($event.target as HTMLInputElement).value)
              "
            />
          </div>
          <div>
            <label class="text-xs uppercase text-slate-400">Length (s)</label>
            <input
              type="number"
              step="0.1"
              min="0.1"
              :value="videoLength.toFixed(2)"
              class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
              data-test="video-length"
              @change="
                onChangeLength(($event.target as HTMLInputElement).value)
              "
            />
          </div>
          <div>
            <label class="text-xs uppercase text-slate-400">FPS</label>
            <input
              type="number"
              :value="modelValue.fps ?? 24"
              class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
              @change="onChangeFps(($event.target as HTMLInputElement).value)"
            />
          </div>
        </section>

        <section class="mt-4">
          <button
            type="button"
            class="text-xs uppercase tracking-wide text-slate-400 hover:text-slate-200"
            @click="advancedOpen = !advancedOpen"
          >
            {{ advancedOpen ? "▾" : "▸" }} Advanced
          </button>

          <div
            v-if="advancedOpen"
            class="mt-2 grid grid-cols-1 gap-4 sm:grid-cols-2"
          >
            <div v-if="showScheduler">
              <label class="text-xs uppercase text-slate-400">Scheduler</label>
              <select
                :value="modelValue.scheduler ?? 'default'"
                class="mt-1 w-full rounded-lg bg-slate-900/60 px-2 py-1 text-slate-100"
                @change="
                  patch(
                    'scheduler',
                    ($event.target as HTMLSelectElement).value as Scheduler,
                  )
                "
              >
                <option
                  v-for="s in schedulerOptions"
                  :key="String(s)"
                  :value="s"
                >
                  {{ s }}
                </option>
              </select>
            </div>
          </div>
        </section>

        <section class="space-y-3 mt-6">
          <h3 class="text-sm uppercase text-zinc-500">Model Discovery</h3>
          <label class="flex items-center justify-between gap-3 text-sm">
            <span>Hugging Face token</span>
            <input
              name="hf_token"
              type="password"
              v-model="hfToken"
              placeholder="hf_..."
              class="bg-zinc-950 border border-zinc-800 rounded px-2 py-1 w-64"
              @change="saveSetting('huggingface.token', hfToken)"
            />
          </label>
          <label class="flex items-center justify-between gap-3 text-sm">
            <span>Civitai token</span>
            <input
              name="civitai_token"
              type="password"
              v-model="civitaiToken"
              placeholder="cv_..."
              class="bg-zinc-950 border border-zinc-800 rounded px-2 py-1 w-64"
              @change="saveSetting('civitai.token', civitaiToken)"
            />
          </label>
          <label class="flex items-center justify-between gap-3 text-sm">
            <span>Show NSFW models</span>
            <input
              name="catalog_show_nsfw"
              type="checkbox"
              v-model="showNsfw"
              @change="saveSetting('catalog.show_nsfw', String(showNsfw))"
            />
          </label>
        </section>
      </div>
    </div>
  </Teleport>
</template>
