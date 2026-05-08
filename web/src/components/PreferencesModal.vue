<script setup lang="ts">
import { ref } from "vue";

// Set-once preferences only — per-generation knobs live in
// GenerateParamsPanel. The default scheduler here is a UI default the
// server uses when a request omits the per-call scheduler override; only
// the named string variants are user-selectable.
type SchedulerName = "default" | "ddim" | "euler-ancestral" | "unipc";

const hfToken = ref("");
const civitaiToken = ref("");
const showNsfw = ref(false);
const defaultScheduler = ref<SchedulerName>("default");

async function saveSetting(key: string, value: string): Promise<void> {
  await fetch("/api/settings/set", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ key, value }),
  });
}

defineProps<{
  open: boolean;
}>();

const emit = defineEmits<{
  (e: "close"): void;
}>();

const schedulerOptions: SchedulerName[] = [
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
          <h2 class="text-lg font-semibold text-slate-100">Preferences</h2>
          <button
            type="button"
            class="text-slate-400 hover:text-slate-100"
            @click="emit('close')"
          >
            ✕
          </button>
        </div>

        <section class="mt-6 space-y-3">
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

        <section class="mt-6 space-y-3">
          <h3 class="text-sm uppercase text-zinc-500">Generation</h3>
          <label class="flex items-center justify-between gap-3 text-sm">
            <span>Default scheduler</span>
            <select
              name="default_scheduler"
              v-model="defaultScheduler"
              class="bg-zinc-950 border border-zinc-800 rounded px-2 py-1 w-64 text-slate-100"
              @change="saveSetting('generate.scheduler', defaultScheduler)"
            >
              <option v-for="s in schedulerOptions" :key="s" :value="s">
                {{ s }}
              </option>
            </select>
          </label>
        </section>
      </div>
    </div>
  </Teleport>
</template>
