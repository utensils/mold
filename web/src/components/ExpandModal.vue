<script setup lang="ts">
import { computed, ref } from "vue";
import { expandPrompt } from "../api";
import type { ExpandFormState, ModelInfoExtended } from "../types";

const props = defineProps<{
  open: boolean;
  prompt: string;
  expand: ExpandFormState;
  currentModel: ModelInfoExtended | null;
  /** Preview hits the LLM server-side; the queue is a single lane, so
   * previewing while a generation is running would block the UI on the
   * inference slot. Disable the button instead of silently queueing it. */
  queueBusy?: boolean;
  /**
   * The composer's active style as a natural-language directive (`styleHint`)
   * the server weaves into the expander's system message. Null for a chain
   * stage — the style row belongs to the single-print composer, not to stage
   * text — and never appended to the prompt itself.
   */
  styleDirective?: string | null;
}>();
const emit = defineEmits<{
  (e: "update:expand", v: ExpandFormState): void;
  (e: "apply-prompt", v: string): void;
  (e: "close"): void;
}>();

const previewing = ref(false);
const previewError = ref<string | null>(null);
const previewResults = ref<string[]>([]);

const effectiveFamily = computed(
  () => props.expand.familyOverride ?? props.currentModel?.family ?? "flux",
);

const variationsOptions = [1, 3, 5] as const;

function patch<K extends keyof ExpandFormState>(key: K, v: ExpandFormState[K]) {
  emit("update:expand", { ...props.expand, [key]: v });
}

async function preview() {
  previewing.value = true;
  previewError.value = null;
  try {
    const style = props.styleDirective?.trim();
    const res = await expandPrompt({
      prompt: props.prompt,
      model_family: effectiveFamily.value,
      variations: props.expand.variations,
      ...(style ? { style } : {}),
    });
    previewResults.value = res.expanded;
  } catch (e) {
    previewError.value = e instanceof Error ? e.message : String(e);
  } finally {
    previewing.value = false;
  }
}

function pick(text: string) {
  emit("apply-prompt", text);
  emit("close");
}
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-40 flex items-center justify-center bg-bath/70 backdrop-blur-sm"
      @click.self="emit('close')"
    >
      <div
        class="bg-bench border border-edge max-h-[90vh] w-full max-w-xl overflow-y-auto rounded-3xl p-6 sm:p-8"
      >
        <div class="flex items-center justify-between">
          <h2 class="text-lg font-semibold text-rebate">✨ Prompt expansion</h2>
          <button
            type="button"
            class="text-ink-3 hover:text-rebate"
            @click="emit('close')"
          >
            ✕
          </button>
        </div>

        <label class="mt-4 flex items-center gap-2 text-sm text-ink-2">
          <input
            type="checkbox"
            :checked="expand.enabled"
            @change="
              patch('enabled', ($event.target as HTMLInputElement).checked)
            "
          />
          Enable expansion before submit
        </label>

        <div class="mt-3">
          <label class="text-xs uppercase text-ink-3">Variations</label>
          <div class="mt-1 flex gap-1">
            <button
              v-for="n in variationsOptions"
              :key="n"
              type="button"
              class="rounded-full px-3 py-1 text-sm"
              :class="
                expand.variations === n
                  ? 'bg-safelight text-white'
                  : 'bg-bench/60 text-ink-2'
              "
              @click="patch('variations', n)"
            >
              {{ n }}
            </button>
          </div>
        </div>

        <details class="mt-3 text-sm text-ink-2">
          <summary class="cursor-pointer text-ink-3">
            Advanced: model family override
          </summary>
          <input
            type="text"
            :value="expand.familyOverride ?? ''"
            placeholder="auto"
            class="mt-1 w-full rounded-lg bg-bench/60 px-2 py-1 text-rebate"
            @change="
              patch(
                'familyOverride',
                ($event.target as HTMLInputElement).value || null,
              )
            "
          />
        </details>

        <div class="mt-4 flex items-center gap-2">
          <button
            type="button"
            class="rounded-lg bg-safelight px-3 py-1.5 text-sm text-white disabled:opacity-50"
            :disabled="previewing || !prompt.trim() || queueBusy"
            :title="
              queueBusy
                ? 'Another generation is in the queue — wait for it to finish before previewing.'
                : undefined
            "
            data-test="expand-preview"
            @click="preview"
          >
            {{ previewing ? "Expanding…" : "Preview" }}
          </button>
          <span v-if="queueBusy" class="text-xs text-ink-3">
            Queue busy — preview disabled.
          </span>
        </div>

        <div v-if="previewError" class="mt-2 text-sm text-rose-300">
          {{ previewError }}
        </div>

        <ul v-if="previewResults.length" class="mt-4 flex flex-col gap-2">
          <li v-for="(text, i) in previewResults" :key="i">
            <button
              type="button"
              class="w-full rounded-xl bg-bench/60 p-3 text-left text-sm text-rebate hover:bg-surface/80"
              @click="pick(text)"
            >
              {{ text }}
            </button>
          </li>
        </ul>

        <p class="mt-4 text-xs text-ink-3">
          Backend model, temperature, and thinking mode are controlled by the
          server config. Ask your operator to adjust them.
        </p>
      </div>
    </div>
  </Teleport>
</template>
