<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { expandPrompt, type ExpandPromptOptions } from "../../lib/api/expand";
import { ApiError } from "../../lib/api/client";
import { startCatalogDownload } from "../../lib/api/catalog";
import { parseMissingExpandModel } from "../../lib/expandErrors";
import { useToastStore } from "../../stores/toasts";
import { shortcutLabel } from "../../lib/platform";

const props = defineProps<{ prompt: string; family: string }>();
const emit = defineEmits<{
  (e: "apply", payload: { expanded: string; original: string }): void;
  (e: "restore", original: string): void;
}>();

const toasts = useToastStore();
const running = ref(false);
const lastOriginal = ref<string | null>(null);
/** Set when the engine says the expansion model isn't installed. */
const missingModel = ref<string | null>(null);

// Variations popover — preview candidates without touching the prompt.
const VARIATION_OPTIONS = [1, 3, 5] as const;
const root = ref<HTMLElement | null>(null);
const open = ref(false);
const variations = ref<number>(1);
const familyOverride = ref("");
const previewing = ref(false);
const previewError = ref<string | null>(null);
const candidates = ref<string[]>([]);
/** Prompt captured at preview time — every candidate click undoes to this. */
const previewOriginal = ref<string | null>(null);

const requestOptions = computed<ExpandPromptOptions>(() => {
  const family = familyOverride.value.trim() || props.family;
  return family
    ? { variations: variations.value, modelFamily: family }
    : { variations: variations.value };
});

async function pullExpandModel() {
  const model = missingModel.value;
  if (!model) return;
  try {
    await startCatalogDownload(model);
    missingModel.value = null;
    toasts.push(`Pulling ${model} — watch it land in Models`);
  } catch (err) {
    toasts.push(err instanceof ApiError ? err.message : `Couldn't pull ${model}.`, "error");
  }
}

/** Shared 404/503 handling — returns the message to surface. */
function describeExpandError(err: unknown): string {
  // Surface the engine's actual message — "generic error" toasts hide
  // actionable problems like a missing expansion model.
  const message = err instanceof Error ? err.message : null;
  const missing = message && err instanceof ApiError ? parseMissingExpandModel(message) : null;
  if (missing) {
    missingModel.value = missing;
    return `The expansion model ${missing} isn't installed.`;
  }
  return message || "Couldn't expand the prompt.";
}

async function expand() {
  const original = props.prompt.trim();
  if (!original || running.value) return;
  running.value = true;
  try {
    const res = await expandPrompt(original, {
      variations: 1,
      ...(props.family ? { modelFamily: props.family } : {}),
    });
    const expanded = res.expanded[0]?.trim();
    if (!expanded) {
      toasts.push("Expansion returned nothing.", "error");
      return;
    }
    lastOriginal.value = original;
    emit("apply", { expanded, original });
  } catch (err) {
    toasts.push(describeExpandError(err), "error");
  } finally {
    running.value = false;
  }
}

async function preview() {
  const original = props.prompt.trim();
  if (!original || previewing.value) return;
  previewing.value = true;
  previewError.value = null;
  // Drop any prior candidates up front so a failed re-preview can't leave
  // stale, clickable suggestions from an earlier run.
  candidates.value = [];
  previewOriginal.value = null;
  try {
    const res = await expandPrompt(original, requestOptions.value);
    candidates.value = res.expanded.map((t) => t.trim()).filter(Boolean);
    previewOriginal.value = original;
    if (!candidates.value.length) previewError.value = "Expansion returned nothing.";
  } catch (err) {
    previewError.value = describeExpandError(err);
  } finally {
    previewing.value = false;
  }
}

function applyCandidate(text: string) {
  const original = previewOriginal.value ?? props.prompt.trim();
  lastOriginal.value = original;
  emit("apply", { expanded: text, original });
}

function undo() {
  if (lastOriginal.value === null) return;
  emit("restore", lastOriginal.value);
  lastOriginal.value = null;
}

function onDocumentMousedown(e: MouseEvent) {
  if (!open.value) return;
  const target = e.target;
  if (root.value && !(target instanceof Node && root.value.contains(target))) open.value = false;
}
function onDocumentKeydown(e: KeyboardEvent) {
  if (open.value && e.key === "Escape") open.value = false;
}
onMounted(() => {
  document.addEventListener("mousedown", onDocumentMousedown);
  document.addEventListener("keydown", onDocumentKeydown);
});
onUnmounted(() => {
  document.removeEventListener("mousedown", onDocumentMousedown);
  document.removeEventListener("keydown", onDocumentKeydown);
});

// ⌘E in the composer triggers the same expand.
defineExpose({ expand });
</script>

<template>
  <div ref="root" class="relative flex items-center gap-1">
    <button
      type="button"
      class="border-edge h-7 rounded-control border px-2 text-body text-ink-2 hover:text-ink disabled:opacity-50"
      :disabled="running || !prompt.trim()"
      title="Expand prompt"
      @click="expand"
    >
      {{ running ? "Expanding…" : "Expand" }}
      <kbd v-if="!running" class="kbd-hint ml-1">{{ shortcutLabel("E") }}</kbd>
    </button>
    <button
      type="button"
      class="border-edge h-7 rounded-control border px-1.5 text-body text-ink-2 hover:text-ink"
      title="Expansion options"
      aria-label="Expansion options"
      aria-haspopup="true"
      :aria-expanded="open"
      data-test="expand-options"
      @click="open = !open"
    >
      {{ open ? "▴" : "▾" }}
    </button>
    <button
      v-if="missingModel"
      type="button"
      class="h-7 rounded-control border border-safelight px-2 text-body text-safelight hover:brightness-110"
      @click="pullExpandModel"
    >
      Pull {{ missingModel }}
    </button>
    <button
      v-if="lastOriginal !== null"
      type="button"
      class="h-7 rounded-control px-1 text-body text-halide hover:text-ink"
      title="Restore original prompt"
      aria-label="Restore original prompt"
      @click="undo"
    >
      ↩
    </button>

    <div
      v-if="open"
      data-test="expand-popover"
      class="border-edge absolute top-full left-0 z-30 mt-1 w-80 rounded-chrome border bg-bench p-3 shadow-[0_4px_8px_rgba(0,0,0,0.5)]"
    >
      <div class="flex items-center gap-2">
        <span class="edge-code">Variations</span>
        <div class="flex gap-1">
          <button
            v-for="n in VARIATION_OPTIONS"
            :key="n"
            type="button"
            class="h-6 min-w-7 rounded-control border px-2 text-body"
            :class="
              variations === n
                ? 'border-safelight text-safelight'
                : 'border-edge text-ink-2 hover:text-ink'
            "
            :data-test="`expand-variations-${n}`"
            @click="variations = n"
          >
            {{ n }}
          </button>
        </div>
      </div>

      <label class="mt-3 block">
        <span class="edge-code">Family override</span>
        <input
          v-model="familyOverride"
          data-selectable
          type="text"
          :placeholder="family || 'auto'"
          class="border-edge mt-1 h-7 w-full rounded-control border bg-bath px-2 text-body text-ink placeholder:text-ink-3"
          data-test="expand-family"
        />
      </label>

      <button
        type="button"
        class="border-edge mt-3 h-7 w-full rounded-control border px-2 text-body text-ink-2 hover:text-ink disabled:opacity-50"
        :disabled="previewing || !prompt.trim()"
        data-test="expand-preview"
        @click="preview"
      >
        {{ previewing ? "Expanding…" : "Preview candidates" }}
      </button>

      <div v-if="previewError" data-test="expand-preview-error" class="mt-2 text-caption text-stop">
        {{ previewError }}
      </div>

      <ul v-if="candidates.length" class="mt-2 flex max-h-64 flex-col gap-1 overflow-y-auto">
        <li v-for="(text, i) in candidates" :key="i">
          <button
            type="button"
            class="border-edge w-full rounded-control border bg-bath p-2 text-left text-body text-ink hover:border-safelight"
            data-test="expand-candidate"
            @click="applyCandidate(text)"
          >
            {{ text }}
          </button>
        </li>
      </ul>
    </div>
  </div>
</template>
