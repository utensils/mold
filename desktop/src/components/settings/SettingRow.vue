<script setup lang="ts">
import { provenance } from "../../lib/config";
import type { ConfigSource } from "../../lib/api/types";

defineProps<{
  label: string;
  help?: string | undefined;
  /** Engine-config provenance tag (db/file/env); omit for app-side prefs. */
  source?: ConfigSource | undefined;
  /** Env-resolved rows are locked — the environment wins. */
  locked?: boolean | undefined;
  /** Name of the env var that wins when locked. */
  lockedBy?: string | undefined;
  /** Changing this only takes effect after an engine restart. */
  needsEngineRestart?: boolean | undefined;
  /** Show the reset-to-default affordance. */
  resettable?: boolean | undefined;
}>();

defineEmits<{ (e: "reset"): void }>();
</script>

<template>
  <div class="border-edge flex items-start gap-4 border-b py-3 last:border-b-0">
    <div class="min-w-0 flex-1">
      <div class="flex items-center gap-2">
        <span class="text-body font-medium text-ink">{{ label }}</span>
        <span v-if="source" class="edge-code" :title="`stored in ${provenance(source).label}`">
          {{ provenance(source).glyph }} {{ provenance(source).label.toUpperCase() }}
        </span>
        <span v-if="needsEngineRestart" class="edge-code text-halide">RESTART ENGINE</span>
      </div>
      <p v-if="help" class="mt-0.5 max-w-md text-caption text-ink-3">{{ help }}</p>
      <p v-if="locked" class="mt-0.5 text-caption text-halide">
        Locked by <span class="data-mono">{{ lockedBy ?? "the environment" }}</span> — unset it to
        edit here.
      </p>
    </div>
    <div class="flex shrink-0 items-center gap-2 pt-0.5">
      <slot />
      <button
        v-if="resettable && !locked"
        type="button"
        class="h-7 rounded-control px-1.5 text-caption text-ink-3 hover:text-stop"
        title="Reset to default"
        @click="$emit('reset')"
      >
        ↺
      </button>
    </div>
  </div>
</template>
