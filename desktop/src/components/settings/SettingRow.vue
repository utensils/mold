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
  lockedReason?: string | undefined;
  /** Changing this only takes effect after an engine restart. */
  needsEngineRestart?: boolean | undefined;
  /** Show the reset-to-default affordance. */
  resettable?: boolean | undefined;
}>();

defineEmits<{ (e: "reset"): void }>();
</script>

<template>
  <div class="border-border flex items-start gap-4 border-b py-3 last:border-b-0">
    <div class="min-w-0 flex-1">
      <div class="flex items-center gap-2">
        <span class="text-sm font-medium text-fg">{{ label }}</span>
        <span
          v-if="source"
          class="font-mono text-micro text-fg-dim whitespace-nowrap"
          :title="`stored in ${provenance(source).label}`"
        >
          {{ provenance(source).glyph }} {{ provenance(source).label.toUpperCase() }}
        </span>
        <span
          v-if="needsEngineRestart"
          class="font-mono text-micro text-fg-dim whitespace-nowrap text-sapphire"
          >RESTART ENGINE</span
        >
      </div>
      <p v-if="help" class="mt-0.5 max-w-md text-micro text-fg-dim">{{ help }}</p>
      <p v-if="locked" class="mt-0.5 text-micro text-sapphire">
        <template v-if="lockedReason">{{ lockedReason }}</template>
        <template v-else>
          Locked by <span class="font-mono text-xs">{{ lockedBy ?? "the environment" }}</span> —
          unset it to edit here.
        </template>
      </p>
    </div>
    <div class="flex shrink-0 items-center gap-2 pt-0.5">
      <slot />
      <button
        v-if="resettable && !locked"
        type="button"
        class="h-7 rounded-control px-1.5 text-micro text-fg-dim hover:text-error"
        title="Reset to default"
        aria-label="Reset to default"
        @click="$emit('reset')"
      >
        ↺
      </button>
    </div>
  </div>
</template>
