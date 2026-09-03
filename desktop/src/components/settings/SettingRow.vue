<script setup lang="ts">
/*
 * One settings row (README §04): a plain label over one line of help on the
 * left, the control on the right, 52px tall, separated by hairlines inside
 * the section panel. Provenance, restart, and lock notes stay mono.
 */
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
  <div
    class="flex min-h-[52px] items-center gap-3.5 border-b border-border px-3.5 py-3 last:border-b-0"
  >
    <div class="flex min-w-0 flex-1 flex-col gap-0.5">
      <div class="flex items-center gap-2">
        <span class="text-sm font-medium text-fg">{{ label }}</span>
        <span
          v-if="source"
          class="font-mono text-micro text-fg-dim"
          :title="`stored in ${provenance(source).label}`"
        >
          {{ provenance(source).glyph }} {{ provenance(source).label.toUpperCase() }}
        </span>
        <span v-if="needsEngineRestart" class="font-mono text-micro text-sapphire"
          >RESTART ENGINE</span
        >
      </div>
      <p v-if="help" class="max-w-md text-xs text-fg-dim">{{ help }}</p>
      <p v-if="locked" class="text-xs text-sapphire">
        <template v-if="lockedReason">{{ lockedReason }}</template>
        <template v-else>
          Locked by <span class="font-mono">{{ lockedBy ?? "the environment" }}</span> — unset it to
          edit here.
        </template>
      </p>
    </div>
    <div class="flex shrink-0 items-center gap-2">
      <slot />
      <button
        v-if="resettable && !locked"
        type="button"
        class="h-[26px] rounded-control px-1.5 text-micro text-fg-dim hover:text-error"
        title="Reset to default"
        aria-label="Reset to default"
        @click="$emit('reset')"
      >
        ↺
      </button>
    </div>
  </div>
</template>
