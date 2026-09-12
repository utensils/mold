<script setup lang="ts">
/*
 * One settings row: a plain label over one line of help on the left, the
 * control on the right, 52px tall, separated by hairlines inside the section
 * panel. Provenance, restart, and lock notes stay mono, because they are
 * facts about the machine rather than prose.
 */
import { provenance, type ConfigSource } from "../../api/config";

defineProps<{
  label: string;
  help?: string | undefined;
  /** Engine-config provenance tag (db/file/env); omit for app-side prefs. */
  source?: ConfigSource | undefined;
  /** Why this row cannot be edited here; its presence is the lock. */
  lockedReason?: string | undefined;
  /** Changing this only takes effect after an engine restart. */
  needsEngineRestart?: boolean | undefined;
  /**
   * Changing this only takes effect after the whole APP restarts — the value
   * is frozen per process by `runtime_env`, and the desktop engine is a thread
   * inside this process, so restarting the engine cannot pick it up.
   */
  needsAppRestart?: boolean | undefined;
  /** Show the reset-to-default affordance. */
  resettable?: boolean | undefined;
}>();

defineEmits<{ (e: "reset"): void }>();
</script>

<template>
  <div class="ms-setting-row">
    <div class="ms-setting-row__text">
      <div class="ms-setting-row__head">
        <span class="ms-setting-row__label">{{ label }}</span>
        <span
          v-if="source"
          class="ms-setting-row__note"
          :title="`stored in ${provenance(source).label}`"
        >
          {{ provenance(source).glyph }}
          {{ provenance(source).label.toUpperCase() }}
        </span>
        <span
          v-if="needsAppRestart"
          class="ms-setting-row__note ms-setting-row__note--restart"
        >
          RESTART APP
        </span>
        <span
          v-else-if="needsEngineRestart"
          class="ms-setting-row__note ms-setting-row__note--restart"
        >
          RESTART ENGINE
        </span>
      </div>
      <p v-if="help" class="ms-setting-row__help">{{ help }}</p>
      <p v-if="lockedReason" class="ms-setting-row__locked">
        {{ lockedReason }}
      </p>
    </div>
    <div class="ms-setting-row__control">
      <slot />
      <button
        v-if="resettable && !lockedReason"
        type="button"
        class="ms-setting-row__reset"
        data-test="setting-reset"
        title="Reset to default"
        aria-label="Reset to default"
        @click="$emit('reset')"
      >
        ↺
      </button>
    </div>
  </div>
</template>

<style scoped>
.ms-setting-row {
  display: flex;
  align-items: center;
  gap: var(--mold-sp-3);
  box-sizing: border-box;
  min-height: var(--mold-row-h-table, 52px);
  padding: var(--mold-sp-2) var(--mold-sp-3);
}
/* Hairlines sit BETWEEN rows, so a panel that ends with something other than
 * a row (a meter, an empty-state sentence) never leaves a stray rule. */
.ms-setting-row + .ms-setting-row {
  border-top: var(--mold-bw) solid var(--mold-border);
}
.ms-setting-row__text {
  display: flex;
  flex: 1 1 auto;
  min-width: 0;
  flex-direction: column;
  gap: 2px;
}
.ms-setting-row__head {
  display: flex;
  align-items: center;
  gap: var(--mold-sp-2);
}
.ms-setting-row__label {
  color: var(--mold-text);
  font-size: var(--mold-fs-sm);
  font-weight: 500;
}
.ms-setting-row__note {
  color: var(--mold-text-dim);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  white-space: nowrap;
}
.ms-setting-row__note--restart {
  color: var(--mold-sapphire);
}
.ms-setting-row__help {
  max-width: 448px;
  margin: 0;
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-xs);
}
.ms-setting-row__locked {
  margin: 0;
  color: var(--mold-sapphire);
  font-size: var(--mold-fs-xs);
}
.ms-setting-row__control {
  display: flex;
  flex: none;
  align-items: center;
  gap: var(--mold-sp-2);
}
.ms-setting-row__reset {
  height: var(--mold-ctl-md, 26px);
  padding: 0 var(--mold-sp-1);
  border: none;
  border-radius: var(--mold-radius-2);
  background: none;
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-micro);
  cursor: pointer;
}
.ms-setting-row__reset:hover {
  color: var(--mold-error);
}

/* A phone has no room for a label beside a field: the control drops under the
 * text and takes the row's width, and the words keep their line length. */
@media (max-width: 639px) {
  .ms-setting-row {
    flex-wrap: wrap;
  }
  .ms-setting-row__text {
    flex-basis: 100%;
  }
  .ms-setting-row__control {
    max-width: 100%;
    margin-left: auto;
  }
}
</style>
