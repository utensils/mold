<script setup lang="ts">
/*
 * Badge pill — small mono status/count pill (spec §03). Tones follow the fixed
 * token jobs: accent (selection), info (halide), success, warning, danger
 * (stop), neutral. Solid tints the tone at 18% behind tone ink; outline keeps
 * a transparent bg with a 1px tone border. Non-interactive — content only.
 */
export type BadgeTone =
  "accent" | "info" | "success" | "warning" | "danger" | "neutral";

withDefaults(
  defineProps<{
    tone?: BadgeTone;
    outline?: boolean;
  }>(),
  { tone: "accent", outline: false },
);
</script>

<template>
  <span
    class="ms-badge"
    :data-tone="tone"
    :data-outline="outline ? 'true' : undefined"
  >
    <slot />
  </span>
</template>

<style scoped>
.ms-badge {
  --ms-badge-tone: var(--safelight);
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-family: var(--f-mono);
  font-size: 10px;
  font-weight: 700;
  line-height: 1.5;
  border: 1px solid transparent;
  border-radius: var(--radius-pill);
  padding: 1px 8px;
  background: color-mix(in srgb, var(--ms-badge-tone) 18%, transparent);
  color: var(--ms-badge-tone);
  white-space: nowrap;
}

/* Accent solid derives from the selection set — never a new mix. */
.ms-badge[data-tone="accent"] {
  background: var(--sel-bg);
  color: var(--sel-ink);
}

.ms-badge[data-tone="info"] {
  --ms-badge-tone: var(--halide);
}

.ms-badge[data-tone="success"] {
  --ms-badge-tone: var(--success);
}

.ms-badge[data-tone="warning"] {
  --ms-badge-tone: var(--warning);
}

.ms-badge[data-tone="danger"] {
  --ms-badge-tone: var(--stop);
}

.ms-badge[data-tone="neutral"] {
  --ms-badge-tone: var(--ink-2);
}

/* Outline wins over the accent solid overrides above (source order). */
.ms-badge[data-outline="true"] {
  background: transparent;
  border-color: var(--ms-badge-tone);
  color: var(--ms-badge-tone);
}
</style>
