<script setup lang="ts">
/*
 * Clip pill — one clip on the composer rail (mockup 1c): thumbnail well,
 * "Opening clip" / "Clip N" label, frame-count chip, selection ring. The
 * remove affordance appears on hover/focus only when removal is allowed
 * (two-clip floor is the caller's rule).
 */
const props = withDefaults(
  defineProps<{
    label: string;
    frames: number;
    active?: boolean;
    removable?: boolean;
    /** Render plan badge for edit sessions: cached tick or re-render mark. */
    plan?: "cached" | "rerender" | "new" | null;
  }>(),
  { active: false, removable: false, plan: null },
);

const emit = defineEmits<{
  select: [];
  remove: [];
}>();

function onRemove(event: MouseEvent) {
  event.stopPropagation();
  emit("remove");
}
</script>

<template>
  <span
    class="ms-clip"
    :data-on="active ? 'true' : undefined"
    :data-plan="plan ?? undefined"
  >
    <button
      type="button"
      class="ms-clip__body"
      :aria-label="`${label} · ${frames} frames`"
      :aria-current="active ? 'true' : undefined"
      @click="emit('select')"
    >
      <span class="ms-clip__thumb" aria-hidden="true">
        <slot name="thumb" />
        <span
          v-if="plan === 'cached'"
          class="ms-clip__plan ms-clip__plan--cached"
          >✓</span
        >
        <span
          v-else-if="plan === 'rerender'"
          class="ms-clip__plan ms-clip__plan--rerender"
          >↻</span
        >
      </span>
      <span class="ms-clip__label">{{ label }}</span>
      <span class="ms-clip__frames">{{ frames }}f</span>
    </button>
    <button
      v-if="removable"
      type="button"
      class="ms-clip__remove"
      :aria-label="`Remove ${label}`"
      @click="onRemove"
    >
      ✕
    </button>
  </span>
</template>

<style scoped>
.ms-clip {
  position: relative;
  display: inline-flex;
  flex: 0 0 auto;
}

.ms-clip__body {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 4px 12px 4px 4px;
  background: transparent;
  border: 1px solid var(--ce);
  border-radius: var(--radius-pill);
  color: var(--ink-2);
  font-family: var(--f-body);
  font-size: 12px;
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}

.ms-clip__body:hover {
  color: var(--rebate);
  border-color: var(--ink-3);
}

.ms-clip[data-on="true"] .ms-clip__body {
  border-color: var(--sel-border);
  background: var(--sel-bg);
  color: var(--sel-ink);
  box-shadow: var(--sel-ring);
}

.ms-clip__body:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-clip__thumb {
  position: relative;
  display: block;
  width: 34px;
  height: 22px;
  border-radius: 2px;
  border: 1px solid color-mix(in srgb, var(--rebate) 18%, transparent);
  background:
    radial-gradient(
      120% 90% at 30% 20%,
      color-mix(in srgb, var(--halide) 22%, transparent),
      transparent 60%
    ),
    var(--print);
  overflow: hidden;
  flex: 0 0 auto;
}

.ms-clip__thumb :deep(img),
.ms-clip__thumb :deep(video) {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.ms-clip__plan {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  font-size: 11px;
  font-weight: 800;
}

.ms-clip__plan--cached {
  color: var(--success);
  background: color-mix(in srgb, var(--print) 55%, transparent);
}

.ms-clip__plan--rerender {
  color: var(--warning);
  background: color-mix(in srgb, var(--print) 55%, transparent);
}

.ms-clip__frames {
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
}

.ms-clip[data-on="true"] .ms-clip__frames {
  color: var(--sel-ink);
}

.ms-clip__remove {
  position: absolute;
  top: -6px;
  right: -6px;
  width: 18px;
  height: 18px;
  display: none;
  place-items: center;
  padding: 0;
  border: 1px solid var(--ce);
  border-radius: 50%;
  background: var(--bench);
  color: var(--ink-2);
  font-size: 10px;
  cursor: pointer;
}

.ms-clip:hover .ms-clip__remove,
.ms-clip:focus-within .ms-clip__remove {
  display: grid;
}

.ms-clip__remove:hover {
  color: var(--stop);
  border-color: var(--stop);
}
</style>
