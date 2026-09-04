<script setup lang="ts">
/*
 * Navigation item — sidebar row (default) or phone tab-bar column
 * (variant="tab"). A row is 36px with an 18px icon and a sans label; the
 * selected row takes the accent tint plus a 1px inset accent ring (README
 * §06). The trailing slot carries a mono count, keycap, or status dot; the
 * `badge` prop is the accent-filled count for work in progress.
 */
import { computed } from "vue";
import Icon from "./Icon.vue";
import type { IconName } from "../icons";

const props = withDefaults(
  defineProps<{
    icon: IconName;
    label: string;
    active?: boolean;
    /** Row variant only: center the icon and hide the label. */
    collapsed?: boolean;
    variant?: "row" | "tab";
    /** Accent-filled count for work in progress. */
    badge?: string | number;
  }>(),
  { active: false, collapsed: false, variant: "row" },
);

const emit = defineEmits<{ select: [] }>();

const isTab = computed(() => props.variant === "tab");
const showLabel = computed(() => isTab.value || !props.collapsed);
const hasBadge = computed(
  () => props.badge !== undefined && props.badge !== "",
);
</script>

<template>
  <button
    type="button"
    class="ms-nav"
    :class="[
      isTab ? 'ms-nav--tab' : 'ms-nav--row',
      { 'ms-nav--collapsed': !isTab && collapsed },
    ]"
    :data-on="active ? 'true' : undefined"
    :aria-current="active ? 'page' : undefined"
    :aria-label="showLabel ? undefined : label"
    :title="showLabel ? undefined : label"
    @click="emit('select')"
  >
    <Icon class="ms-nav__icon" :name="icon" :size="isTab ? 22 : 18" />
    <span v-if="showLabel" class="ms-nav__label">{{ label }}</span>
    <span v-if="showLabel && $slots.trailing" class="ms-nav__trailing">
      <slot name="trailing" />
    </span>
    <span v-if="hasBadge" class="ms-nav__badge">{{ badge }}</span>
  </button>
</template>

<style scoped>
.ms-nav {
  position: relative;
  display: flex;
  align-items: center;
  border: 0;
  background: transparent;
  cursor: pointer;
  font-family: var(--mold-font-sans);
  transition:
    background var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-nav:focus-visible {
  outline: 2px solid var(--mold-border-focus);
  outline-offset: 1px;
}

.ms-nav__icon {
  flex: 0 0 auto;
}

/* ── Row (sidebar) ─────────────────────────────────────────────── */
.ms-nav--row {
  width: 100%;
  min-height: var(--mold-row-h, 36px);
  gap: 10px;
  padding: 0 9px;
  border-radius: var(--mold-radius-2);
  color: var(--mold-text-2);
  font-size: var(--mold-fs-sm);
  font-weight: 500;
  text-align: left;
}

.ms-nav--row:hover {
  background: var(--mold-row-hover, var(--mold-surface));
  color: var(--mold-text);
}

.ms-nav--row[data-on="true"] {
  background: var(--mold-accent-tint);
  box-shadow: inset 0 0 0 1px var(--mold-blue);
  color: var(--mold-text);
}

.ms-nav__label {
  min-width: 0;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ms-nav__trailing {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 6px;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}

.ms-nav--collapsed {
  justify-content: center;
  padding: 0;
}

/* ── Tab (phone tab bar) ───────────────────────────────────────── */
.ms-nav--tab {
  flex-direction: column;
  justify-content: center;
  gap: 3px;
  padding: 4px 0;
  border-radius: var(--mold-radius-1);
  color: var(--mold-text-dim);
}

.ms-nav--tab[data-on="true"] {
  color: var(--mold-blue);
}

.ms-nav--tab .ms-nav__label {
  flex: 0 0 auto;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
}

/* ── Badge: accent count for work in progress ──────────────────── */
.ms-nav__badge {
  display: inline-flex;
  flex: 0 0 auto;
  align-items: center;
  justify-content: center;
  min-width: 19px;
  height: 17px;
  padding: 0 6px;
  border-radius: var(--mold-radius-2);
  background: var(--mold-blue);
  color: var(--mold-on-accent);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  font-weight: 700;
  line-height: 1;
}

.ms-nav--collapsed .ms-nav__badge,
.ms-nav--tab .ms-nav__badge {
  position: absolute;
  top: 3px;
  right: 4px;
  min-width: 15px;
  height: 15px;
  padding: 0 4px;
  font-size: 9px;
}

.ms-nav--tab .ms-nav__badge {
  top: 0;
  right: calc(50% - 20px);
}
</style>
