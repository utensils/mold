<script setup lang="ts">
/*
 * Navigation item — sidebar row (default) or phone tab-bar column
 * (variant="tab"). Active row gets the selection tint; active tab lights the
 * accent. Optional count badge (top-right) for workspace activity (G11).
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
    /** Optional count badge shown top-right. */
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
    @click="emit('select')"
  >
    <Icon class="ms-nav__icon" :name="icon" :size="isTab ? 22 : 17" />
    <span v-if="showLabel" class="ms-nav__label">{{ label }}</span>
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
  font-family: var(--f-body);
  transition:
    background var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}

.ms-nav:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-nav__icon {
  flex: 0 0 auto;
}

/* ── Row (sidebar) ─────────────────────────────────────────────── */
.ms-nav--row {
  width: 100%;
  height: 37px;
  gap: 11px;
  padding: 0 12px;
  border-radius: var(--radius-control);
  color: var(--ink-2);
  font-size: 13.5px;
  text-align: left;
}

.ms-nav--row:hover {
  color: var(--rebate);
}

.ms-nav--row[data-on="true"] {
  background: var(--sel-bg);
  color: var(--rebate);
}

.ms-nav--row[data-on="true"] .ms-nav__icon {
  color: var(--sel-ink);
}

.ms-nav--row .ms-nav__label {
  font-weight: 600;
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
  border-radius: var(--radius-control-sm);
  color: var(--ink-3);
}

.ms-nav--tab[data-on="true"] {
  color: var(--safelight);
}

.ms-nav--tab .ms-nav__label {
  font-family: var(--f-mono);
  font-size: 10px;
}

/* ── Badge ─────────────────────────────────────────────────────── */
.ms-nav__badge {
  position: absolute;
  top: 3px;
  right: 8px;
  min-width: 15px;
  height: 15px;
  padding: 0 4px;
  border-radius: var(--radius-pill);
  background: var(--safelight);
  color: var(--on-accent);
  font-family: var(--f-mono);
  font-size: 9px;
  font-weight: 700;
  line-height: 15px;
  text-align: center;
}

.ms-nav--collapsed .ms-nav__badge {
  right: 4px;
}

.ms-nav--tab .ms-nav__badge {
  top: 0;
  right: calc(50% - 20px);
}
</style>
