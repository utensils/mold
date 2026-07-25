<script setup lang="ts">
/*
 * Accordion section — the disclosure primitive for Advanced panels (spec §07).
 * Bordered bath container; header row = optional halide icon plate + title
 * with one-line summary + trailing chevron. With `headerInteractive: false`
 * the header is a plain div (no chevron) for rows whose action lives in a
 * trailing control (e.g. an upscale switch) via the "action" slot.
 */
import { useId } from "vue";
import Icon from "./Icon.vue";
import type { IconName } from "../icons";

const props = withDefaults(
  defineProps<{
    open: boolean;
    title: string;
    summary?: string;
    icon?: IconName;
    /** Optional accent treatment for contexts that need stronger visual grouping. */
    tone?: "plain" | "halide";
    /** When false the header is a plain div and no chevron renders. */
    headerInteractive?: boolean;
  }>(),
  { headerInteractive: true, tone: "plain" },
);

const emit = defineEmits<{ toggle: [] }>();

const bodyId = useId();

function onHeaderClick() {
  if (props.headerInteractive) emit("toggle");
}
</script>

<template>
  <section
    class="ms-acc"
    :class="{
      'ms-acc--halide': tone === 'halide',
      'ms-acc--open': open,
    }"
  >
    <component
      :is="headerInteractive ? 'button' : 'div'"
      class="ms-acc__head"
      :class="{ 'ms-acc__head--interactive': headerInteractive }"
      :type="headerInteractive ? 'button' : undefined"
      :aria-expanded="headerInteractive ? open : undefined"
      :aria-controls="headerInteractive ? bodyId : undefined"
      @click="onHeaderClick"
    >
      <span v-if="icon" class="ms-acc__plate" aria-hidden="true">
        <Icon :name="icon" :size="17" />
      </span>
      <span class="ms-acc__text">
        <span class="ms-acc__title">{{ title }}</span>
        <span v-if="summary" class="ms-acc__summary">{{ summary }}</span>
      </span>
      <slot name="action" />
      <Icon
        v-if="headerInteractive"
        class="ms-acc__chevron"
        :name="open ? 'chevron-up' : 'chevron-down'"
        :size="16"
      />
    </component>
    <div v-if="open" :id="bodyId" class="ms-acc__body">
      <slot />
    </div>
  </section>
</template>

<style scoped>
.ms-acc {
  background: var(--bath);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card);
  overflow: hidden;
}

.ms-acc--halide {
  background: color-mix(in srgb, var(--halide) 3%, var(--bench));
  border-color: color-mix(in srgb, var(--halide) 22%, var(--edge));
  box-shadow: inset 0 1px 0 var(--card-hi);
}

.ms-acc--halide.ms-acc--open {
  background: color-mix(in srgb, var(--halide) 5%, var(--bench));
  border-color: color-mix(in srgb, var(--halide) 34%, var(--edge));
}

.ms-acc__head {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 13px;
  border: 0;
  background: transparent;
  color: var(--rebate);
  padding: 15px 16px;
  text-align: left;
  font-family: var(--f-body);
}

.ms-acc__head--interactive {
  cursor: pointer;
  transition: background var(--dur-quick) var(--ease);
}

.ms-acc__head--interactive:hover {
  background: color-mix(in srgb, var(--rebate) 5%, transparent);
}

.ms-acc--halide .ms-acc__head--interactive:hover {
  background: color-mix(in srgb, var(--halide) 7%, transparent);
}

.ms-acc__head--interactive:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-acc__plate {
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  border-radius: var(--radius-control);
  background: color-mix(in srgb, var(--halide) 16%, transparent);
  color: var(--halide);
  display: flex;
  align-items: center;
  justify-content: center;
}

.ms-acc__text {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.ms-acc__title {
  font-size: 13.5px;
  font-weight: 600;
}

.ms-acc__summary {
  font-size: 11px;
  color: var(--ink-3);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.ms-acc__chevron {
  flex: 0 0 auto;
  color: var(--ink-3);
}

.ms-acc__body {
  border-top: 1px solid var(--edge);
  padding: 14px 16px;
}
</style>
