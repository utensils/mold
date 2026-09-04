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
    tone?: "plain" | "info";
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
      'ms-acc--info': tone === 'info',
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
  background: var(--mold-bg-deep);
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  overflow: hidden;
}

.ms-acc--info {
  background: color-mix(in srgb, var(--mold-sapphire) 3%, var(--mold-bg));
  border-color: color-mix(
    in srgb,
    var(--mold-sapphire) 22%,
    var(--mold-border)
  );
}

.ms-acc--info.ms-acc--open {
  background: color-mix(in srgb, var(--mold-sapphire) 5%, var(--mold-bg));
  border-color: color-mix(
    in srgb,
    var(--mold-sapphire) 34%,
    var(--mold-border)
  );
}

.ms-acc__head {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 13px;
  border: 0;
  background: transparent;
  color: var(--mold-text);
  padding: 15px 16px;
  text-align: left;
  font-family: var(--mold-font-sans);
}

.ms-acc__head--interactive {
  cursor: pointer;
  transition: background var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-acc__head--interactive:hover {
  background: color-mix(in srgb, var(--mold-text) 5%, transparent);
}

.ms-acc--info .ms-acc__head--interactive:hover {
  background: color-mix(in srgb, var(--mold-sapphire) 7%, transparent);
}

.ms-acc__head--interactive:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}

.ms-acc__plate {
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  border-radius: var(--mold-radius-2);
  background: color-mix(in srgb, var(--mold-sapphire) 16%, transparent);
  color: var(--mold-sapphire);
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
  color: var(--mold-text-dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.ms-acc__chevron {
  flex: 0 0 auto;
  color: var(--mold-text-dim);
}

.ms-acc__body {
  border-top: 1px solid var(--mold-border);
  padding: 14px 16px;
}
</style>
