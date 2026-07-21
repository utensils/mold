<script setup lang="ts">
/*
 * Empty state — dashed frame plus one line of what-to-do (spec Empty state,
 * prototype generate canvas). Centered column: 120px dashed frame around a
 * glyph, display headline (optional gradient brand treatment), guidance
 * line, and at most one action in the named "action" slot.
 */
import Icon from "./Icon.vue";
import type { IconName } from "../icons";

withDefaults(
  defineProps<{
    icon?: IconName;
    headline: string;
    guidance: string;
    /** Apply the gradient brand treatment to the headline. */
    brand?: boolean;
  }>(),
  { icon: "image", brand: false },
);
</script>

<template>
  <div class="ms-empty">
    <div class="ms-empty__frame">
      <Icon :name="icon" :size="40" :stroke-width="1.3" />
    </div>
    <h3
      class="ms-empty__headline"
      :class="{ 'ms-empty__headline--brand': brand }"
    >
      {{ headline }}
    </h3>
    <p class="ms-empty__guidance">{{ guidance }}</p>
    <div v-if="$slots.action" class="ms-empty__action">
      <slot name="action" />
    </div>
  </div>
</template>

<style scoped>
.ms-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.ms-empty__frame {
  width: 120px;
  height: 120px;
  margin-bottom: 22px;
  border: 1.5px dashed var(--ce);
  /* Frame radius sits between card and card-lg, per the prototype. */
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--ink-3);
}

.ms-empty__headline {
  margin: 0 0 7px;
  font-family: var(--f-display);
  font-size: 20px;
  font-weight: 600;
  color: var(--rebate);
}

.ms-empty__headline--brand {
  background-image: var(--grad);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.ms-empty__guidance {
  margin: 0;
  max-width: 340px;
  font-size: 13.5px;
  line-height: 1.5;
  color: var(--ink-3);
}

.ms-empty__action {
  margin-top: 18px;
}
</style>
