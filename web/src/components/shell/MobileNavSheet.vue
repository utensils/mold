<script setup lang="ts">
/* Compact browser navigation: real links preserve open-in-new-tab and
 * history, while the shared overlay hook owns focus and Escape. */
import { computed, ref, toRef } from "vue";
import { useRoute } from "vue-router";
import SheetPanel from "@ui/components/SheetPanel.vue";
import Icon from "@ui/components/Icon.vue";
import { WORKSPACES, SETTINGS_DESTINATION } from "../../lib/workspaces";
import { useOverlayFocus } from "../../composables/useOverlayFocus";

const props = defineProps<{ open: boolean }>();
const emit = defineEmits<{ close: [] }>();
const host = ref<HTMLElement | { $el?: unknown } | null>(null);
const { onKeydown } = useOverlayFocus(toRef(props, "open"), host, () =>
  emit("close"),
);

const route = useRoute();

const destinations = [...WORKSPACES, SETTINGS_DESTINATION];

const activeName = computed(() => String(route.name ?? ""));
</script>

<template>
  <SheetPanel
    ref="host"
    :open="open"
    variant="bottom"
    title="Navigation"
    @close="emit('close')"
    @keydown="onKeydown"
  >
    <nav class="flex flex-col gap-0.5" aria-label="Navigation">
      <router-link
        v-for="dest in destinations"
        :key="dest.name"
        :to="dest.path"
        class="mobile-workspace-link"
        :aria-current="dest.match.includes(activeName) ? 'page' : undefined"
        :data-test="`mobile-nav-${dest.name}`"
        @click="emit('close')"
      >
        <Icon :name="dest.icon" :size="20" />
        {{ dest.label }}
      </router-link>
    </nav>
  </SheetPanel>
</template>

<style scoped>
.mobile-workspace-link {
  display: flex;
  align-items: center;
  gap: 12px;
  min-height: 48px;
  padding: 12px;
  color: var(--mold-text-2);
  border-radius: var(--mold-radius-2);
  font-size: 1rem;
  text-decoration: none;
}
.mobile-workspace-link[aria-current="page"] {
  background: var(--mold-accent-tint);
  box-shadow: inset 0 0 0 1px var(--mold-blue);
  color: var(--mold-text);
}
</style>
