<script setup lang="ts">
/*
 * Phone-web navigation sheet (spec §04, prototype m-web nav sheet). A bottom
 * SheetPanel of the five destinations — Gallery, Create, Models, Machines,
 * Settings — each a NavItem row. Tapping a row navigates and closes. Renders
 * inside the app frame via SheetPanel's absolute overlay.
 */
import { computed } from "vue";
import { useRoute, useRouter } from "vue-router";
import SheetPanel from "@ui/components/SheetPanel.vue";
import NavItem from "@ui/components/NavItem.vue";
import type { IconName } from "@ui/icons";

defineProps<{ open: boolean }>();
const emit = defineEmits<{ close: [] }>();

const route = useRoute();
const router = useRouter();

interface Destination {
  name: string;
  label: string;
  icon: IconName;
  path: string;
  /** Route names that light this row up. */
  match: string[];
}

const destinations: Destination[] = [
  {
    name: "gallery",
    label: "Gallery",
    icon: "library",
    path: "/",
    match: ["gallery"],
  },
  {
    name: "create",
    label: "Create",
    icon: "create",
    path: "/create",
    match: ["create"],
  },
  {
    name: "models",
    label: "Models",
    icon: "models",
    path: "/models",
    match: ["models"],
  },
  {
    name: "machines",
    label: "Machines",
    icon: "machines",
    path: "/machines",
    match: ["machines", "machine-detail"],
  },
  {
    name: "settings",
    label: "Settings",
    icon: "settings",
    path: "/settings",
    match: ["settings"],
  },
];

const activeName = computed(() => String(route.name ?? ""));

function go(dest: Destination) {
  if (!dest.match.includes(activeName.value)) void router.push(dest.path);
  emit("close");
}
</script>

<template>
  <SheetPanel :open="open" variant="bottom" title="" @close="emit('close')">
    <nav class="flex flex-col gap-0.5" aria-label="Navigation">
      <NavItem
        v-for="dest in destinations"
        :key="dest.name"
        :icon="dest.icon"
        :label="dest.label"
        :active="dest.match.includes(activeName)"
        :data-test="`mobile-nav-${dest.name}`"
        @select="go(dest)"
      />
    </nav>
  </SheetPanel>
</template>
