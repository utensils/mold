<script setup lang="ts">
/*
 * ⌘K command palette (spec §06). Presentational PalettePanel + the command
 * registry: navigation, quick actions, theme, and installed models (loaded
 * lazily each time the palette opens). App owns the open/toggle keybinding;
 * this component owns the query and the filtered item list.
 */
import { computed, ref, watch } from "vue";
import { useRouter } from "vue-router";
import PalettePanel from "@ui/components/PalettePanel.vue";
import {
  baseCommands,
  filterCommands,
  modelCommands,
  type Command,
  type CommandContext,
} from "../../lib/commands";
import { fetchModels } from "../../api";
import { useGenerateForm } from "../../composables/useGenerateForm";
import type { ModelInfoExtended } from "../../types";

const props = defineProps<{ open: boolean }>();
const emit = defineEmits<{ close: [] }>();

const router = useRouter();
const form = useGenerateForm();

const query = ref("");
const models = ref<ModelInfoExtended[]>([]);

const ctx: CommandContext = {
  go: (path) => {
    void router.push(path);
  },
  runModel: (name) => {
    form.state.value.model = name;
    void router.push("/create");
  },
  openDownloads: () => {
    window.dispatchEvent(new CustomEvent("mold:open-downloads"));
  },
  newPrint: () => {
    void router.push("/create");
    window.dispatchEvent(new CustomEvent("mold:new-print"));
  },
};

const commands = computed<Command[]>(() => [
  ...baseCommands(ctx),
  ...modelCommands(models.value, ctx),
]);

const filtered = computed(() => filterCommands(commands.value, query.value));

const items = computed(() =>
  filtered.value.map((c) => ({ id: c.id, section: c.section, label: c.label })),
);

function run(id: string) {
  const command = commands.value.find((c) => c.id === id);
  if (command) command.run();
  emit("close");
}

watch(
  () => props.open,
  async (open) => {
    if (!open) return;
    query.value = "";
    try {
      models.value = await fetchModels();
    } catch {
      // Palette still works with navigation/action/theme commands offline.
    }
  },
);
</script>

<template>
  <PalettePanel
    :open="open"
    :query="query"
    :items="items"
    empty-text="No matches"
    @close="emit('close')"
    @update:query="query = $event"
    @run="run"
  />
</template>
