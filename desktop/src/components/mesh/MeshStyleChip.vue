<script setup lang="ts">
import { computed } from "vue";
import Icon from "@ui/components/Icon.vue";
import { familyLabel } from "@studio/lib/modelFamily";
import ModelPicker from "../create/ModelPicker.vue";
import { modelDisplayName } from "../../lib/models";
import type { ModelEntry } from "../../lib/api/types";

/**
 * A style chip on the 3-D Studio's composer — the peer of New image's
 * `StylePicker`, and deliberately the same control: the chip IS the picker,
 * opening `ModelPicker`'s menu upward in place. A chip that was only a door
 * onto a second, fuller selector is what made one of the two read as broken
 * (`.claude/rules/desktop.md`).
 *
 * The picker itself is Generate's, unchanged — its search, family groups,
 * availability and friendly labels — with candidates narrowed by the caller
 * through `outputKindForModel`. There is no second selector.
 */
const props = defineProps<{
  models: ModelEntry[];
  selected: ModelEntry | null;
  /** The plain word for what this chip picks: "3-D style" / "Picture style". */
  label: string;
  /**
   * What the chip says with nothing picked yet.
   *
   * Authored, never derived: lowercasing the label produced "Choose a 3-d
   * style", and "3-D" is the lexicon's spelling wherever a kind is named.
   */
  placeholder: string;
  /** The menu's mono kicker, and where its Browse more goes. */
  kicker: string;
  browseTarget: string;
  disabledReason?: string | null;
}>();

const emit = defineEmits<{ pick: [model: ModelEntry] }>();

/*
 * Plain words in sans, technical truth in mono, on the same row — the display
 * name, falling back to the family's friendly label when the name is just the
 * raw id, exactly as `StylePicker` resolves it.
 */
const styleLabel = computed(() => {
  const model = props.selected;
  if (!model) return "";
  const display = modelDisplayName(model);
  return display && display !== model.name ? display : (familyLabel(model.family) ?? model.name);
});
const styleId = computed(() => props.selected?.name ?? "");
const testId = computed(
  () => `mesh-style-chip-${props.label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
);
</script>

<template>
  <ModelPicker
    class="ms-style"
    :models="models"
    :selected="selected"
    :kicker="kicker"
    :browse-target="browseTarget"
    :disabled-reason="disabledReason ? () => disabledReason ?? null : null"
    placement="up"
    @pick="emit('pick', $event)"
  >
    <template #trigger="{ open, toggle }">
      <button
        type="button"
        :data-test="testId"
        class="ms-chip ms-chip--style"
        :aria-expanded="open"
        aria-haspopup="listbox"
        :title="label"
        :aria-label="label"
        @click="toggle"
      >
        <Icon name="layers" :size="13" />
        <span class="ms-chip__label">{{ styleLabel || placeholder }}</span>
        <span v-if="styleId && styleId !== styleLabel" class="ms-chip__id">{{ styleId }}</span>
        <span class="ms-chip__caret">▼</span>
      </button>
    </template>
  </ModelPicker>
</template>

<style scoped>
/*
 * The chip's look lives here, not in the composer's sheet: slot content is
 * compiled in the PARENT and never inherits the child's scoped CSS. That is
 * also why `.ms-chip` is deliberately absent from `ui/kit.css` — `ui/
 * components/Chip.vue` already owns that name for the 24px filter chip.
 */
.ms-style {
  flex-shrink: 0;
}
.ms-chip {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 28px;
  padding: 0 10px;
  flex-shrink: 0;
  white-space: nowrap;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-2);
  cursor: pointer;
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-chip:hover {
  border-color: var(--mold-border-focus);
  color: var(--mold-text);
}
.ms-chip--style {
  background: var(--mold-surface);
  color: var(--mold-text);
  font-weight: 500;
}
.ms-chip__label {
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.ms-chip__id {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}
.ms-chip__caret {
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
}
</style>
