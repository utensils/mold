<script setup lang="ts">
import SettingRow from "@studio/components/settings/SettingRow.vue";
import ToggleControl from "@studio/components/settings/ToggleControl.vue";
import EngineRow from "./EngineRow.vue";
import { useLibraryPrefsStore } from "../../stores/libraryPrefs";

/**
 * Settings ▸ My images & trash.
 *
 * Two different kinds of setting live here and the split is deliberate. Trash
 * retention is the ENGINE's (`gallery.trash_retention_days` on the primary's
 * `/api/config`), and remote machines keep their own — Machines ▸ machine ▸
 * Storage edits those. "Tag new prints with their title" is a property of this
 * install's Create form, so it stays on this side of the wire and reaches no
 * host at all.
 *
 * The app's own auto-tag switch and the engine's `generate.auto_tag_title` sit
 * ADJACENT, in that order: the engine key governs what `mold run` puts in a
 * request, and its help sentence points at "the switch above". Separate them
 * and the sentence lies.
 */
const libraryPrefs = useLibraryPrefsStore();
</script>

<template>
  <div data-test="library-section">
    <EngineRow schema-key="gallery.trash_retention_days" />
    <SettingRow
      label="Tag new prints with their title"
      help="A titled print picks up its own slug as a tag — shown as a removable chip in Create before you generate. Never changes prints you already made."
    >
      <ToggleControl
        :model-value="libraryPrefs.autoTagTitle"
        aria-label="Tag new prints with their title"
        data-test="library-auto-tag-title"
        @commit="libraryPrefs.setAutoTagTitle($event)"
      />
    </SettingRow>
    <EngineRow schema-key="generate.auto_tag_title" />
    <!-- A footnote after the last row: inset like a row, because the card
         itself has no padding and the rows are full-bleed. -->
    <p class="px-3.5 py-3 text-micro text-fg-dim" data-test="library-remote-note">
      Remote machines keep their own retention — change it in Machines ▸ machine ▸ Storage.
    </p>
  </div>
</template>
