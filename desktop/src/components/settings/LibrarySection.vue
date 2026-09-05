<script setup lang="ts">
import ConfigSettingRow from "./ConfigSettingRow.vue";
import SettingRow from "./SettingRow.vue";
import ToggleControl from "./ToggleControl.vue";
import { useLibraryPrefsStore } from "../../stores/libraryPrefs";

/**
 * Settings ▸ Library.
 *
 * Two different kinds of setting live here and the split is deliberate. Trash
 * retention is the ENGINE's (`gallery.trash_retention_days` on the primary's
 * `/api/config`), and remote machines keep their own — Machines ▸ machine ▸
 * Storage edits those. "Tag new prints with their title" is a property of this
 * install's Create form, so it stays on this side of the wire and reaches no
 * host at all.
 */
const libraryPrefs = useLibraryPrefsStore();
</script>

<template>
  <div data-test="library-section">
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
    <ConfigSettingRow schema-key="gallery.trash_retention_days" />
    <!-- A footnote after the last row: inset like a row, because the card
         itself has no padding and the rows are full-bleed. -->
    <p class="px-3.5 py-3 text-micro text-fg-dim" data-test="library-remote-note">
      Remote machines keep their own retention — change it in Machines ▸ machine ▸ Storage.
    </p>
  </div>
</template>
