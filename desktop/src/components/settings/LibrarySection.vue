<script setup lang="ts">
import ConfigSettingRow from "./ConfigSettingRow.vue";

/**
 * Settings ▸ Library: the engine's trash retention for THIS device
 * (`gallery.trash_retention_days` on the primary's `/api/config`). Remote
 * machines keep their own retention — Machines ▸ machine ▸ Storage edits it
 * per host — so this section never pretends to be fleet-wide.
 */
</script>

<template>
  <div data-test="library-section">
    <ConfigSettingRow schema-key="gallery.trash_retention_days" />
    <p class="mt-2 text-caption text-ink-3" data-test="library-remote-note">
      Remote machines keep their own retention — change it in Machines ▸ machine ▸ Storage.
    </p>
  </div>
</template>
