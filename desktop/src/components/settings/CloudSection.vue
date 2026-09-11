<script setup lang="ts">
/*
 * Settings ▸ Cloud GPUs: the rented-machine credentials and defaults —
 * every `runpod.*` and `lambda.*` key the engine reports.
 *
 * These seventeen keys were most of the old Advanced dump, labelled by raw key
 * under "Server-provided configuration key.". They are curated now, so the
 * section asks the store which of them this engine actually has rather than
 * naming them a second time here: a key the engine does not report draws no
 * row at all.
 */
import EngineRow from "./EngineRow.vue";
import { useSettingsConfigStore } from "../../stores/settingsConfig";

const config = useSettingsConfigStore();
</script>

<template>
  <div>
    <EngineRow
      v-for="row in config.rowsForSection('cloud')"
      :key="row.key"
      :schema-key="row.key"
      data-test="cloud-row"
    />
    <!-- Inset like a row: the section card has no padding of its own. -->
    <p
      v-if="config.rowsForSection('cloud').length === 0"
      class="max-w-md px-3.5 py-3 text-micro text-fg-dim"
      data-test="cloud-empty"
    >
      This engine reports no rented-GPU settings. Rent one from Machines and they appear here.
    </p>
  </div>
</template>
