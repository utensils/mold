<script setup lang="ts">
import { useCatalog } from "../composables/useCatalog";

const cat = useCatalog();
</script>

<template>
  <aside class="w-60 border-r border-zinc-800 p-3 overflow-y-auto">
    <h2 class="text-xs uppercase text-zinc-500 mb-2">Families</h2>
    <ul class="text-sm">
      <li
        v-for="row in cat.families.value"
        :key="row.family"
        @click="cat.setFilter({ family: row.family })"
        class="cursor-pointer py-1 px-2 rounded hover:bg-zinc-800"
        :class="{ 'bg-zinc-800': cat.filter.value.family === row.family }"
      >
        <div class="font-medium">{{ row.family }}</div>
        <div class="text-[11px] text-zinc-500">
          {{ row.foundation }} foundation · {{ row.finetune }} fine-tunes
        </div>
      </li>
    </ul>
  </aside>
</template>
