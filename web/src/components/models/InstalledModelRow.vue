<script setup lang="ts">
/*
 * Installed model row (spec WEB MODELS Installed, prototype lines 1596-1600):
 * a halide icon plate, the mono model name with an outline "★ loaded" badge,
 * a family · size line, and a chevron. The whole row is one button that opens
 * the model detail drawer.
 */
import ModelMetadataBadges from "@studio/components/ModelMetadataBadges.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import Icon from "@ui/components/Icon.vue";
import type { ModelInfoExtended } from "../../types";
import { modelDisplayName } from "@studio/lib/modelDisplay";
import { formatGB } from "../../util/format";

const props = defineProps<{ model: ModelInfoExtended }>();
const emit = defineEmits<{ open: [] }>();
</script>

<template>
  <button
    type="button"
    class="row"
    data-test="installed-row"
    @click="emit('open')"
  >
    <span class="row__plate">
      <Icon name="models" :size="19" />
    </span>
    <span class="row__body">
      <span class="row__head">
        <span class="row__name">{{ modelDisplayName(props.model) }}</span>
        <ModelMetadataBadges
          :kind="props.model.kind"
          :family="props.model.family"
          :nsfw="props.model.nsfw ?? false"
          :show-modality="false"
        />
        <BadgePill
          v-if="props.model.is_loaded"
          tone="accent"
          outline
          data-test="loaded-badge"
        >
          ★ loaded
        </BadgePill>
      </span>
      <span class="row__meta">
        {{ props.model.family }} ·
        {{ formatGB(props.model.size_gb * 1_000_000_000) }}
      </span>
    </span>
    <Icon name="chevron-right" :size="16" class="row__chevron" />
  </button>
</template>

<style scoped>
.row {
  display: flex;
  align-items: center;
  gap: 14px;
  width: 100%;
  text-align: left;
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card);
  padding: 14px 16px;
  color: var(--rebate);
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    background var(--dur-quick) var(--ease);
}

.row:hover {
  background: color-mix(in srgb, var(--rebate) 4%, var(--bench));
  border-color: var(--ce);
}

.row:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.row__plate {
  width: 40px;
  height: 40px;
  flex: 0 0 40px;
  border-radius: 9px;
  background: color-mix(in srgb, var(--halide) 18%, transparent);
  color: var(--halide);
  display: flex;
  align-items: center;
  justify-content: center;
}

.row__body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.row__head {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 9px;
  min-width: 0;
}

.row__name {
  flex: 1 1 160px;
  font-family: var(--f-mono);
  font-size: 13.5px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  min-width: 0;
}

.row__meta {
  font-size: 11px;
  color: var(--ink-3);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.row__chevron {
  flex: 0 0 auto;
  color: var(--ink-3);
}
</style>
