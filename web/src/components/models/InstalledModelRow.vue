<script setup lang="ts">
/*
 * Installed model row (spec WEB MODELS Installed, prototype lines 1596-1600):
 * a halide icon plate, the mono model name with an outline "★ loaded" badge,
 * a family · size line, and a chevron. The row itself is one button that opens
 * the model detail drawer; a multi-host registry adds a single sibling control
 * for installing the model on a machine that does not have it yet.
 */
import { computed } from "vue";
import ModelMetadataBadges from "@studio/components/ModelMetadataBadges.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import Icon from "@ui/components/Icon.vue";
import Tooltip from "@ui/components/Tooltip.vue";
import type { ModelInfoExtended } from "../../types";
import { modelDisplayName } from "@studio/lib/modelDisplay";
import { useModelInstallTargets } from "../../composables/useModelInstallTargets";
import { formatGB } from "../../util/format";

const props = defineProps<{ model: ModelInfoExtended }>();
const emit = defineEmits<{ open: []; install: [] }>();

/*
 * The Installed shelf lists what THIS server has. A connected machine that
 * lacks the model still needs an install action, so the row grows one — and
 * only one — extra control, and single-host users never see it.
 */
const installTargets = useModelInstallTargets();
const installPlan = computed(() =>
  installTargets.planFor(props.model.name, true),
);
const installHint = computed(() => {
  const machines = installPlan.value.targets
    .filter((t) => t.action === "install")
    .map((t) => t.host.label)
    .join(", ");
  return `Install ${modelDisplayName(props.model)} on ${machines}`;
});
</script>

<template>
  <div class="rowline">
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
    <Tooltip v-if="installPlan.canInstall" :text="installHint">
      <button
        type="button"
        class="rowline__install"
        data-test="install-elsewhere-btn"
        :aria-label="installHint"
        @click="emit('install')"
      >
        <Icon name="download" :size="14" />
        Install
      </button>
    </Tooltip>
  </div>
</template>

<style scoped>
.rowline {
  display: flex;
  align-items: stretch;
  gap: 8px;
  min-width: 0;
}

.rowline .row {
  flex: 1;
  min-width: 0;
}

.rowline__install {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card);
  padding: 0 14px;
  color: var(--ink-2);
  font-family: var(--f-body);
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}

.rowline__install:hover {
  border-color: var(--safelight);
  color: var(--rebate);
}

.rowline__install:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

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
