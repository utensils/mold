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
import { styleDisplayName } from "@studio/lib/styleLabel";
import { modelKindValue } from "@studio/lib/modelMetadata";
import { catalogPullLabel, catalogSizeInfo } from "@studio/lib/catalogLabel";
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
  return `Get ${styleDisplayName(props.model)} on ${machines}`;
});

/** The friendly name leads; the runnable id stays below it in mono. */
const displayName = computed(() => styleDisplayName(props.model));
/** Every row on this shelf is a style, so "Checkpoint" says nothing. Only a
 *  row that is something else — a LoRA, a VAE, an upscaler — earns a badge. */
const badgeKind = computed(() => {
  const kind = modelKindValue({
    kind: props.model.kind,
    family: props.model.family,
  });
  return kind === "checkpoint" ? null : kind;
});
/** What sending this style to another machine will cost in bytes. */
const getLabel = computed(() =>
  catalogPullLabel(
    catalogSizeInfo({ size_bytes: props.model.size_gb * 1_000_000_000 }),
    "Get it",
  ),
);
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
          <span class="row__name" data-test="installed-row-name">{{
            displayName
          }}</span>
          <ModelMetadataBadges
            :kind="badgeKind"
            :family="null"
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
        <span
          v-if="displayName !== props.model.name"
          class="row__id"
          data-test="installed-row-id"
          >{{ props.model.name }}</span
        >
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
        {{ getLabel }}
      </button>
    </Tooltip>
  </div>
</template>

<style scoped>
.row__id {
  display: block;
  font-family: var(--f-mono);
  font-size: 0.75rem;
  color: var(--ink-3);
  overflow-wrap: anywhere;
}

.rowline {
  display: flex;
  align-items: stretch;
  gap: 8px;
  flex-wrap: wrap;
  min-width: 0;
}

.rowline .row {
  flex: 1 1 20rem;
  max-width: 100%;
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
  min-height: 44px;
  color: var(--ink-2);
  font-family: var(--f-body);
  font-size: 0.875rem;
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
  flex: 1 1 10rem;
  font-family: var(--f-body);
  font-size: 1rem;
  font-weight: 600;
  overflow-wrap: anywhere;
  min-width: 0;
}

.row__meta {
  font-size: 0.875rem;
  color: var(--ink-3);
  overflow-wrap: anywhere;
}

.row__chevron {
  flex: 0 0 auto;
  color: var(--ink-3);
}
</style>
