<script setup lang="ts">
import { computed } from "vue";
import ModelMetadataBadges from "./ModelMetadataBadges.vue";
import {
  formatMiniMaxH3Bytes,
  planMiniMaxH3Install,
  presentMiniMaxH3Fleet,
  type MiniMaxH3ComponentState,
  type MiniMaxH3HostInput,
  type MiniMaxH3InstallPlan,
} from "../lib/minimaxH3Inventory";

const props = withDefaults(
  defineProps<{
    hosts: readonly MiniMaxH3HostInput[];
    heading?: string;
  }>(),
  { heading: "MiniMax H3 runtime" },
);

const fleet = computed(() => presentMiniMaxH3Fleet(props.hosts));

function stateLabel(state: MiniMaxH3ComponentState): string {
  if (state === "installed") return "Installed";
  if (state === "missing") return "Missing";
  if (state === "downloading") return "Downloading";
  return "Failed";
}

function planLabel(plan: MiniMaxH3InstallPlan | null): string {
  if (!plan) return "Unavailable";
  if (plan.action === "installed") return "Ready";
  if (plan.action === "waiting") return `Downloading on ${plan.hostLabel}`;
  const verb = plan.action === "repair" ? "Repair plan" : "Install plan";
  return `${verb} · ${formatMiniMaxH3Bytes(plan.bytesToDownload)}`;
}

function percentage(done: number, total: number): number {
  return total > 0
    ? Math.max(0, Math.min(100, Math.round((done / total) * 100)))
    : 0;
}
</script>

<template>
  <section
    v-if="fleet.length"
    class="h3-inventory"
    aria-labelledby="h3-inventory-heading"
    data-test="h3-inventory"
  >
    <header class="h3-inventory__header">
      <div>
        <h2 id="h3-inventory-heading">{{ heading }}</h2>
        <p>Capability-reported task partitions and shared files.</p>
      </div>
      <span
        >{{ fleet.length }} {{ fleet.length === 1 ? "host" : "hosts" }}</span
      >
    </header>

    <article
      v-for="host in fleet"
      :key="host.hostId"
      class="h3-host"
      :data-test="`h3-host-${host.hostId}`"
    >
      <header class="h3-host__header">
        <div>
          <h3>{{ host.hostLabel }}</h3>
          <p>
            {{
              host.tasks.length === 1
                ? host.tasks[0]?.taskLabel
                : "Advertised tasks"
            }}
            · {{ formatMiniMaxH3Bytes(host.advertisedTasksDiskBytes) }} on disk
            <template v-if="host.remainingBytes">
              · {{ formatMiniMaxH3Bytes(host.remainingBytes) }} not ready
            </template>
          </p>
        </div>
        <span class="h3-state h3-state--installed">CUDA only</span>
      </header>

      <dl class="h3-qualification" data-test="h3-qualification">
        <div>
          <dt>Host RAM</dt>
          <dd>
            {{
              formatMiniMaxH3Bytes(host.qualification.minimum_host_ram_bytes)
            }}
            minimum
          </dd>
        </div>
        <div>
          <dt>VRAM</dt>
          <dd>
            {{ formatMiniMaxH3Bytes(host.qualification.minimum_vram_bytes) }}
            minimum
          </dd>
        </div>
        <div>
          <dt>Attention</dt>
          <dd>{{ host.qualification.attention_profile }}</dd>
        </div>
        <div>
          <dt>Quantization</dt>
          <dd>{{ host.qualification.quantization_profile }}</dd>
        </div>
        <div>
          <dt>Metal</dt>
          <dd>Unsupported</dd>
        </div>
      </dl>

      <div class="h3-tasks">
        <section
          v-for="task in host.tasks"
          :key="task.task"
          class="h3-task"
          :data-test="`h3-task-${host.hostId}-${task.task}`"
        >
          <header>
            <div>
              <span class="h3-task__eyebrow">{{ task.taskLabel }}</span>
              <h4>{{ task.displayName }}</h4>
            </div>
            <span class="h3-state" :class="`h3-state--${task.readiness}`">
              {{ stateLabel(task.readiness) }}
            </span>
          </header>
          <ModelMetadataBadges kind="checkpoint" modality="video" />
          <p>
            {{ task.tier }} · {{ formatMiniMaxH3Bytes(task.diskBytes) }} exact
            footprint
          </p>
          <strong class="h3-task__plan">
            {{ planLabel(planMiniMaxH3Install(host, [task.task])) }}
          </strong>
        </section>
      </div>

      <section class="h3-components" aria-label="H3 component readiness">
        <header>
          <h4>Component readiness</h4>
          <span>Shared files are counted once</span>
        </header>
        <ul>
          <li
            v-for="component in host.components"
            :key="component.id"
            :data-test="`h3-component-${host.hostId}-${component.id}`"
          >
            <div class="h3-component__copy">
              <strong>{{ component.displayName }}</strong>
              <span>
                {{ component.kindLabel }} ·
                {{
                  component.scope === "shared"
                    ? "Shared"
                    : component.scope.toUpperCase()
                }}
                ·
                {{ formatMiniMaxH3Bytes(component.sizeBytes) }}
              </span>
              <span v-if="component.error" class="h3-component__error">{{
                component.error
              }}</span>
              <span v-else-if="component.download">
                {{ component.download.hostLabel }} ·
                {{ formatMiniMaxH3Bytes(component.download.bytesDone) }} /
                {{ formatMiniMaxH3Bytes(component.download.bytesTotal) }}
              </span>
            </div>
            <div v-if="component.download" class="h3-component__progress">
              <span
                role="progressbar"
                :aria-label="`Downloading ${component.displayName} on ${component.download.hostLabel}`"
                aria-valuemin="0"
                aria-valuemax="100"
                :aria-valuenow="
                  percentage(
                    component.download.bytesDone,
                    component.download.bytesTotal,
                  )
                "
              >
                <i
                  :style="{
                    width: `${percentage(
                      component.download.bytesDone,
                      component.download.bytesTotal,
                    )}%`,
                  }"
                />
              </span>
            </div>
            <span class="h3-state" :class="`h3-state--${component.state}`">
              {{ stateLabel(component.state) }}
            </span>
          </li>
        </ul>
      </section>
    </article>
  </section>
</template>

<style scoped>
.h3-inventory {
  margin: 16px 0;
  color: var(--color-ink, var(--ink, currentColor));
}

.h3-inventory__header,
.h3-host__header,
.h3-task > header,
.h3-components > header,
.h3-components li {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.h3-inventory__header {
  margin-bottom: 10px;
}

.h3-inventory h2,
.h3-inventory h3,
.h3-inventory h4,
.h3-inventory p,
.h3-inventory dl,
.h3-inventory ul {
  margin: 0;
}

.h3-inventory__header h2,
.h3-host__header h3 {
  font-size: 15px;
  font-weight: 700;
}

.h3-inventory__header p,
.h3-host__header p,
.h3-task p,
.h3-components > header span,
.h3-component__copy > span {
  color: var(--color-ink-2, var(--ink-2, #667085));
  font-size: 12px;
  line-height: 1.4;
}

.h3-inventory__header > span {
  color: var(--color-ink-2, var(--ink-2, #667085));
  font-size: 12px;
}

.h3-host {
  padding: 14px;
  border: 1px solid var(--color-edge, var(--edge, #d0d5dd));
  border-radius: 14px;
  background: var(--color-bench, var(--bench, rgba(255, 255, 255, 0.03)));
}

.h3-host + .h3-host {
  margin-top: 10px;
}

.h3-qualification {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 8px;
  margin-top: 12px !important;
}

.h3-qualification > div {
  min-width: 0;
  padding: 8px 10px;
  border-radius: 9px;
  background: var(--color-bath, var(--bath, rgba(127, 127, 127, 0.08)));
}

.h3-qualification dt {
  color: var(--color-ink-2, var(--ink-2, #667085));
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.h3-qualification dd {
  margin: 2px 0 0;
  overflow-wrap: anywhere;
  font-size: 12px;
  font-weight: 600;
}

.h3-tasks {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 10px;
}

.h3-task {
  min-width: 0;
  padding: 11px;
  border: 1px solid var(--color-edge, var(--edge, #d0d5dd));
  border-radius: 10px;
}

.h3-task__eyebrow {
  color: var(--color-ink-2, var(--ink-2, #667085));
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
}

.h3-task h4,
.h3-components h4 {
  margin-top: 2px;
  font-size: 13px;
}

.h3-task :deep(.model-metadata-badges) {
  margin: 8px 0;
}

.h3-task__plan {
  display: block;
  margin-top: 8px;
  font-size: 12px;
}

.h3-components {
  margin-top: 12px;
}

.h3-components ul {
  margin-top: 6px !important;
  padding: 0;
  list-style: none;
  border-top: 1px solid var(--color-edge, var(--edge, #d0d5dd));
}

.h3-components li {
  min-height: 44px;
  padding: 8px 0;
  border-bottom: 1px solid var(--color-edge, var(--edge, #d0d5dd));
}

.h3-component__copy {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
}

.h3-component__copy strong {
  overflow-wrap: anywhere;
  font-size: 12px;
}

.h3-component__error {
  color: var(--color-danger, #b42318) !important;
}

.h3-component__progress {
  width: min(120px, 22vw);
  align-self: center;
}

.h3-component__progress > span {
  display: block;
  height: 4px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--color-edge, var(--edge, #d0d5dd));
}

.h3-component__progress i {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--color-rebate, var(--rebate, #3b82f6));
}

.h3-state {
  flex: none;
  padding: 3px 7px;
  border-radius: 999px;
  background: var(--color-bath, var(--bath, rgba(127, 127, 127, 0.12)));
  color: var(--color-ink-2, var(--ink-2, #667085));
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
}

.h3-state--installed {
  color: var(--color-success, #067647);
}

.h3-state--missing {
  color: var(--color-warning, #b54708);
}

.h3-state--downloading {
  color: var(--color-rebate, var(--rebate, #2563eb));
}

.h3-state--failed {
  color: var(--color-danger, #b42318);
}

@media (max-width: 640px) {
  .h3-tasks {
    grid-template-columns: 1fr;
  }

  .h3-qualification {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .h3-components li {
    flex-wrap: wrap;
  }

  .h3-component__progress {
    order: 3;
    width: 100%;
  }
}
</style>
