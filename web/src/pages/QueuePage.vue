<script setup lang="ts">
import ModalPanel from "@ui/components/ModalPanel.vue";
import QueueEntryDetail from "@studio/components/QueueEntryDetail.vue";
import { useQueueInspection } from "../composables/useQueueInspection";
import { ORIGIN_HOST_ID } from "../lib/hostRegistry";
import { computed } from "vue";
import { useRouter } from "vue-router";
import ActivityStrip from "../components/create/ActivityStrip.vue";
import { useGenerateStream, type Job } from "../composables/useGenerateStream";
import { useHostRouting } from "../composables/useHostRouting";
import { useLiveActivity } from "../composables/useLiveActivity";
import { useActivityRows } from "../composables/useActivityRows";
import { useQueueSections } from "../composables/useQueueSections";
import { useOpenLiveWork } from "../composables/useOpenLiveWork";
import { selectedQueueGeneration } from "@studio/api/generationSelection";
import type { OutputMetadata } from "../types";
import {
  setGenerationHandoff,
  setLocalJobHandoff,
} from "../composables/useGenerationHandoff";
import { workspaceLabel } from "../lib/workspaces";
import { toast } from "../lib/toasts";

const router = useRouter();
const routing = useHostRouting();
const liveActivity = useLiveActivity(routing);
const stream = useGenerateStream();
const { sharedActivityRows, localActivityJobs } = useActivityRows(
  stream.jobs,
  liveActivity.rows,
);
const openLiveWork = useOpenLiveWork(routing);
const inspection = useQueueInspection(routing, liveActivity.refresh);
function inspectLocal(job: Job, opener: HTMLElement) {
  const row = liveActivity.rows.value.find(
    (row) =>
      row.hostId === (job.hostId ?? ORIGIN_HOST_ID) && row.id === job.serverId,
  );
  if (row?.execution === "chain" || row?.kind === "sequence") {
    void openLiveWork(row);
    return;
  }
  if (row)
    void inspection.open(
      row,
      {
        job,
        cancel: () => stream.cancel(job.id),
        retry: () => stream.retry(job.id),
      },
      opener,
    );
  else
    toast(
      "error",
      "This job is not in the machine's current activity. Refresh its machine details before changing it.",
    );
}
async function reuseInspected() {
  const snapshot = await inspection.snapshot();
  if (!snapshot) return;
  if (snapshot.localJob) {
    setLocalJobHandoff(snapshot.localJob.id);
  } else {
    const selection = selectedQueueGeneration<OutputMetadata>(
      [snapshot.detail.job],
      snapshot.row.id,
    );
    if (!selection) {
      inspection.error.value =
        "This machine cannot restore settings for this job.";
      return;
    }
    setGenerationHandoff({
      metadata: selection.metadata,
      seedPinned: true,
      queueSelection: {
        hostId: snapshot.row.hostId,
        jobId: selection.jobId,
        running: selection.running,
      },
    });
  }
  inspection.close();
  await router.push("/create");
}
const unavailableHosts = computed(() =>
  routing.hosts.value.filter((host) => host.status !== "ready"),
);
const sections = useQueueSections(
  localActivityJobs,
  sharedActivityRows,
  routing.queueStatus,
  routing.hosts,
);
const hasWork = computed(() =>
  sections.value.some((section) => section.count > 0),
);

async function openJob(job: Job) {
  setLocalJobHandoff(job.id);
  await router.push("/create");
}
async function act(action: "cancel" | "retry", id: string) {
  try {
    await stream[action](id);
  } catch (error) {
    toast("error", error instanceof Error ? error.message : String(error));
  }
}
</script>

<template>
  <main class="workspace-page queue-page" data-test="queue-page">
    <h1 class="workspace-heading">{{ workspaceLabel("queue") }}</h1>
    <p class="workspace-description">
      Work continues on your machines when you leave this page.
    </p>
    <p v-if="unavailableHosts.length" class="queue-notice" role="status">
      Reconnecting to
      {{ unavailableHosts.map((host) => host.label).join(", ") }}. Last-known
      work stays visible until the machine responds.
    </p>
    <div v-if="hasWork" class="queue-sections">
      <template v-for="section in sections" :key="section.id">
        <section
          v-if="section.count"
          class="queue-section"
          :aria-labelledby="`queue-${section.id}`"
          :data-test="`queue-section-${section.id}`"
        >
          <h2 :id="`queue-${section.id}`">
            {{ section.label }} <span>{{ section.count }}</span>
          </h2>
          <ActivityStrip
            expanded
            :jobs="section.local"
            :shared="section.shared"
            :queue-status="routing.queueStatus.value"
            @open="openJob"
            @shared-open="openLiveWork"
            @shared-inspect="
              (row, opener) => inspection.open(row, undefined, opener)
            "
            @inspect="inspectLocal"
            @cancel="act('cancel', $event)"
            @retry="act('retry', $event)"
            @dismiss="stream.remove"
          />
        </section>
      </template>
    </div>
    <section v-else class="queue-empty">
      <h2>
        {{
          unavailableHosts.length
            ? "Waiting for your machines"
            : "Nothing waiting here"
        }}
      </h2>
      <p>
        {{
          unavailableHosts.length
            ? "The queue will refresh when your machines reconnect."
            : "Your next image or clip will appear here while it is being made."
        }}
      </p>
      <router-link to="/create">{{ workspaceLabel("create") }}</router-link>
    </section>
    <p class="queue-footer">
      Finished results are in
      <router-link to="/library">{{ workspaceLabel("library") }}</router-link
      >.
    </p>
    <Teleport to="body">
      <div v-if="inspection.selected.value" class="queue-dialog-layer">
        <ModalPanel
          :open="true"
          :width="720"
          label="Queue job details"
          @close="inspection.close"
        >
          <QueueEntryDetail
            v-if="inspection.model.value"
            compact
            :model="inspection.model.value"
            :transfer-host-id="inspection.selected.value?.hostId"
            :cancelling="inspection.pendingAction.value === 'cancel'"
            :retrying="inspection.pendingAction.value === 'retry'"
            :error="inspection.error.value"
            confirm="inline"
            @close="inspection.close"
            @reuse="reuseInspected"
            @cancel="inspection.act('cancel')"
            @retry="inspection.act('retry')"
          />
          <p v-else role="status">
            {{ inspection.error.value ?? "Loading job details…" }}
          </p>
          <button
            v-if="inspection.canPause.value"
            type="button"
            class="queue-pause"
            @click="
              inspection.act(
                inspection.detail.value?.job.state === 'paused'
                  ? 'resume'
                  : 'pause',
              )
            "
          >
            {{
              inspection.detail.value?.job.state === "paused"
                ? "Resume this job"
                : "Pause this job"
            }}
          </button>
          <button
            v-if="!inspection.model.value"
            type="button"
            class="queue-pause"
            @click="inspection.close"
          >
            Close
          </button>
        </ModalPanel>
      </div>
    </Teleport>
  </main>
</template>

<style scoped>
.queue-dialog-layer {
  position: fixed;
  inset: 0;
  z-index: 60;
}
.queue-dialog-layer :deep(.ms-modal) {
  padding: 12px;
}
.queue-dialog-layer :deep(.ms-modal__panel) {
  max-width: 100%;
  max-height: 100%;
  display: flex;
  flex-direction: column;
}
.queue-dialog-layer :deep(.ms-modal__body) {
  min-height: 0;
  overflow: auto;
  display: flex;
  flex-direction: column;
}
.queue-dialog-layer :deep(.qed) {
  flex: 0 0 auto;
}
.queue-dialog-layer :deep(.qed__body) {
  flex: none;
  overflow: visible;
}
.queue-dialog-layer :deep(.qed__actions) {
  flex-wrap: wrap;
}
.queue-dialog-layer :deep(.qed__title) {
  overflow-wrap: anywhere;
}
.queue-pause {
  margin-top: 16px;
  min-height: 44px;
  padding: 8px 16px;
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
  color: var(--mold-text);
  background: var(--mold-bg-deep);
}

.queue-sections {
  display: grid;
  gap: 24px;
}
.queue-section {
  min-width: 0;
}
.queue-section h2 {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 16px;
  margin-bottom: 8px;
  font-size: 1rem;
  font-weight: 600;
  border-bottom: 1px solid var(--mold-border);
}
.queue-section h2 span {
  font-family: var(--mold-font-mono);
  font-size: 0.8125rem;
  color: var(--mold-text-dim);
}
.queue-section :deep(.activity__head) {
  display: none;
}

.queue-notice,
.queue-empty {
  padding: 20px;
  margin-bottom: 20px;
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  line-height: 1.5;
}
.queue-empty h2 {
  font-size: 1.125rem;
  font-weight: 600;
}
.queue-empty p {
  margin-block: 8px 16px;
  color: var(--mold-text-dim);
}
.queue-page a {
  color: var(--mold-blue);
  text-decoration: underline;
  text-underline-offset: 3px;
}
.queue-footer {
  margin-top: 24px;
  color: var(--mold-text-dim);
}
.queue-page :deep(.activity__pill) {
  max-width: 100%;
  width: 100%;
  min-height: 48px;
}
.queue-page :deep(.activity__queued) {
  display: block;
}
.queue-page :deep(.activity__pill-text) {
  flex: 1;
  white-space: normal;
  overflow-wrap: anywhere;
}
.queue-page :deep(.activity__cancel),
.queue-page :deep(.activity__row-action),
.queue-page :deep(.activity__dismiss) {
  min-width: 44px;
  min-height: 44px;
}
.queue-page :deep(.activity__kicker) {
  font-size: 0.75rem;
}
.queue-page :deep(.live-activity-surface) {
  min-height: 64px;
  padding: 14px 16px;
  border-radius: var(--mold-radius-2);
}
.queue-page :deep(.live-activity-copy strong),
.queue-page :deep(.activity__prompt),
.queue-page :deep(.activity__pill-text),
.queue-page :deep(.activity__error) {
  font-size: 0.9375rem;
  white-space: normal;
  overflow-wrap: anywhere;
}
.queue-page :deep(.live-activity-copy span) {
  font-size: 0.8125rem;
  white-space: normal;
  overflow-wrap: anywhere;
}
</style>
