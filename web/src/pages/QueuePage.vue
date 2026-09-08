<script setup lang="ts">
import { computed } from "vue";
import { useRouter } from "vue-router";
import ActivityStrip from "../components/create/ActivityStrip.vue";
import { useGenerateStream, type Job } from "../composables/useGenerateStream";
import { useHostRouting } from "../composables/useHostRouting";
import { useLiveActivity } from "../composables/useLiveActivity";
import { useActivityRows } from "../composables/useActivityRows";
import { useOpenLiveWork } from "../composables/useOpenLiveWork";
import { setLocalJobHandoff } from "../composables/useGenerationHandoff";
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
const unavailableHosts = computed(() =>
  routing.hosts.value.filter((host) => host.status !== "ready"),
);
const hasWork = computed(
  () =>
    sharedActivityRows.value.length > 0 ||
    localActivityJobs.value.some(
      (job) => job.state === "running" || job.state === "error",
    ),
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
    <ActivityStrip
      v-if="hasWork"
      expanded
      :jobs="localActivityJobs"
      :shared="sharedActivityRows"
      :queue-status="routing.queueStatus.value"
      @open="openJob"
      @shared-open="openLiveWork"
      @cancel="act('cancel', $event)"
      @retry="act('retry', $event)"
      @dismiss="stream.remove"
    />
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
  </main>
</template>

<style scoped>
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
