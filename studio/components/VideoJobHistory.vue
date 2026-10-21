<script setup lang="ts">
import { computed, onBeforeUnmount, ref, shallowRef, watch } from "vue";
import { apiJsonTo } from "../api/client";
import { listVideoJobs } from "../api/videoJobList";
import type { VideoJobOwner, VideoJobAdmission } from "../api/videoJobs";
import type { VideoJob } from "../lib/api/videoJobTypes";
import { readVideoJobOperations } from "../lib/videoJobRecovery";
import { useHeldQueueTransfer } from "../composables/useHeldQueueTransfer";
import VideoJobControl from "./VideoJobControl.vue";
const props = defineProps<{ existingKeys?: string[] }>();
const fleet = useHeldQueueTransfer();
const visibleJob = (owner: VideoJobOwner, job: VideoJob) =>
  !props.existingKeys?.includes(
    `${owner.hostId}:${owner.instanceId}:${job.id}`,
  );
interface HostPage {
  owner: VideoJobOwner;
  jobs: VideoJob[];
  cursor: string | null;
  more: boolean;
}
const pages = shallowRef<HostPage[]>([]),
  pending = shallowRef<VideoJobAdmission[]>([]),
  busy = ref(false),
  error = ref("");
let revision = 0,
  disposed = false;
function current(owner: VideoJobOwner) {
  const host = fleet?.hosts.value.find(
    (host) => host.id === owner.hostId && host.ready,
  );
  return (
    !!host &&
    host.instanceId === owner.instanceId &&
    host.target.baseUrl === owner.target.baseUrl &&
    host.target.apiKey === owner.target.apiKey &&
    host.target.connectionMode === owner.target.connectionMode
  );
}
const unanswered = computed(() =>
  pending.value.filter(
    (record) =>
      !pages.value.some(
        (page) =>
          page.owner.hostId === record.hostId &&
          page.owner.instanceId === record.instanceId &&
          page.jobs.some((job) => job.id === record.operationId),
      ),
  ),
);
async function check() {
  if (disposed) return;
  const token = ++revision;
  busy.value = true;
  error.value = "";
  try {
    pending.value = readVideoJobOperations();
    const result: HostPage[] = [];
    for (const host of [...(fleet?.hosts.value ?? [])]) {
      if (!host.ready) continue;
      const owner = {
        hostId: host.id,
        label: host.label,
        instanceId: host.instanceId,
        target: { ...host.target },
      };
      try {
        const caps = await apiJsonTo<{
          video_jobs?: { supported?: boolean; literal_prompts?: boolean };
        }>(owner.target, "/api/capabilities");
        if (disposed || token !== revision) return;
        if (
          !current(owner) ||
          caps.video_jobs?.supported !== true ||
          caps.video_jobs.literal_prompts !== true
        )
          continue;
        const page = await listVideoJobs(
          owner,
          undefined,
          () => !disposed && token === revision && current(owner),
        );
        result.push({
          owner,
          jobs: page.jobs,
          cursor: page.next_cursor,
          more: page.has_more,
        });
      } catch (cause) {
        if (!disposed && token === revision)
          error.value += `${owner.label}: ${cause instanceof Error ? cause.message : String(cause)} `;
      }
    }
    if (!disposed && token === revision)
      pages.value = result.filter((page) => current(page.owner));
  } catch (cause) {
    if (!disposed && token === revision)
      error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    if (token === revision) busy.value = false;
  }
}
async function more(page: HostPage) {
  if (busy.value || !page.more || !page.cursor) return;
  const token = revision;
  busy.value = true;
  error.value = "";
  try {
    const result = await listVideoJobs(
      page.owner,
      page.cursor,
      () => !disposed && token === revision && current(page.owner),
    );
    if (disposed || token !== revision || !current(page.owner)) return;
    const ids = new Set(page.jobs.map((job) => job.id));
    pages.value = pages.value.map((existing) =>
      existing === page
        ? {
            ...page,
            jobs: [
              ...page.jobs,
              ...result.jobs.filter((job) => !ids.has(job.id)),
            ],
            cursor: result.next_cursor,
            more: result.has_more,
          }
        : existing,
    );
  } catch (cause) {
    if (!disposed && token === revision && current(page.owner))
      error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    if (token === revision) busy.value = false;
  }
}
// Retire old reads synchronously, then coalesce the connection updates from
// one readiness refresh. Progress/uptime never changes this authority key.
watch(
  () =>
    JSON.stringify(
      (fleet?.hosts.value ?? [])
        .map((host) => [
          host.id,
          host.ready,
          host.instanceId,
          host.target.baseUrl,
          host.target.apiKey,
          host.target.connectionMode,
        ])
        .sort((a, b) => String(a[0]).localeCompare(String(b[0]))),
    ),
  () => {
    const token = ++revision;
    pages.value = pages.value.filter((page) => current(page.owner));
    void Promise.resolve().then(() => {
      if (!disposed && token === revision) void check();
    });
  },
  { immediate: true, flush: "sync" },
);
onBeforeUnmount(() => {
  disposed = true;
  revision++;
});
</script>
<template>
  <section aria-label="Saved video jobs">
    <button type="button" :disabled="busy" @click="check">
      {{ busy ? "Checking video jobs…" : "Check saved video jobs" }}
    </button>
    <p v-if="error" role="alert">{{ error }}</p>
    <div
      v-for="record in unanswered"
      :key="`${record.hostId}:${record.instanceId}:${record.operationId}`"
    >
      <p>
        Unanswered video attempt · {{ record.hostId }} ·
        {{ record.operationId }}
      </p>
      <VideoJobControl
        :host-id="record.hostId"
        :instance-id="record.instanceId"
        :job-id="record.operationId"
        :operation="record"
        @changed="check"
      />
    </div>
    <div
      v-for="page in pages"
      :key="`${page.owner.hostId}:${page.owner.instanceId}`"
    >
      <article
        v-for="job in page.jobs.filter((job) => visibleJob(page.owner, job))"
        :key="job.id"
      >
        <p>
          {{ page.owner.label }} · {{ job.state }} ·
          {{ job.completed_clips }} of {{ job.plan.clip_count }} clips
        </p>
        <p v-if="job.error_detail || job.error">
          {{ job.error_detail ?? job.error }}
        </p>
        <VideoJobControl
          :owner="page.owner"
          :job-id="job.id"
          @changed="check"
        />
      </article>
      <button
        v-if="page.more"
        type="button"
        :disabled="busy || !current(page.owner)"
        @click="more(page)"
      >
        Load more videos · {{ page.owner.label }}
      </button>
    </div>
  </section>
</template>
