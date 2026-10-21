<script setup lang="ts">
import ParentVideoRestart from "./ParentVideoRestart.vue";
import { computed, inject, onBeforeUnmount, ref, shallowRef, watch } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import { useHeldQueueTransfer } from "../composables/useHeldQueueTransfer";
import { createVideoJobRun } from "../composables/useVideoJobRun";
import {
  mutateVideoJob,
  type VideoJobAdmission,
  type VideoJobOwner,
} from "../api/videoJobs";
import { VIDEO_LIBRARY_NAVIGATION } from "../lib/videoLibraryNavigation";
import { formatRequestedVideoDuration } from "../lib/videoRequestedDuration";
const props = defineProps<{
  hostId?: string | undefined;
  jobId?: string | null | undefined;
  instanceId?: string | null | undefined;
  owner?: VideoJobOwner | null | undefined;
  operation?: VideoJobAdmission | null | undefined;
  isCurrent?: ((owner: VideoJobOwner) => boolean) | undefined;
  autoOpen?: boolean;
  hideTrigger?: boolean;
}>();
const emit = defineEmits<{ (event: "changed"): void }>();
const fleet = useHeldQueueTransfer(),
  openLibrary = inject(VIDEO_LIBRARY_NAVIGATION, null);
const open = ref(false),
  busy = ref(false),
  deleting = ref(false);
const captured = shallowRef<VideoJobOwner | null>(null);
let revision = 0,
  timer: ReturnType<typeof setTimeout> | undefined;
const available = computed(() => {
  if (props.owner) return props.owner;
  const host = fleet?.hosts.value.find(
    (host) =>
      host.id === props.hostId &&
      host.ready &&
      (!props.instanceId || props.instanceId === host.instanceId),
  );
  return host
    ? {
        hostId: host.id,
        label: host.label,
        instanceId: host.instanceId,
        target: { ...host.target },
      }
    : null;
});
function current(owner: VideoJobOwner): boolean {
  if (props.isCurrent) return props.isCurrent(owner);
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
const run = createVideoJobRun((owner) => open.value && current(owner));
const { detail, error, cancelledBeforeAdmission, admission } = run;
const settled = computed(
  () =>
    cancelledBeforeAdmission.value ||
    (detail.value &&
      ["completed", "failed", "interrupted", "cancelled", "paused"].includes(
        detail.value.state,
      )),
);
async function refresh(attempt = revision) {
  if (timer) clearTimeout(timer);
  const owner = captured.value;
  if (!open.value || !owner || !current(owner)) {
    error.value = "Reconnect this video's original machine to check its Queue.";
    return;
  }
  await run.refresh();
  if (attempt !== revision || !open.value || !current(owner)) return;
  if (!settled.value && !error.value && document.visibilityState !== "hidden")
    timer = setTimeout(() => void refresh(attempt), 2000);
}
function close() {
  open.value = false;
  revision += 1;
  if (timer) clearTimeout(timer);
}
async function show() {
  const owner = available.value,
    id = props.jobId;
  if (!owner || !id || !current(owner)) return;
  if (timer) clearTimeout(timer);
  captured.value = { ...owner, target: { ...owner.target } };
  deleting.value = false;
  open.value = true;
  const attempt = ++revision;
  await run.recover(
    owner,
    props.operation ?? {
      kind: "video_job",
      operationId: id,
      hostId: owner.hostId,
      instanceId: owner.instanceId!,
      baseUrl: owner.target.baseUrl,
      ...(owner.target.connectionMode
        ? { connectionMode: owner.target.connectionMode }
        : {}),
      createdAt: Date.now(),
    },
  );
  if (attempt === revision && open.value && !settled.value && !error.value)
    timer = setTimeout(() => void refresh(attempt), 2000);
}
async function action(kind: "cancel" | "resume" | "delete") {
  const owner = captured.value,
    id = props.jobId,
    attempt = revision;
  if (!owner || !id || busy.value || !current(owner)) return;
  busy.value = true;
  error.value = "";
  try {
    if (kind === "cancel") await run.cancel();
    else if (kind === "resume") await run.resume();
    else
      await mutateVideoJob(
        owner,
        id,
        kind,
        () => attempt === revision && current(owner),
      );
    if (attempt !== revision || !current(owner)) return;
    emit("changed");
    if (kind === "delete") close();
    else await refresh(attempt);
  } catch (cause) {
    if (attempt === revision)
      error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    busy.value = false;
    deleting.value = false;
  }
}
function showSaved() {
  if (
    captured.value &&
    detail.value?.output_filename &&
    current(captured.value)
  ) {
    openLibrary?.({
      owner: captured.value,
      filename: detail.value.output_filename,
    });
    close();
  }
}
watch(
  () =>
    JSON.stringify([
      props.jobId,
      props.owner?.hostId,
      props.owner?.instanceId,
      props.owner?.target.baseUrl,
      props.owner?.target.apiKey,
      props.owner?.target.connectionMode,
      props.hostId,
      props.instanceId,
      props.autoOpen,
    ]),
  () => {
    close();
    if (props.autoOpen && props.jobId) void show();
  },
  { immediate: true },
);
onBeforeUnmount(() => {
  close();
  run.dispose();
});
</script>
<template>
  <button
    v-if="!hideTrigger"
    type="button"
    :disabled="!jobId || !available"
    class="video-job-control"
    data-test="video-job-details"
    @click.stop="show"
  >
    Video details
  </button>
  <Teleport to="body"
    ><div v-if="open" class="video-job-overlay">
      <ModalPanel :open="open" label="Video details" @close="close">
        <template #title>Video · {{ captured?.label }}</template>
        <p v-if="!detail && !error" role="status">
          {{
            cancelledBeforeAdmission
              ? "Cancelled before queueing"
              : "Checking this video…"
          }}
        </p>
        <template v-if="detail">
          <p>{{ detail.plan.model }}</p>
          <p role="status">
            {{
              detail.state === "completed"
                ? detail.output_filename
                  ? "Saved to Library"
                  : "Completed, but no saved Library video was reported."
                : `${detail.state.charAt(0).toUpperCase()}${detail.state.slice(1)}`
            }}
          </p>
          <p>
            {{ detail.completed_clips }} of {{ detail.plan.clip_count }} clips ·
            {{ formatRequestedVideoDuration(detail.plan.total_duration_ms) }}
            seconds
          </p>
          <p v-if="detail.error_detail || detail.error">
            {{ detail.error_detail ?? detail.error }}
          </p>
          <div class="video-job-actions">
            <ParentVideoRestart
              v-if="
                captured &&
                ['failed', 'paused', 'interrupted', 'cancelled'].includes(
                  detail.state,
                )
              "
              :owner="captured"
              :job-id="detail.id"
            />
            <button
              v-if="detail.output_filename && captured && openLibrary"
              type="button"
              @click="showSaved"
            >
              View in Library
            </button>
            <button
              v-if="detail.can_cancel"
              type="button"
              :disabled="busy"
              @click="action('cancel')"
            >
              Stop video
            </button>
            <button
              v-if="detail.can_resume"
              type="button"
              :disabled="busy"
              @click="action('resume')"
            >
              Resume video
            </button>
            <button
              v-if="detail.can_delete"
              type="button"
              :disabled="busy"
              @click="deleting = true"
            >
              Delete recovery data…
            </button>
          </div>
          <div v-if="deleting" role="alert">
            <p>Delete this job's recovery data? Saved Library videos remain.</p>
            <button type="button" :disabled="busy" @click="action('delete')">
              Delete recovery data
            </button>
            <button type="button" :disabled="busy" @click="deleting = false">
              Keep job
            </button>
          </div>
        </template>
        <button
          v-if="!detail && admission && !cancelledBeforeAdmission"
          type="button"
          :disabled="busy"
          @click="action('cancel')"
        >
          Stop video
        </button>
        <p v-if="error" role="alert">{{ error }}</p>
        <button type="button" :disabled="busy" @click="refresh()">
          Check queue
        </button>
        <template #footer
          ><button type="button" @click="close">Close</button></template
        >
      </ModalPanel>
    </div></Teleport
  >
</template>
<style scoped>
.video-job-overlay {
  position: fixed;
  inset: 0;
  z-index: var(--mold-z-modal);
}
.video-job-control,
.video-job-overlay button {
  min-height: 44px;
  font: inherit;
  color: var(--mold-text);
  background: var(--mold-bg);
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
  padding: 8px 12px;
}
.video-job-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
</style>
