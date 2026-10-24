<script setup lang="ts">
import ParentVideoRestart from "./ParentVideoRestart.vue";
import { computed, inject, onBeforeUnmount, ref, shallowRef, watch } from "vue";
import ModalPanel from "@ui/components/ModalPanel.vue";
import { useHeldQueueTransfer } from "../composables/useHeldQueueTransfer";
import { createVideoJobRun } from "../composables/useVideoJobRun";
import { type VideoJobAdmission, type VideoJobOwner } from "../api/videoJobs";
import { videoLibraryOutputExists } from "../api/videoLibraryOutput";
import { VIDEO_LIBRARY_NAVIGATION } from "../lib/videoLibraryNavigation";
import { videoJobRowFromJob } from "../lib/videoJobRow";
import { modelDisplayName } from "../lib/modelDisplay";
import { copyTextToClipboard } from "../lib/notificationClipboard";
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
const emit = defineEmits<{
  (event: "changed"): void;
  (event: "removed", id: string): void;
}>();
const fleet = useHeldQueueTransfer(),
  openLibrary = inject(VIDEO_LIBRARY_NAVIGATION, null);
const open = ref(false),
  busy = ref(false),
  deleting = ref(false);
const outputState = ref<"unknown" | "checking" | "available" | "missing">(
    "unknown",
  ),
  outputError = ref("");
let outputRevision = 0,
  removedEmitted = false;
const captured = shallowRef<VideoJobOwner | null>(null);
let revision = 0,
  timer: ReturnType<typeof setTimeout> | undefined;
const available = computed(() => {
  if (props.owner) return current(props.owner) ? props.owner : null;
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
const copied = ref(false);
const modelName = computed(() =>
  detail.value?.plan?.model
    ? modelDisplayName({ name: detail.value.plan.model })
    : "Video",
);
/** The state chip an ordinary job's details wear, from the row contract. */
const stateCode = computed(() =>
  cancelledBeforeAdmission.value
    ? "CANCELLED"
    : (row.value?.symbol.toUpperCase() ?? "CHECKING"),
);
const technical = computed(() =>
  [
    `${captured.value?.label ?? ""} · Job ${props.jobId ?? ""}`,
    detail.value?.plan?.model ? `Model ${detail.value.plan.model}` : "",
    detail.value?.error_detail ?? detail.value?.error ?? "",
  ]
    .filter(Boolean)
    .join("\n"),
);
async function copyDetails() {
  copied.value = await copyTextToClipboard(technical.value);
  if (copied.value) setTimeout(() => (copied.value = false), 2000);
}
const row = computed(() =>
  detail.value ? videoJobRowFromJob(detail.value) : null,
);
function finishRemoval() {
  if (!run.removed.value || removedEmitted || !props.jobId) return false;
  removedEmitted = true;
  emit("removed", props.jobId);
  emit("changed");
  close();
  return true;
}
async function checkOutput(): Promise<boolean> {
  const owner = captured.value,
    filename = detail.value?.output_filename,
    id = detail.value?.id;
  if (!owner || !filename || !openLibrary) return false;
  const attempt = revision,
    token = ++outputRevision;
  const fresh = () =>
    attempt === revision &&
    token === outputRevision &&
    open.value &&
    current(owner) &&
    detail.value?.id === id &&
    detail.value?.output_filename === filename;
  outputState.value = "checking";
  outputError.value = "";
  try {
    const exists = await videoLibraryOutputExists(owner, filename, fresh);
    if (!fresh()) return false;
    outputState.value = exists ? "available" : "missing";
    return exists;
  } catch (cause) {
    if (fresh()) {
      outputState.value = "unknown";
      outputError.value = `Could not check the saved video: ${cause instanceof Error ? cause.message : String(cause)}`;
    }
    return false;
  }
}
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
  if (finishRemoval()) return;
  await checkOutput();
  if (attempt !== revision || !open.value || !current(owner)) return;
  // The details read the job live, even after a failed action, so a lost
  // Stop reply reconciles without a manual "Check queue".
  if (!settled.value && document.visibilityState !== "hidden")
    timer = setTimeout(() => void refresh(attempt), 2000);
}
function close() {
  open.value = false;
  revision += 1;
  outputRevision += 1;
  if (timer) clearTimeout(timer);
}
async function show() {
  const owner = available.value,
    id = props.jobId;
  if (!owner || !id || !current(owner)) return;
  if (timer) clearTimeout(timer);
  captured.value = { ...owner, target: { ...owner.target } };
  deleting.value = false;
  removedEmitted = false;
  outputState.value = "unknown";
  outputError.value = "";
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
    { acknowledged: !props.operation },
  );
  if (attempt !== revision || !open.value || !current(owner)) return;
  if (finishRemoval()) return;
  await checkOutput();
  if (attempt === revision && open.value && !settled.value)
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
    else await run.remove();
    if (attempt !== revision || !current(owner)) return;
    if (finishRemoval()) return;
    emit("changed");
    await refresh(attempt);
  } catch (cause) {
    if (attempt === revision)
      error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    busy.value = false;
    deleting.value = false;
  }
}
async function showSaved() {
  if (
    !busy.value &&
    (await checkOutput()) &&
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
    v-if="!hideTrigger && !run.removed.value"
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
      <ModalPanel
        :open="open"
        :width="720"
        label="Video details"
        @close="close"
      >
        <div class="vjd" data-test="video-job-detail">
          <header class="vjd__head">
            <span class="vjd__state" :data-state="row?.symbol ?? 'checking'">{{
              stateCode
            }}</span>
            <h2 class="vjd__title" :title="detail?.plan?.model">
              {{ modelName }}
            </h2>
            <button
              type="button"
              class="vjd__close"
              aria-label="Close video details"
              @click="close"
            >
              ✕
            </button>
          </header>
          <div class="vjd__body">
            <p v-if="!detail && !error" role="status" class="vjd__caption">
              {{
                cancelledBeforeAdmission
                  ? "Cancelled before queueing"
                  : "Checking this video…"
              }}
            </p>
            <template v-if="detail">
              <p role="status" class="vjd__caption">
                {{
                  detail.state === "completed"
                    ? detail.output_filename
                      ? outputState === "available"
                        ? "Saved to Library"
                        : "Completed"
                      : "Completed, but no saved Library video was reported."
                    : row?.caption
                }}
              </p>
              <dl class="vjd__facts">
                <div>
                  <dt>Video</dt>
                  <dd>{{ row?.detail }}</dd>
                </div>
                <div>
                  <dt>Machine</dt>
                  <dd>{{ captured?.label }}</dd>
                </div>
              </dl>
              <p
                v-if="detail.error_detail || detail.error"
                class="vjd__problem"
                role="alert"
              >
                {{ detail.error_detail ?? detail.error }}
              </p>
              <p
                v-if="outputState === 'missing'"
                role="status"
                class="vjd__hint"
              >
                This video is no longer in this machine's Library. Its recovery
                data can still be removed.
              </p>
              <p
                v-if="outputState === 'checking'"
                role="status"
                class="vjd__hint"
              >
                Checking the saved video…
              </p>
            </template>
            <details
              class="vjd__technical"
              data-test="video-job-technical-details"
            >
              <summary>Technical details</summary>
              <pre>{{ technical }}</pre>
            </details>
          </div>
          <footer class="vjd__foot">
            <p v-if="error" role="alert" class="vjd__error">{{ error }}</p>
            <p v-if="outputError" role="alert" class="vjd__error">
              {{ outputError }}
            </p>
            <div v-if="deleting" role="alert" class="vjd__confirm">
              <p class="vjd__hint">
                Delete this job's recovery data? Saved Library videos remain.
              </p>
              <div class="vjd__actions">
                <button
                  type="button"
                  :disabled="busy"
                  @click="deleting = false"
                >
                  Keep job
                </button>
                <button
                  type="button"
                  class="vjd__danger"
                  :disabled="busy"
                  @click="action('delete')"
                >
                  Delete recovery data
                </button>
              </div>
            </div>
            <div class="vjd__actions" data-test="video-job-actions">
              <button
                v-if="
                  detail?.output_filename &&
                  captured &&
                  openLibrary &&
                  outputState === 'available'
                "
                :disabled="busy"
                type="button"
                @click="showSaved"
              >
                View in Library
              </button>
              <ParentVideoRestart
                v-if="
                  detail &&
                  captured &&
                  ['failed', 'paused', 'interrupted', 'cancelled'].includes(
                    detail.state,
                  )
                "
                :owner="captured"
                :job-id="detail.id"
              />
              <button type="button" @click="copyDetails">
                {{ copied ? "Copied" : "Copy details" }}
              </button>
              <button
                v-if="detail?.can_resume"
                type="button"
                :disabled="busy"
                @click="action('resume')"
              >
                Resume video
              </button>
              <button
                v-if="detail?.can_delete && !deleting"
                type="button"
                class="vjd__danger"
                :disabled="busy"
                @click="deleting = true"
              >
                Delete recovery data…
              </button>
              <button
                v-if="
                  detail
                    ? detail.can_cancel
                    : admission && !cancelledBeforeAdmission
                "
                type="button"
                class="vjd__danger"
                :disabled="busy"
                @click="action('cancel')"
              >
                Stop video
              </button>
            </div>
          </footer>
        </div>
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
.video-job-overlay :deep(.ms-modal) {
  padding: 12px;
}
.video-job-overlay :deep(.ms-modal__panel) {
  max-width: 100%;
  max-height: 100%;
  display: flex;
  flex-direction: column;
}
.video-job-overlay :deep(.ms-modal__body) {
  min-height: 0;
  overflow: auto;
}
.video-job-control {
  min-height: 44px;
  font: inherit;
  color: var(--mold-text);
  background: var(--mold-bg);
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
  padding: 8px 12px;
}
/* The same anatomy and button styles as QueueEntryDetail (`.qed`), so a video
 * parent's details read like any other job's. */
.vjd {
  display: flex;
  min-width: 0;
  flex-direction: column;
  color: var(--mold-text-2, currentColor);
}
.vjd__head {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--mold-border, var(--mold-border-control));
}
.vjd__state {
  padding: 2px 6px;
  border: 1px solid var(--line, var(--mold-border-control));
  border-radius: var(--mold-radius-2);
  font-size: var(--mold-fs-micro);
  letter-spacing: 0.04em;
  white-space: nowrap;
}
.vjd__state:is(
  [data-state="rendering"],
  [data-state="joining"],
  [data-state="stopping"]
) {
  border-color: var(--mold-blue);
  color: var(--mold-blue);
}
.vjd__state:is([data-state="paused"], [data-state="interrupted"]) {
  border-color: var(--mold-warning);
  color: var(--mold-warning);
}
.vjd__title {
  overflow: hidden;
  min-width: 0;
  flex: 1;
  margin: 0;
  color: var(--mold-text);
  font-size: var(--mold-fs-md);
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.vjd__close {
  min-width: 44px;
  min-height: 44px;
  border: 0;
  background: none;
  color: inherit;
  cursor: pointer;
}
.vjd__body {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 12px;
  padding: 14px 16px;
}
.vjd__caption {
  margin: 0;
  color: var(--mold-text);
  font-weight: 600;
}
.vjd__facts {
  display: grid;
  margin: 0;
  gap: 6px 12px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.vjd__facts dt {
  color: var(--mold-text-dim, currentColor);
  font-size: var(--mold-fs-xs);
}
.vjd__facts dd {
  margin: 0;
  overflow-wrap: anywhere;
}
.vjd__problem {
  margin: 0;
  padding: 10px;
  border: 1px solid var(--mold-error);
  border-radius: var(--mold-radius-2);
  color: var(--mold-error);
  white-space: pre-wrap;
  word-break: break-word;
}
.vjd__technical {
  min-width: 0;
  user-select: text;
}
.vjd__technical summary {
  cursor: pointer;
}
.vjd__technical pre {
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-size: var(--mold-fs-xs);
}
.vjd__foot {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 8px;
  padding: 12px 16px;
  border-top: 1px solid var(--mold-border, var(--mold-border-control));
}
.vjd__error {
  margin: 0;
  color: var(--mold-error);
  word-break: break-word;
}
.vjd__hint {
  margin: 0;
  color: var(--mold-text-dim, currentColor);
  line-height: 1.4;
}
.vjd__confirm {
  display: grid;
  gap: 8px;
}
.vjd__actions {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  gap: 8px;
}
/* Direct children only: ParentVideoRestart's own dialog renders beside its
 * trigger and keeps its own buttons. */
.vjd__actions > button,
.vjd__actions > :deep(button) {
  min-height: 44px;
  flex: 1 1 auto;
  padding: 0 10px;
  border: 1px solid var(--line, var(--mold-border-control));
  border-radius: var(--mold-radius-2);
  background: none;
  color: inherit;
  font: inherit;
  cursor: pointer;
}
.vjd__actions > button:disabled {
  cursor: default;
  opacity: 0.5;
}
.vjd__actions > .vjd__danger {
  border-color: var(--mold-error);
  color: var(--mold-error);
}
</style>
