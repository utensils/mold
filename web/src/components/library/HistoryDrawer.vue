<script setup lang="ts">
/*
 * Library ▸ History drawer (web). Today it carries one lens, **Sequences** —
 * the durable chain jobs across every registered host, which is where settled
 * sequence work resolves now that the Create activity strip is present tense.
 * Runs / Prompts parity with desktop is a separate backlog item, so the tab bar
 * stays hidden while there is only one tab: this reads as a titled drawer, not
 * a lonely tab.
 *
 * Opens via ?panel=history (the Create activity digest deep-links here with
 * ?tab=sequences); LibraryPage owns the open state and closing.
 */
import { computed, ref, watch } from "vue";
import DrawerPanel from "@ui/components/DrawerPanel.vue";
import EmptyStateBlock from "@ui/components/EmptyStateBlock.vue";
import SequenceJobRow from "@ui/components/SequenceJobRow.vue";
import {
  HISTORY_JOBS_RENDER_CAP,
  mergeActivity,
  sequenceToVM,
  type ActivityAction,
  type ActivityJobVM,
} from "@studio/lib/activity";
import { useChainJobs } from "../../composables/useChainJobs";
import { listHosts } from "../../lib/hostRegistry";
import { requestConfirm, toast } from "../../lib/toasts";

const props = defineProps<{ open: boolean }>();
const emit = defineEmits<{
  close: [];
  /** Re-enter a durable job in Create (watch it, or edit its clips). */
  "open-sequence": [payload: { hostId: string; jobId: string; edit: boolean }];
}>();

type SequenceVM = ActivityJobVM & { kind: "sequence" };

const chainJobs = useChainJobs();

/** Chain polling stops when nothing is active, so a settled list goes stale
 *  unless the drawer refetches on open. */
watch(
  () => props.open,
  (open) => {
    if (open) void chainJobs.fetchAll();
  },
  { immediate: true },
);

const hostLabelFor = (hostId: string) =>
  listHosts().find((h) => h.id === hostId)?.name ?? hostId;

const rows = computed<SequenceVM[]>(() => {
  const vms: ActivityJobVM[] = [];
  for (const [hostId, host] of Object.entries(chainJobs.state.byHost)) {
    for (const job of host.jobs) {
      vms.push(sequenceToVM(job, { hostId, hostLabel: hostLabelFor(hostId) }));
    }
  }
  return mergeActivity([], vms).filter(
    (vm): vm is SequenceVM => vm.kind === "sequence",
  );
});

const visible = computed(() => rows.value.slice(0, HISTORY_JOBS_RENDER_CAP));
const capNote = computed(() =>
  rows.value.length > HISTORY_JOBS_RENDER_CAP
    ? `showing ${HISTORY_JOBS_RENDER_CAP} of ${rows.value.length}`
    : null,
);

const timeLabel = (vm: SequenceVM) =>
  new Date(vm.settledAtMs ?? vm.createdAtMs).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });

const fail = (error: unknown) =>
  toast("error", error instanceof Error ? error.message : String(error));

async function onAction(action: ActivityAction, vm: SequenceVM) {
  switch (action) {
    case "watch":
      emit("open-sequence", {
        hostId: vm.hostId,
        jobId: vm.jobId,
        edit: false,
      });
      return;
    case "edit":
      emit("open-sequence", { hostId: vm.hostId, jobId: vm.jobId, edit: true });
      return;
    case "cancel":
      await chainJobs.cancel(vm.hostId, vm.jobId).catch(fail);
      return;
    case "resume":
      await chainJobs.resume(vm.hostId, vm.jobId).catch(fail);
      return;
    case "delete": {
      const ok = await requestConfirm({
        title: "Delete this sequence?",
        body: `Removes the job and its cached clips from ${vm.hostLabel}. The finished video in the Library is kept.`,
        confirmLabel: "Delete",
        danger: true,
      });
      if (ok) await chainJobs.remove(vm.hostId, vm.jobId).catch(fail);
      return;
    }
    default:
      return;
  }
}

function selectSequence(vm: SequenceVM) {
  emit("open-sequence", {
    hostId: vm.hostId,
    jobId: vm.jobId,
    edit: false,
  });
}

// `Clear inactive` is a server DELETE, unlike the strip's client-side dismiss —
// the confirm names the count so the two can never blur together.
const confirmingClear = ref(false);
const clearing = ref(false);

const inactiveCount = computed(
  () => rows.value.filter((vm) => vm.settledAtMs !== null).length,
);

const clearLabel = computed(() =>
  confirmingClear.value
    ? `Delete ${inactiveCount.value} inactive sequence${inactiveCount.value === 1 ? "" : "s"}?`
    : "Clear inactive",
);

async function clearInactive() {
  if (!confirmingClear.value) {
    confirmingClear.value = true;
    return;
  }
  confirmingClear.value = false;
  clearing.value = true;
  try {
    const { cleared, failed } = await chainJobs.clearInactive();
    if (failed > 0)
      toast("error", `${failed} sequence(s) could not be cleared.`);
    else
      toast("info", `Cleared ${cleared} sequence${cleared === 1 ? "" : "s"}.`);
  } catch (error) {
    fail(error);
  } finally {
    clearing.value = false;
  }
}

async function cleanUpDisk() {
  const ok = await requestConfirm({
    title: "Clean up sequence cache?",
    body: "This discards cached scene media used for scene playback and sequence editing on connected machines. Final videos in the Library remain.",
    confirmLabel: "Clean up",
    danger: true,
  });
  if (!ok) return;
  try {
    const outcome = await chainJobs.gc();
    toast(
      "info",
      `Cleaned up ${outcome.pruned_artifact_dirs} artifact folder(s).`,
    );
  } catch (error) {
    fail(error);
  }
}
</script>

<template>
  <!-- Viewport-fixed host: @ui DrawerPanel is `position: absolute; inset: 0`,
       so it fills its nearest positioned ancestor by design. The Library is a
       long scrolling column — mounted inline the drawer draws at the top of
       the page and renders off-screen at any scroll offset. -->
  <div v-if="open" class="fixed inset-0 z-40" data-test="history-drawer-host">
    <DrawerPanel
      :open="open"
      :width="520"
      title="History"
      @close="emit('close')"
    >
      <template #header>
        <span class="hd-kicker">History · Sequences</span>
      </template>

      <div class="flex flex-col gap-2">
        <EmptyStateBlock
          v-if="rows.length === 0"
          data-test="sequences-empty"
          icon="video"
          headline="No sequences yet"
          guidance="A sequence you develop keeps its job here — watch it, edit its clips, or clear it out."
        >
          <template #action>
            <button
              type="button"
              class="rounded-control bg-safelight px-3 py-1.5 text-sm font-semibold text-on-accent"
              @click="
                emit('close');
                $router.push('/create');
              "
            >
              Go to Create
            </button>
          </template>
        </EmptyStateBlock>

        <template v-else>
          <div
            v-for="vm in visible"
            :key="vm.key"
            data-test="sequence-row"
            class="rounded-control px-2 py-1.5 hover:bg-bath"
          >
            <SequenceJobRow
              dense
              :vm="vm"
              :time-label="timeLabel(vm)"
              @action="onAction"
              @select="selectSequence"
            />
          </div>
          <p v-if="capNote" data-test="sequence-cap-note" class="hd-note">
            {{ capNote }}
          </p>
        </template>

        <!-- Maintenance: destructive, host-scoped, and deliberately not in the
             Create composer any more. -->
        <div class="mt-2 flex items-center gap-2 border-t border-edge pt-2">
          <button
            type="button"
            data-test="seq-clear-inactive"
            class="h-7 rounded-control border border-edge px-2.5 text-sm transition-colors"
            :class="
              confirmingClear
                ? 'border-stop bg-stop font-semibold text-on-accent'
                : 'text-ink-2 hover:text-stop'
            "
            :disabled="clearing"
            title="Delete every sequence job that isn't running or queued"
            @blur="confirmingClear = false"
            @click="clearInactive"
          >
            {{ clearLabel }}
          </button>
          <button
            type="button"
            data-test="seq-cleanup-disk"
            class="h-7 rounded-control border border-edge px-2.5 text-sm text-ink-2 transition-colors hover:text-ink"
            title="Sweep ephemeral jobs and prune artifact directories"
            @click="cleanUpDisk"
          >
            Clean up disk
          </button>
        </div>
      </div>
    </DrawerPanel>
  </div>
</template>

<style scoped>
.hd-kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.hd-note {
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
}
</style>
