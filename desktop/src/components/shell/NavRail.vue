<script setup lang="ts">
import { computed, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import NavItem from "@ui/components/NavItem.vue";
import Keycap from "@ui/components/Keycap.vue";
import type { IconName } from "@ui/icons";
import logoUrl from "../../assets/logo.png";
import StatusPopover from "./StatusPopover.vue";
import PanelResizeHandle from "./PanelResizeHandle.vue";
import { dragWidth } from "../../lib/panelResize";
import {
  mergeActivity,
  partitionActivity,
  sequenceToVM,
  type ActivityJobVM,
} from "@studio/lib/activity";
import { useGenerationStore, jobStatusCode, railOrder, type Job } from "../../stores/generation";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useChainJobsStore } from "../../stores/chainJobs";
import { useComposerStore } from "../../stores/composer";
import { useContextMenuStore, type MenuEntry } from "../../stores/contextMenu";
import { useGalleryStore } from "../../stores/gallery";
import { useHostsStore } from "../../stores/hosts";
import { useHostModelsStore } from "../../stores/hostModels";
import { useToastStore } from "../../stores/toasts";
import { badgeCount } from "../../lib/notifications";
import { shortcutLabel } from "../../lib/platform";
import { modelDisplayNameForId } from "../../lib/models";

const route = useRoute();
const router = useRouter();
const appPrefs = useAppPrefsStore();
const generation = useGenerationStore();
const chains = useChainJobsStore();
const composer = useComposerStore();
const contextMenu = useContextMenuStore();
const hosts = useHostsStore();
const hostModels = useHostModelsStore();
const gallery = useGalleryStore();
const toasts = useToastStore();

// Workspace badges (§08 G11): Library counts prints developed since the last
// visit; Machines shows a stop dot when any connected host is offline.
const libraryBadge = computed(() => badgeCount(gallery.newCount));
const machinesErrored = computed(() => hosts.all.some((h) => h.status === "error"));

// Collapse snaps to the 62px icon rail; expanded width is drag-resizable at
// the right edge and persists via appPrefs.navRailWidth (double-click resets
// to the 210px default).
const collapsed = computed(() => appPrefs.sidebarCollapsed);

// Live width while dragging; null follows the persisted preference.
// Persistence happens only on commit, never per pointermove.
const draftRailWidth = ref<number | null>(null);
const railWidth = computed(() =>
  collapsed.value ? 62 : (draftRailWidth.value ?? appPrefs.navRailWidth),
);

function onRailResize(dx: number) {
  draftRailWidth.value = dragWidth("navRail", appPrefs.navRailWidth, dx, "right");
}

async function onRailCommit() {
  const width = draftRailWidth.value;
  if (width === null) return;
  if (width !== appPrefs.navRailWidth) await appPrefs.update({ navRailWidth: width });
  draftRailWidth.value = null;
}

function onRailReset() {
  draftRailWidth.value = null;
  void appPrefs.update({ navRailWidth: null });
}

// Host management, discovery, and per-host actions live entirely in the
// Machines workspace — the rail keeps only the Machines destination (with
// its offline stop-dot badge) and the status popover.

interface Destination {
  route: string;
  label: string;
  icon: IconName;
}

// Five destinations (spec §04). Chains fold into Create, History into
// Library's drawer, Jobs/RunPod into Machines — all still reachable by
// deep-link and the command palette. Settings is pinned separately at the
// bottom. ⌘1–⌘4 track this order (see lib/shortcuts).
const destinations: Destination[] = [
  { route: "/create", label: "Create", icon: "create" },
  { route: "/library", label: "Library", icon: "library" },
  { route: "/models", label: "Models", icon: "models" },
  { route: "/machines", label: "Machines", icon: "machines" },
];

function isActive(path: string): boolean {
  return route.path === path;
}

/** Queue order: every live job first (submission order), then the freshest
 *  finished prints — the rail is a working queue, not a full history. */
const railJobs = computed<Job[]>(() => {
  const live = railOrder(
    generation.jobs.filter((j) => j.status !== "complete" && j.status !== "error"),
  );
  const done = generation.jobs
    .filter((j) => j.status === "complete" || j.status === "error")
    .slice(-3)
    .reverse();
  return [...live, ...done];
});

/** Now developing also means sequences (G14): the rail read `generation.jobs`
 *  only, so a running sequence rendered on the canvas while the sidebar said
 *  "nothing developing". Live work only — the last-3 finished window stays
 *  prints-only, because settled sequences already have two homes (the print in
 *  Library, the job in Library ▸ History ▸ Sequences). */
const railSequences = computed(() =>
  partitionActivity(
    mergeActivity(
      [],
      chains.allJobs.map(({ hostId, job }) =>
        sequenceToVM(job, {
          hostId,
          hostLabel: hosts.all.find((h) => h.id === hostId)?.label ?? hostId,
        }),
      ),
    ),
  ).active.filter((vm): vm is ActivityJobVM & { kind: "sequence" } => vm.kind === "sequence"),
);

const developingCount = computed(() => generation.pending.length + railSequences.value.length);

/** `clip 3/5 · developing…` — the sequence's answer to developingLabel. */
function sequenceLine(vm: ActivityJobVM & { kind: "sequence" }): string {
  const clip = Math.min(vm.currentStage + 1, vm.stageCount);
  return vm.state === "queued"
    ? `clip ${clip}/${vm.stageCount} · queued`
    : `clip ${clip}/${vm.stageCount} · developing…`;
}
const modelLabel = (name: string) => modelDisplayNameForId(name, hostModels.unionInstalled);

function jobRunning(job: Job): boolean {
  return job.status === "denoising" || job.status === "finishing" || job.status === "loading";
}

/** Lowercase mono progress line for the developing strip. */
function developingLabel(job: Job): string {
  if (job.status === "denoising") return `developing ${job.step}/${job.total}`;
  return jobStatusCode(job).toLowerCase();
}

function jobMenu(job: Job): MenuEntry[] {
  const live = job.status !== "complete" && job.status !== "error";
  return [
    {
      label: "Cancel",
      danger: true,
      disabled: !live,
      action: () => void generation.cancel(job.clientId).then(() => toasts.push("Cancelled")),
    },
    { separator: true },
    {
      label: "Use prompt",
      action: () => {
        composer.set({
          prompt: job.prompt,
          model: job.model,
          seed: null,
          width: job.width,
          height: job.height,
          steps: job.total,
          guidance: job.guidance,
        });
        void router.push("/create");
      },
    },
    {
      label: "Show in library",
      disabled: job.status !== "complete",
      action: () => void router.push("/library"),
    },
    { separator: true },
    {
      label: "Clear finished",
      disabled: !generation.jobs.some((j) => j.status === "complete" || j.status === "error"),
      action: () => generation.prune(0),
    },
  ];
}
</script>

<template>
  <nav
    class="nav-rail relative flex shrink-0 flex-col border-r border-edge bg-bench pt-3.5 pb-3"
    :class="collapsed ? 'px-1.5' : 'px-2.5'"
    :style="{ width: `${railWidth}px` }"
    aria-label="Primary"
  >
    <PanelResizeHandle
      v-if="!collapsed"
      class="absolute inset-y-0 -right-0.5 z-10"
      label="Resize sidebar"
      @resize="onRailResize"
      @commit="onRailCommit"
      @reset="onRailReset"
    />
    <!-- header: logo + gradient wordmark + STUDIO kicker -->
    <div class="mb-4 flex items-center gap-2.5 px-2" :class="collapsed ? 'justify-center' : ''">
      <img :src="logoUrl" alt="mold" class="h-6 w-6 shrink-0 object-contain" />
      <span v-if="!collapsed" class="ms-wordmark select-none">mold</span>
      <span v-if="!collapsed" class="rail-studio mt-[3px] select-none">STUDIO</span>
    </div>

    <!-- nav destinations -->
    <div class="flex flex-col gap-[3px]">
      <div v-for="d in destinations" :key="d.route" class="relative">
        <NavItem
          :icon="d.icon"
          :label="d.label"
          :collapsed="collapsed"
          :active="isActive(d.route)"
          :badge="d.route === '/library' ? (libraryBadge ?? '') : ''"
          @select="router.push(d.route)"
        />
        <span
          v-if="d.route === '/machines' && machinesErrored"
          data-test="machines-error-dot"
          class="pointer-events-none absolute top-2 right-2 h-2 w-2 rounded-full bg-stop"
        />
      </div>
    </div>

    <!-- now developing -->
    <div v-if="!collapsed" data-test="developing-region" class="flex min-h-0 flex-1 flex-col">
      <div class="mt-5 mb-1.5 flex items-center gap-2 px-3">
        <span class="rail-kicker">Now developing</span>
        <span v-if="developingCount > 1" class="rail-kicker ml-auto">
          {{ developingCount }}
        </span>
      </div>
      <div
        v-if="railJobs.length > 0 || railSequences.length > 0"
        data-test="developing-jobs"
        class="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-2"
      >
        <a
          v-for="vm in railSequences"
          :key="vm.key"
          href="#"
          data-test="developing-sequence"
          class="flex items-center gap-2.5 rounded-[8px] px-1 py-1 hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)]"
          @click.prevent="router.push('/create')"
        >
          <span
            class="h-[30px] w-[30px] shrink-0 overflow-hidden rounded-[6px] border border-[color-mix(in_srgb,var(--rebate)_12%,transparent)] bg-print-surface"
          >
            <span
              v-if="vm.state === 'running'"
              class="ms-shimmer block h-full w-full"
              aria-hidden="true"
            />
            <span v-else class="block h-full w-full bg-print-surface" aria-hidden="true" />
          </span>
          <span class="min-w-0 flex-1">
            <span class="block truncate text-[11.5px] text-ink-2" :title="vm.model">
              {{ modelLabel(vm.model) }} · {{ vm.hostLabel }}
            </span>
            <span class="block font-utility text-[9.5px] text-safelight">
              {{ sequenceLine(vm) }}
            </span>
          </span>
        </a>
        <a
          v-for="job in railJobs"
          :key="job.clientId"
          href="#"
          class="flex items-center gap-2.5 rounded-[8px] px-1 py-1 hover:bg-[color-mix(in_srgb,var(--rebate)_6%,transparent)]"
          @click.prevent="router.push('/create')"
          @contextmenu.prevent="contextMenu.open($event, jobMenu(job))"
        >
          <span
            class="h-[30px] w-[30px] shrink-0 overflow-hidden rounded-[6px] border border-[color-mix(in_srgb,var(--rebate)_12%,transparent)] bg-print-surface"
          >
            <img
              v-if="job.resultUrl && !job.result?.video_frames"
              :src="job.resultUrl"
              alt=""
              class="h-full w-full object-cover"
            />
            <img
              v-else-if="job.previewUrl"
              :src="job.previewUrl"
              alt=""
              class="h-full w-full object-cover"
              style="filter: blur(1px)"
            />
            <span v-else-if="jobRunning(job)" class="ms-shimmer block h-full w-full" />
            <span v-else class="block h-full w-full bg-print-surface" />
          </span>
          <span class="min-w-0 flex-1">
            <span class="block truncate text-[11.5px] text-ink-2" :title="job.prompt">
              {{ modelLabel(job.model)
              }}<template v-if="job.hostLabel"> · {{ job.hostLabel }}</template>
            </span>
            <span
              class="block font-utility text-[9.5px]"
              :class="job.status === 'error' ? 'text-stop' : 'text-safelight'"
            >
              {{ developingLabel(job) }}
            </span>
          </span>
        </a>
      </div>
      <p v-else class="px-3 text-caption text-ink-3">nothing developing</p>
    </div>
    <div v-else class="flex-1" />

    <!-- status + settings -->
    <StatusPopover :collapsed="collapsed" />
    <div class="relative mt-1">
      <NavItem
        icon="settings"
        label="Settings"
        :collapsed="collapsed"
        :active="isActive('/settings')"
        @select="router.push('/settings')"
      />
      <Keycap
        v-if="!collapsed"
        class="pointer-events-none absolute top-1/2 right-3 -translate-y-1/2"
      >
        {{ shortcutLabel(",") }}
      </Keycap>
    </div>
  </nav>
</template>

<style scoped>
.nav-rail {
  transition: width var(--dur-base) var(--ease);
}

.ms-wordmark {
  font-family: var(--f-display);
  font-weight: 800;
  font-size: 17px;
  letter-spacing: -0.01em;
  line-height: 1;
  background: var(--grad);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.rail-studio {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.16em;
  color: var(--ink-3);
}

.rail-kicker {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-3);
  white-space: nowrap;
}
</style>
