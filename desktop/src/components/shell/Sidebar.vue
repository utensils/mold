<script setup lang="ts">
/*
 * The sidebar (README §04): MAKE and SETUP destinations, the target machine's
 * card, and the queue living underneath it — work in progress is context, not
 * content; the image keeps the viewport. Collapses to a 62px icon rail;
 * expanded width is drag-resizable and persists as appPrefs.navRailWidth.
 */
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import NavItem from "@ui/components/NavItem.vue";
import logoUrl from "../../assets/logo.png";
import PanelResizeHandle from "./PanelResizeHandle.vue";
import QueueRail from "./QueueRail.vue";
import { dragWidth } from "../../lib/panelResize";
import { shortcutLabel } from "../../lib/platform";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useDownloadsStore } from "../../stores/downloads";
import { useGalleryStore } from "../../stores/gallery";
import { useHostsStore } from "../../stores/hosts";
import { useHostStatusStore } from "../../stores/hostStatus";
import { useLiveActivityStore } from "../../stores/liveActivity";
import { useQueueActivity } from "../../composables/useQueueActivity";
import { formatGB } from "../../lib/format";

const route = useRoute();
const router = useRouter();
const appPrefs = useAppPrefsStore();
const downloads = useDownloadsStore();
const gallery = useGalleryStore();
const hosts = useHostsStore();
const hostStatus = useHostStatusStore();
const liveActivity = useLiveActivityStore();
const queue = useQueueActivity();

interface Destination {
  route: string;
  label: string;
  icon: "create" | "list" | "library" | "models" | "machines";
  shortcut: string;
}

/** MAKE: the things a person came to do. SETUP: what makes them possible. */
const MAKE: Destination[] = [
  { route: "/create", label: "New image", icon: "create", shortcut: "1" },
  { route: "/queue", label: "Queue", icon: "list", shortcut: "2" },
  { route: "/library", label: "My images", icon: "library", shortcut: "3" },
];
const SETUP: Destination[] = [
  { route: "/models", label: "Styles", icon: "models", shortcut: "4" },
  { route: "/machines", label: "Machines", icon: "machines", shortcut: "5" },
];

const collapsed = computed(() => appPrefs.sidebarCollapsed);

// Live width while dragging; null follows the persisted preference.
// Persistence happens only on commit, never per pointermove.
const draftWidth = ref<number | null>(null);
const width = computed(() => (collapsed.value ? 62 : (draftWidth.value ?? appPrefs.navRailWidth)));

function onResize(dx: number) {
  draftWidth.value = dragWidth("navRail", appPrefs.navRailWidth, dx, "right");
}
async function onCommit() {
  const next = draftWidth.value;
  if (next === null) return;
  if (next !== appPrefs.navRailWidth) await appPrefs.update({ navRailWidth: next });
  draftWidth.value = null;
}
function onReset() {
  draftWidth.value = null;
  void appPrefs.update({ navRailWidth: null });
}

function isActive(path: string): boolean {
  return route.path === path || (path !== "/create" && route.path.startsWith(`${path}/`));
}

// Trailing readouts: prints developed since the last visit badge My images
// (G11), downloads in flight badge Styles, an unreachable host dots Machines.
const newPrints = computed(() => gallery.newCount);
const pictureCount = computed(() => gallery.basePrintCount);
const downloading = computed(() => downloads.hostedInFlight.length);
const machinesErrored = computed(() => hosts.all.some((h) => h.status === "error"));

const machine = computed(() => hostStatus.displayHost);
const machineTone = computed(() => {
  switch (hostStatus.connection) {
    case "error":
      return "bg-error";
    case "connecting":
      return "bg-sapphire ms-pulse";
    case "idle":
      return "bg-fg-dim";
    default:
      return "bg-success";
  }
});

onMounted(() => {
  if (!import.meta.env.TEST) liveActivity.start();
});
onBeforeUnmount(() => {
  if (!import.meta.env.TEST) liveActivity.stop();
});
</script>

<template>
  <nav
    class="sidebar relative flex min-h-0 shrink-0 flex-col overflow-hidden border-r border-border bg-chrome pt-3 pb-3"
    :class="collapsed ? 'px-1.5' : 'px-2.5'"
    :style="{ width: `${width}px` }"
    aria-label="Primary"
  >
    <PanelResizeHandle
      v-if="!collapsed"
      class="absolute inset-y-0 -right-0.5 z-10"
      label="Resize sidebar"
      @resize="onResize"
      @commit="onCommit"
      @reset="onReset"
    />

    <!-- wordmark -->
    <div
      class="mb-3 flex shrink-0 items-center gap-2 px-2 pb-1"
      :class="collapsed ? 'justify-center' : ''"
    >
      <img :src="logoUrl" alt="mold" class="h-[22px] w-[22px] shrink-0 object-contain" />
      <template v-if="!collapsed">
        <span class="font-mono text-base font-bold text-fg select-none">mold</span>
        <span class="font-mono text-micro text-fg-faint select-none">studio</span>
      </template>
    </div>

    <!-- MAKE -->
    <div v-if="!collapsed" class="ms-group-label shrink-0 px-[9px] pt-1.5 pb-1 uppercase">Make</div>
    <div class="flex flex-col gap-0.5">
      <NavItem
        v-for="d in MAKE"
        :key="d.route"
        :icon="d.icon"
        :label="d.label"
        :collapsed="collapsed"
        :active="isActive(d.route)"
        :badge="
          d.route === '/queue' && queue.liveCount.value > 0
            ? queue.liveCount.value
            : d.route === '/library' && newPrints > 0
              ? newPrints
              : ''
        "
        @select="router.push(d.route)"
      >
        <template #trailing>
          <span v-if="d.route === '/create'" class="opacity-70">{{ shortcutLabel("N") }}</span>
          <span v-else-if="d.route === '/library' && newPrints === 0">{{ pictureCount }}</span>
        </template>
      </NavItem>
    </div>

    <!-- SETUP -->
    <div v-if="!collapsed" class="ms-group-label mt-3 shrink-0 px-[9px] pt-1.5 pb-1 uppercase">
      Setup
    </div>
    <div class="flex flex-col gap-0.5" :class="collapsed ? 'mt-2' : ''">
      <div v-for="d in SETUP" :key="d.route" class="relative">
        <span
          v-if="collapsed && d.route === '/machines' && machinesErrored"
          data-test="machines-error-dot"
          class="pointer-events-none absolute top-1.5 right-1.5 z-10 h-2 w-2 rounded-full bg-error"
        />
        <NavItem
          :icon="d.icon"
          :label="d.label"
          :collapsed="collapsed"
          :active="isActive(d.route)"
          @select="router.push(d.route)"
        >
          <template #trailing>
            <span
              v-if="d.route === '/models' && downloading > 0"
              data-test="styles-downloading"
              class="text-warning"
              >{{ downloading }}↓</span
            >
            <span
              v-else-if="d.route === '/machines'"
              data-test="machines-dot"
              class="h-[7px] w-[7px] rounded-full"
              :class="machinesErrored ? 'bg-error' : 'bg-success'"
            />
          </template>
        </NavItem>
      </div>
    </div>

    <!-- target machine -->
    <button
      v-if="!collapsed && machine"
      type="button"
      data-test="machine-card"
      class="mt-3 flex shrink-0 flex-col gap-1.5 rounded-control border border-border bg-surface px-2.5 py-2 text-left transition-colors duration-100 hover:border-border-focus"
      @click="router.push(machine.primary ? '/machines' : `/machines/${machine.id}`)"
    >
      <span class="flex items-center gap-2">
        <span class="h-[7px] w-[7px] shrink-0 rounded-full" :class="machineTone" />
        <span class="truncate font-mono text-xs font-bold text-fg">{{ machine.label }}</span>
        <span class="flex-1" />
        <span v-if="hostStatus.gpus.length" class="font-mono text-micro text-fg-dim">
          {{ hostStatus.vramPct }}%
        </span>
      </span>
      <span class="text-xs leading-snug text-fg-dim">{{ hostStatus.sentence }}</span>
      <span class="block h-[5px] overflow-hidden bg-bg-crust" aria-hidden="true">
        <span
          class="block h-full"
          :class="hostStatus.vramCritical ? 'bg-error' : 'bg-accent'"
          :style="{ width: `${hostStatus.vramPct}%` }"
        />
      </span>
      <span v-if="hostStatus.gpus.length" class="sr-only">
        {{ formatGB(hostStatus.vramUsed) }} of {{ formatGB(hostStatus.vramTotal) }} graphics memory
        in use
      </span>
    </button>

    <!-- queue lives here, under the machine -->
    <QueueRail v-if="!collapsed" class="mt-2.5 min-h-0 flex-1" />
    <div v-else class="flex-1" />

    <!-- settings -->
    <div class="mt-2 shrink-0">
      <NavItem
        icon="settings"
        label="Settings"
        :collapsed="collapsed"
        :active="isActive('/settings')"
        @select="router.push('/settings')"
      >
        <template #trailing>
          <span class="opacity-70">{{ shortcutLabel(",") }}</span>
        </template>
      </NavItem>
    </div>
  </nav>
</template>

<style scoped>
.sidebar {
  transition: width var(--mold-dur-base) var(--mold-ease-out);
}
</style>
