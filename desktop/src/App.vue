<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import { useRouter } from "vue-router";
import TitleBar from "./components/shell/TitleBar.vue";
import NavRail from "./components/shell/NavRail.vue";
import BenchRail from "./components/shell/BenchRail.vue";
import Toasts from "./components/shell/Toasts.vue";
import { resolveShellShortcut } from "./lib/shortcuts";
import { useConnectionStore } from "./stores/connection";
import { useGenerationStore } from "./stores/generation";
import { useToastStore } from "./stores/toasts";

const router = useRouter();
const sidebarOpen = ref(true);
const connection = useConnectionStore();
const generation = useGenerationStore();
const toasts = useToastStore();

function onKeydown(e: KeyboardEvent) {
  const action = resolveShellShortcut(e);
  if (!action) return;
  e.preventDefault();
  if (action.kind === "navigate") void router.push(action.route);
  else if (action.kind === "toggle-sidebar") sidebarOpen.value = !sidebarOpen.value;
  else if (action.kind === "cancel-job") {
    const job = generation.active;
    if (job && job.status !== "complete" && job.status !== "error") {
      void generation.cancel().then(() => toasts.push("Cancelled"));
    }
  }
  // command-palette lands with the ⌘K work in a later milestone.
}

onMounted(async () => {
  window.addEventListener("keydown", onKeydown);
  void connection.init();
  // The window starts hidden (tauri.conf.json visible:false) to avoid a
  // white flash; reveal it once the shell has mounted. No-op in a browser.
  if ("__TAURI_INTERNALS__" in window) {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    await getCurrentWindow().show();
  }
});
onUnmounted(() => window.removeEventListener("keydown", onKeydown));
</script>

<template>
  <div class="grid h-full grid-rows-[44px_1fr_28px]">
    <TitleBar />
    <div class="grid min-h-0 grid-cols-[auto_1fr]">
      <NavRail v-if="sidebarOpen" />
      <main class="min-h-0 min-w-0 overflow-hidden">
        <router-view />
      </main>
    </div>
    <BenchRail />
    <Toasts />
  </div>
</template>
