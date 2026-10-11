import {
  createRouter,
  createWebHistory,
  createWebHashHistory,
  type RouteRecordRaw,
} from "vue-router";
import QueuePage from "./pages/QueuePage.vue";
import CreatePage from "./pages/CreatePage.vue";
import LibraryPage from "./pages/LibraryPage.vue";
import ModelsPage from "./pages/ModelsPage.vue";
import MachinesPage from "./pages/MachinesPage.vue";
import HostDetailPage from "./pages/HostDetailPage.vue";
import SettingsPage from "./pages/SettingsPage.vue";
import NotFoundPage from "./pages/NotFoundPage.vue";
import { captureInitialPairingLink } from "./lib/initialPairing";

// Five browser workspaces retain existing deep links; retired URLs are unknown.
export const routes: RouteRecordRaw[] = [
  { path: "/", redirect: { name: "create" } },
  { path: "/create", name: "create", component: CreatePage },
  { path: "/queue", name: "queue", component: QueuePage },
  { path: "/library", name: "library", component: LibraryPage },
  { path: "/models", name: "models", component: ModelsPage },
  { path: "/machines", name: "machines", component: MachinesPage },
  {
    path: "/machines/:id",
    name: "host-detail",
    component: HostDetailPage,
  },
  { path: "/settings", name: "settings", component: SettingsPage },
  { path: "/:pathMatch(.*)*", name: "not-found", component: NotFoundPage },
];

captureInitialPairingLink();
export const router = createRouter({
  history:
    import.meta.env.VITE_STANDALONE_STUDIO === "1"
      ? createWebHashHistory(import.meta.env.BASE_URL)
      : createWebHistory(import.meta.env.BASE_URL),
  routes,
});
