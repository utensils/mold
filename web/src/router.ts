import {
  createRouter,
  createWebHistory,
  type RouteRecordRaw,
} from "vue-router";
import GalleryPage from "./pages/GalleryPage.vue";
import GeneratePage from "./pages/GeneratePage.vue";
import CatalogPage from "./pages/CatalogPage.vue";
import MachinesPage from "./pages/MachinesPage.vue";
import MachineDetailPage from "./pages/MachineDetailPage.vue";
import SettingsPage from "./pages/SettingsPage.vue";

// Mold Studio IA (spec §04): four workspaces plus Settings. Route names match
// the nav pills — gallery / create / models / machines — with Settings pushed
// from the account affordance. Legacy paths (/generate, /catalog) redirect so
// bookmarks and older links keep resolving.
export const routes: RouteRecordRaw[] = [
  { path: "/", name: "gallery", component: GalleryPage },
  { path: "/create", name: "create", component: GeneratePage },
  { path: "/generate", redirect: { name: "create" } },
  { path: "/models", name: "models", component: CatalogPage },
  { path: "/catalog", redirect: { name: "models" } },
  { path: "/machines", name: "machines", component: MachinesPage },
  {
    path: "/machines/:id",
    name: "machine-detail",
    component: MachineDetailPage,
  },
  { path: "/settings", name: "settings", component: SettingsPage },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});
