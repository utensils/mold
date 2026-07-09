import { createRouter, createWebHistory } from "vue-router";

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/generate" },
    { path: "/generate", name: "generate", component: () => import("./views/GenerateView.vue") },
    { path: "/gallery", name: "gallery", component: () => import("./views/GalleryView.vue") },
    { path: "/chains", name: "chains", component: () => import("./views/ChainsView.vue") },
    { path: "/models", name: "models", component: () => import("./views/ModelsView.vue") },
    { path: "/history", name: "history", component: () => import("./views/HistoryView.vue") },
    { path: "/settings", name: "settings", component: () => import("./views/SettingsView.vue") },
  ],
});
