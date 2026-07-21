import { createRouter, createWebHistory } from "vue-router";

// `meta.title` names the active workspace in the unified titlebar
// ("Mold Studio — {title}"). The 8 destinations are unchanged; only the
// chrome that frames them is new.
export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/generate" },
    {
      path: "/generate",
      name: "generate",
      meta: { title: "Generate" },
      component: () => import("./views/GenerateView.vue"),
    },
    {
      path: "/gallery",
      name: "gallery",
      meta: { title: "Gallery" },
      component: () => import("./views/GalleryView.vue"),
    },
    {
      path: "/chains",
      name: "chains",
      meta: { title: "Chains" },
      component: () => import("./views/ChainsView.vue"),
    },
    {
      path: "/models",
      name: "models",
      meta: { title: "Catalog" },
      component: () => import("./views/ModelsView.vue"),
    },
    {
      path: "/history",
      name: "history",
      meta: { title: "History" },
      component: () => import("./views/HistoryView.vue"),
    },
    {
      path: "/jobs",
      name: "jobs",
      meta: { title: "Jobs" },
      component: () => import("./views/JobsView.vue"),
    },
    {
      path: "/hosts/:id",
      name: "host-detail",
      meta: { title: "Host" },
      component: () => import("./views/HostDetailView.vue"),
    },
    {
      path: "/runpod",
      name: "runpod",
      meta: { title: "RunPod" },
      component: () => import("./views/RunPodView.vue"),
    },
    {
      path: "/settings",
      name: "settings",
      meta: { title: "Settings" },
      component: () => import("./views/SettingsView.vue"),
    },
  ],
});
