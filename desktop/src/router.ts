import { createRouter, createWebHistory } from "vue-router";

// `meta.title` names the active workspace in the unified titlebar
// ("Mold Studio — {title}"). The information architecture collapses to five
// destinations — Create, Library, Models, Machines, Settings — with every
// legacy path kept as a redirect so deep-links and a persisted last-route keep
// resolving to their new home.
export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/create" },
    {
      path: "/create",
      name: "create",
      meta: { title: "Create" },
      component: () => import("./views/GenerateView.vue"),
    },
    {
      // Multi-prompt chains are authored inside Create; the view is unchanged.
      path: "/create/chain",
      name: "chains",
      meta: { title: "Create" },
      component: () => import("./views/ChainsView.vue"),
    },
    {
      path: "/library",
      name: "library",
      meta: { title: "Library" },
      component: () => import("./views/LibraryView.vue"),
    },
    {
      path: "/models",
      name: "models",
      meta: { title: "Models" },
      component: () => import("./views/ModelsView.vue"),
    },
    {
      path: "/jobs",
      name: "jobs",
      meta: { title: "Jobs" },
      component: () => import("./views/JobsView.vue"),
    },
    {
      path: "/machines",
      name: "machines",
      meta: { title: "Machines" },
      component: () => import("./views/MachinesView.vue"),
    },
    {
      // RunPod provisioning lives under Machines. Declared before the
      // `/machines/:id` host-detail route so it wins the literal segment.
      path: "/machines/runpod",
      name: "runpod",
      meta: { title: "RunPod" },
      component: () => import("./views/RunPodView.vue"),
    },
    {
      path: "/machines/:id",
      name: "host-detail",
      meta: { title: "Host" },
      component: () => import("./views/HostDetailView.vue"),
    },
    {
      path: "/settings",
      name: "settings",
      meta: { title: "Settings" },
      component: () => import("./views/SettingsView.vue"),
    },
    // Legacy paths fold into the five destinations so existing links,
    // deep-links, and a persisted last-route keep working. `router.replace`
    // during restore runs these redirects too (restoring "/gallery" lands on
    // "/library", "/history" opens the Library history drawer).
    { path: "/generate", redirect: "/create" },
    { path: "/gallery", redirect: (to) => ({ path: "/library", query: to.query }) },
    { path: "/chains", redirect: "/create/chain" },
    { path: "/history", redirect: { path: "/library", query: { panel: "history" } } },
    { path: "/hosts/:id", redirect: (to) => `/machines/${to.params.id}` },
    { path: "/runpod", redirect: "/machines/runpod" },
  ],
});
