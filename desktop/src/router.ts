import { createRouter, createWebHistory } from "vue-router";

// `meta.title` names the active destination in the unified titlebar. The
// sidebar's destinations are New image, Queue, My images, Styles, and
// Machines (⌘1–⌘5) plus Settings (⌘,), with every legacy path kept as a
// redirect so deep-links and a persisted last-route keep resolving.
export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/create" },
    {
      path: "/create",
      name: "create",
      meta: { title: "New image" },
      component: () => import("./views/GenerateView.vue"),
    },
    {
      path: "/queue",
      name: "queue",
      meta: { title: "Queue" },
      component: () => import("./views/QueueView.vue"),
    },
    {
      path: "/library",
      name: "library",
      meta: { title: "My images" },
      component: () => import("./views/LibraryView.vue"),
    },
    {
      path: "/models",
      name: "models",
      meta: { title: "Styles" },
      component: () => import("./views/ModelsView.vue"),
    },
    {
      // Master/detail (README §03): the machine list frames every pane, so
      // the titlebar keeps saying Machines on a machine and on Rent a GPU.
      path: "/machines",
      name: "machines",
      meta: { title: "Machines" },
      component: () => import("./views/MachinesView.vue"),
      children: [
        {
          // Declared before `:id` so it wins the literal segment.
          path: "runpod",
          name: "runpod",
          component: () => import("./views/RunPodView.vue"),
        },
        {
          path: ":id",
          name: "host-detail",
          component: () => import("./views/HostDetailView.vue"),
        },
      ],
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
    // "/library", "/history" opens the History column in My images).
    { path: "/generate", redirect: "/create" },
    { path: "/gallery", redirect: (to) => ({ path: "/library", query: to.query }) },
    { path: "/history", redirect: { path: "/library", query: { panel: "history" } } },
    { path: "/jobs", redirect: "/queue" },
    { path: "/hosts/:id", redirect: (to) => `/machines/${to.params.id}` },
    { path: "/runpod", redirect: "/machines/runpod" },
  ],
});
