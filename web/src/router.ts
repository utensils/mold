import { createRouter, createWebHistory, type RouteRecordRaw } from "vue-router";
import GalleryPage from "./pages/GalleryPage.vue";
import GeneratePage from "./pages/GeneratePage.vue";

const routes: RouteRecordRaw[] = [
  { path: "/", name: "gallery", component: GalleryPage },
  { path: "/generate", name: "generate", component: GeneratePage },
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
});
