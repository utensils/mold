import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./App.vue";
import { applyPlatformAttribute } from "./lib/platform";
import { router } from "./router";
import { retireSequenceStorage } from "@studio/lib/retireSequenceStorage";
import "./styles/base.css";

applyPlatformAttribute(document.documentElement);
// Scene-by-scene authoring is gone; its draft and its IndexedDB media are not
// until something frees them. Fire and forget — nothing on screen waits on it.
void retireSequenceStorage();
createApp(App).use(createPinia()).use(router).mount("#app");
