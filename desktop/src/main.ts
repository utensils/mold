import { installSecureNative } from "./lib/secureNative";
import { configureSecureMediaWorker } from "@studio/api/secureMedia";
configureSecureMediaWorker(`${import.meta.env.BASE_URL}secure-media-sw.js`);
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
void installSecureNative()
  .catch(() => {
    // The normal pairing UI reports unreadable or expired identities by name.
  })
  .finally(() => {
    createApp(App).use(createPinia()).use(router).mount("#app");
  });
