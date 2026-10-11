import { createApp } from "vue";
import { createPinia } from "pinia";
import OriginAccess from "./OriginAccess.vue";
import { router } from "./router";
import { installTheme } from "./lib/theme";
import { retireSequenceStorage } from "@studio/lib/retireSequenceStorage";
import "./style.css";
import { configureSecureOrigin } from "@studio/api/relayTransport";
import { standaloneStudio, studioPrimary } from "./lib/secureStudio";
import { configureSecureMediaWorker } from "@studio/api/secureMedia";

configureSecureMediaWorker(`${import.meta.env.BASE_URL}secure-media-sw.js`);

if (standaloneStudio())
  configureSecureOrigin(() => studioPrimary()?.url ?? null);
installTheme();
// Scene-by-scene authoring is gone; its draft and its IndexedDB media are not
// until something frees them. Fire and forget — nothing on screen waits on it.
void retireSequenceStorage();

const app = createApp(OriginAccess);
app.use(createPinia());
app.use(router);
app.mount("#app");
