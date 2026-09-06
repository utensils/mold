import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./App.vue";
import { router } from "./router";
import { installTheme } from "./lib/theme";
import { retireSequenceStorage } from "@studio/lib/retireSequenceStorage";
import "./style.css";

installTheme();
// Scene-by-scene authoring is gone; its draft and its IndexedDB media are not
// until something frees them. Fire and forget — nothing on screen waits on it.
void retireSequenceStorage();

const app = createApp(App);
app.use(createPinia());
app.use(router);
app.mount("#app");
