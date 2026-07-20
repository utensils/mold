import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./MobileApp.vue";
import "../styles/base.css";
import "./mobile.css";
import { installSystemThemeColorSync } from "../lib/theme";
import { applyMobileSettings, loadMobileSettings } from "./settings";

document.documentElement.classList.add("mobile-surface");
applyMobileSettings(loadMobileSettings());
installSystemThemeColorSync();
createApp(App).use(createPinia()).mount("#app");
