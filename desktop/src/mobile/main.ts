import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./MobileApp.vue";
import "./legacy.css";
import "./mobile.css";
import { installSystemThemeSync } from "../lib/theme";
import { applyMobileSettings, loadMobileSettings } from "./settings";

document.documentElement.classList.add("mobile-surface");
applyMobileSettings(loadMobileSettings());
installSystemThemeSync(() => {
  const { theme, matchSystem } = loadMobileSettings();
  return { theme, matchSystem };
});
createApp(App).use(createPinia()).mount("#app");
