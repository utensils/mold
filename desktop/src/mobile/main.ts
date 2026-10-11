import { installSecureNative } from "../lib/secureNative";
import { configureSecureMediaWorker } from "@studio/api/secureMedia";
configureSecureMediaWorker(`${import.meta.env.BASE_URL}secure-media-sw.js`);
import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./MobileApp.vue";
import { isNativeAndroidRuntime, isNativeIOSRuntime } from "./platform";
import "./foundation.css";
import "./mobile.css";
import { installSystemThemeSync } from "../lib/theme";
import { applyMobileSettings, loadMobileSettings } from "./settings";

document.documentElement.classList.add("mobile-surface");
document.documentElement.classList.toggle("native-ios", isNativeIOSRuntime());
document.documentElement.classList.toggle("native-android", isNativeAndroidRuntime());
applyMobileSettings(loadMobileSettings());
installSystemThemeSync(() => {
  const { theme, matchSystem } = loadMobileSettings();
  return { theme, matchSystem };
});
void installSecureNative()
  .catch(() => {
    // The normal pairing UI reports unreadable or expired identities by name.
  })
  .finally(() => {
    createApp(App).use(createPinia()).mount("#app");
  });
