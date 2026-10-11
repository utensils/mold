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
// Hooks are installed synchronously before the first await. Stored credentials
// warm in the background: an OS storage prompt must not hide the whole app.
void installSecureNative().catch(() => {
  // Secure operations still await and validate their own current profile.
});
createApp(App).use(createPinia()).mount("#app");
