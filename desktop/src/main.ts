import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./App.vue";
import { applyPlatformAttribute } from "./lib/platform";
import { router } from "./router";
import "./styles/base.css";

applyPlatformAttribute(document.documentElement);
createApp(App).use(createPinia()).use(router).mount("#app");
