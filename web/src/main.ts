import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./App.vue";
import { router } from "./router";
import { installTheme } from "./lib/theme";
import { installStudioPlatform } from "@studio/platform";
import { browserPlatform } from "./platform/browser";
import "./style.css";

installTheme();
installStudioPlatform(browserPlatform);

const app = createApp(App);
app.use(createPinia());
app.use(router);
app.mount("#app");
