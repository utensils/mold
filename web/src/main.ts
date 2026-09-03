import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./App.vue";
import { router } from "./router";
import { installTheme } from "./lib/theme";
import "./style.css";

installTheme();

const app = createApp(App);
app.use(createPinia());
app.use(router);
app.mount("#app");
