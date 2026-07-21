import { createApp } from "vue";
import App from "./App.vue";
import { router } from "./router";
import { installTheme } from "./lib/theme";
import "./style.css";

installTheme();

const app = createApp(App);
app.use(router);
app.mount("#app");
