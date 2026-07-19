import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./MobileApp.vue";
import "../styles/base.css";
import "./mobile.css";

createApp(App).use(createPinia()).mount("#app");
