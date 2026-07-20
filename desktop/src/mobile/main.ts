import { createApp } from "vue";
import { createPinia } from "pinia";
import App from "./MobileApp.vue";
import "../styles/base.css";
import "./mobile.css";

document.documentElement.classList.add("mobile-surface");
createApp(App).use(createPinia()).mount("#app");
