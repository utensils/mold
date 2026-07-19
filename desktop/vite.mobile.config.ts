import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  clearScreen: false,
  root: ".",
  server: { host: "0.0.0.0", port: 1431, strictPort: true },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  build: {
    target: "safari17",
    outDir: "dist-mobile",
    emptyOutDir: true,
    minify: "esbuild",
    sourcemap: !!process.env.TAURI_ENV_DEBUG,
    rollupOptions: { input: "index.mobile.html" },
  },
});
