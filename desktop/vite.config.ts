import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import tailwindcss from "@tailwindcss/vite";

// Tauri expects a fixed dev port; 1430 avoids web/'s 5173 and other local apps.
export default defineConfig({
  plugins: [vue(), tailwindcss()],
  clearScreen: false,
  server: {
    port: 1430,
    strictPort: true,
  },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  build: {
    // WKWebView on macOS 12+; keep output compatible with Safari 15.
    target: "safari15",
    minify: "esbuild",
    sourcemap: !!process.env.TAURI_ENV_DEBUG,
  },
});
