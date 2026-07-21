import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import tailwindcss from "@tailwindcss/vite";
import { devServerUrlPlugin } from "../studio/vite/serverUrl";

// The shared Mold Studio design system (tokens + primitives) lives outside
// this Vite root at ../ui and is consumed by every surface. `vue` is pinned
// to this app's copy so ui/ sources resolve it despite having no
// node_modules of their own.
const uiDir = fileURLToPath(new URL("../ui", import.meta.url));
const studioDir = fileURLToPath(new URL("../studio", import.meta.url));
const workspaceNodeModules = fileURLToPath(new URL("../node_modules", import.meta.url));

// Tauri expects a fixed dev port; 1430 avoids web/'s 5173 and other local apps.
export default defineConfig({
  plugins: [vue(), tailwindcss(), devServerUrlPlugin("Mold desktop")],
  clearScreen: false,
  resolve: {
    alias: {
      "@ui": uiDir,
      "@studio": studioDir,
      vue: fileURLToPath(new URL("../node_modules/vue", import.meta.url)),
    },
  },
  server: {
    port: 1430,
    strictPort: true,
    fs: {
      allow: [".", uiDir, studioDir, workspaceNodeModules],
    },
  },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  build: {
    // WKWebView on macOS 12+; keep output compatible with Safari 15.
    target: "safari15",
    minify: "esbuild",
    sourcemap: !!process.env.TAURI_ENV_DEBUG,
  },
});
