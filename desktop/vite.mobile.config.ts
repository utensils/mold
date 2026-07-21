import { existsSync, renameSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import tailwindcss from "@tailwindcss/vite";

const uiDir = fileURLToPath(new URL("../ui", import.meta.url));

export default defineConfig({
  plugins: [
    vue(),
    tailwindcss(),
    {
      name: "mold-mobile-index",
      // The desktop and mobile surfaces need distinct source entries, but
      // Tauri's embedded asset resolver always boots index.html at the root.
      writeBundle(options) {
        const dir = options.dir ?? "dist-mobile";
        const from = join(dir, "index.mobile.html");
        if (existsSync(from)) renameSync(from, join(dir, "index.html"));
      },
      configureServer(server) {
        server.middlewares.use((request, _response, next) => {
          if (request.url === "/" || request.url?.startsWith("/?")) {
            request.url = `/index.mobile.html${request.url.slice(1)}`;
          }
          next();
        });
      },
    },
  ],
  clearScreen: false,
  root: ".",
  resolve: {
    alias: {
      "@ui": uiDir,
      vue: fileURLToPath(new URL("./node_modules/vue", import.meta.url)),
    },
  },
  server: {
    host: "0.0.0.0",
    port: 1431,
    strictPort: true,
    fs: { allow: [".", uiDir] },
  },
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
