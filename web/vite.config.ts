import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import tailwindcss from "@tailwindcss/vite";

// In dev we proxy /api + /health to the running mold server so we can
// iterate on the UI against live data. Override the target with
// `MOLD_API_ORIGIN=http://hal9000:7680 bun run dev` when working against
// a remote GPU host.
//
// `process` is provided by Node/Bun at config-load time; we keep this
// untyped instead of pulling in `@types/node` for a single `env` lookup.
declare const process: { env: Record<string, string | undefined> };
const apiTarget = process.env.MOLD_API_ORIGIN ?? "http://localhost:7680";

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  server: {
    host: "0.0.0.0",
    port: 5174,
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true },
      "/health": { target: apiTarget, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2022",
    sourcemap: false,
  },
});
