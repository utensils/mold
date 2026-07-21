// Using vitest/config's defineConfig keeps the `test` block strictly typed;
// the plain vite import doesn't know about it.
import { defineConfig } from "vitest/config";
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

// Shared Mold Studio design system (tokens + primitives) outside this root;
// `vue` is pinned to this app's copy so ui/ sources resolve it.
const uiDir = new URL("../ui", import.meta.url).pathname;

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  resolve: {
    alias: {
      "@ui": uiDir,
      vue: new URL("./node_modules/vue", import.meta.url).pathname,
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5174,
    fs: { allow: [".", uiDir] },
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
  test: {
    environment: "happy-dom",
    setupFiles: ["src/__tests__/setup.ts"],
    include: ["src/**/*.{test,spec}.ts"],
    coverage: {
      reporter: ["text", "html"],
      include: ["src/**/*.ts"],
      exclude: ["src/**/*.d.ts", "src/**/*.test.ts", "src/main.ts"],
    },
  },
});
