import { fileURLToPath, URL } from "node:url";
import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vitest/config";

export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  plugins: [vue()],
  resolve: {
    alias: {
      "@studio": fileURLToPath(new URL(".", import.meta.url)),
      "@ui": fileURLToPath(new URL("../ui", import.meta.url)),
    },
  },
  test: {
    environment: "happy-dom",
    include: ["**/*.test.ts"],
  },
});
