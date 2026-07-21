import { defineConfig, mergeConfig } from "vitest/config";
import viteConfig from "./vite.config";

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      environment: "happy-dom",
      globals: false,
      // The shared design-system primitives (../ui) are tested here so the
      // desktop CI gate covers them; they have no test runner of their own.
      include: ["src/**/*.test.ts", "../ui/**/*.test.ts"],
    },
  }),
);
