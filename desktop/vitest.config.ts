import { fileURLToPath, URL } from "node:url";
import { configDefaults, defineConfig, mergeConfig } from "vitest/config";
import viteConfig from "./vite.config";

export default mergeConfig(
  viteConfig,
  defineConfig({
    resolve: {
      alias: {
        // ../ui test files resolve test-only deps from this app's copies.
        "@vue/test-utils": fileURLToPath(
          new URL("../node_modules/@vue/test-utils", import.meta.url),
        ),
      },
    },
    test: {
      environment: "happy-dom",
      globals: false,
      // The shared design-system primitives (../ui) are tested here so the
      // desktop CI gate covers them; they have no test runner of their own.
      include: ["src/**/*.test.ts", "../ui/**/*.test.ts", "../studio/**/*.test.ts"],
      // The ../ui include widens the glob base to the repo root, which holds
      // .direnv flake-input symlinks to stale full-source snapshots — never
      // collect tests through them.
      exclude: [...configDefaults.exclude, "**/.direnv/**", "**/dist*/**"],
    },
  }),
);
