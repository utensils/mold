import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { analyzeFrontend } from "../frontend-production-graph.mjs";

test("tests cannot keep a production module alive; Vue and dynamic aliases can", () => {
  const root = mkdtempSync(join(tmpdir(), "mold-production-graph-"));
  const write = (file, source) => {
    mkdirSync(dirname(join(root, file)), { recursive: true });
    writeFileSync(join(root, file), source);
  };
  try {
    write("web/src/main.ts", 'import "./App.vue";');
    write(
      "web/src/App.vue",
      '<script setup lang="ts">import { live } from "@studio/live"; const load = () => import("./Lazy.vue");</script><template>{{ live }}</template>',
    );
    write(
      "web/src/Lazy.vue",
      '<script setup lang="ts">import type { Shape } from "@studio/shape";</script>',
    );
    write("studio/live.ts", "export const live = true;");
    write("studio/shape.ts", "export interface Shape { value: string }");
    write("studio/orphan.ts", "export const obsolete = true;");
    write("studio/orphan.test.ts", 'import { obsolete } from "./orphan";');
    const result = analyzeFrontend(root, {
      entries: ["web/src/main.ts"],
      sourceDirs: ["web/src", "studio"],
      exceptions: {},
    });
    assert.deepEqual(result.unreachable, ["studio/orphan.ts"]);
    assert.deepEqual(result.unresolved, []);
    write("studio/live.ts", 'export { obsolete } from "./orphan";');
    assert.deepEqual(
      analyzeFrontend(root, {
        entries: ["web/src/main.ts"],
        sourceDirs: ["web/src", "studio"],
        exceptions: {},
      }).unreachable,
      [],
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("missing local imports and invalid public roots fail closed", () => {
  const root = mkdtempSync(join(tmpdir(), "mold-production-graph-"));
  try {
    mkdirSync(join(root, "studio"));
    writeFileSync(join(root, "studio/main.ts"), 'import "./missing";');
    const result = analyzeFrontend(root, {
      entries: ["studio/main.ts", "studio/absent.ts"],
      sourceDirs: ["studio"],
      exceptions: {},
    });
    assert.equal(result.unresolved.length, 2);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
