import { describe, expect, it } from "vitest";
import { templateText } from "./lexicon";

describe("templateText", () => {
  it("keeps only what a person reads", () => {
    const source = `<script setup lang="ts">const host = "host";</script>
<template>
  <p v-if="count > 0" data-test="installed-row" class="host" title="Get it">Ready to use {{ host }}</p>
  <img alt="a print" :title="hostName" />
</template>
<style scoped>.installed-grid { color: red; }</style>`;
    expect(templateText(source).trim()).toBe("Ready to use Get it a print");
  });
});
