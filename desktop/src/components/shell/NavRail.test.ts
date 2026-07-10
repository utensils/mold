import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import { createPinia } from "pinia";
import { createMemoryHistory, createRouter, type Router } from "vue-router";
import NavRail from "./NavRail.vue";

const stub = { template: "<div />" };

function makeRouter(): Router {
  return createRouter({
    history: createMemoryHistory(),
    routes: ["/generate", "/gallery", "/chains", "/models", "/history", "/settings"].map(
      (path) => ({
        path,
        component: stub,
      }),
    ),
  });
}

async function mountAt(path: string) {
  const router = makeRouter();
  router.push(path);
  await router.isReady();
  return mount(NavRail, {
    global: {
      plugins: [createPinia(), router],
      // DevelopCanvas paints to <canvas>, which happy-dom can't; stub it out
      // (it only renders inside job rows, of which there are none here anyway).
      stubs: { DevelopCanvas: stub },
    },
  });
}

describe("NavRail a11y", () => {
  it("labels the primary navigation landmark", async () => {
    const wrapper = await mountAt("/generate");
    expect(wrapper.get("nav").attributes("aria-label")).toBe("Primary");
  });

  it("marks the active route link with aria-current=page", async () => {
    const wrapper = await mountAt("/gallery");
    const links = wrapper.findAll("a");
    const gallery = links.find((a) => a.text().includes("Gallery"));
    const generate = links.find((a) => a.text().includes("Generate"));
    expect(gallery?.attributes("aria-current")).toBe("page");
    expect(generate?.attributes("aria-current")).toBeUndefined();
  });
});
