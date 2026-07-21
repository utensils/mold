import { describe, expect, it } from "vitest";
import { createMemoryHistory, createRouter } from "vue-router";
import { routes } from "./router";

function makeRouter() {
  return createRouter({ history: createMemoryHistory(), routes });
}

describe("router", () => {
  it("redirects the legacy /generate path to create", async () => {
    const router = makeRouter();
    await router.push("/generate");
    expect(router.currentRoute.value.name).toBe("create");
    expect(router.currentRoute.value.path).toBe("/create");
  });

  it("redirects the legacy /catalog path to models", async () => {
    const router = makeRouter();
    await router.push("/catalog");
    expect(router.currentRoute.value.name).toBe("models");
    expect(router.currentRoute.value.path).toBe("/models");
  });

  it("maps the Mold Studio destinations to workspace route names", async () => {
    const router = makeRouter();

    await router.push("/");
    expect(router.currentRoute.value.name).toBe("gallery");

    await router.push("/create");
    expect(router.currentRoute.value.name).toBe("create");

    await router.push("/models");
    expect(router.currentRoute.value.name).toBe("models");

    await router.push("/machines");
    expect(router.currentRoute.value.name).toBe("machines");

    await router.push("/machines/studio-01");
    expect(router.currentRoute.value.name).toBe("machine-detail");
    expect(router.currentRoute.value.params.id).toBe("studio-01");

    await router.push("/settings");
    expect(router.currentRoute.value.name).toBe("settings");
  });
});
