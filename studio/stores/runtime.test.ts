import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it } from "vitest";
import { useStudioRuntimeStore } from "./runtime";

describe("Studio runtime store", () => {
  beforeEach(() => setActivePinia(createPinia()));

  it("keeps a valid selection and chooses the first route otherwise", () => {
    const store = useStudioRuntimeStore();
    store.replaceRoutes([
      { id: "a", label: "A", target: { baseUrl: "", apiKey: null } },
      { id: "b", label: "B", target: { baseUrl: "http://b", apiKey: null } },
    ]);
    expect(store.selectedHostId).toBe("a");
    store.selectedHostId = "b";
    store.replaceRoutes([
      { id: "b", label: "B", target: { baseUrl: "http://b", apiKey: null } },
    ]);
    expect(store.selectedRoute?.id).toBe("b");
  });
});
