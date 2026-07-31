import { mount } from "@vue/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import type { RoutableHost } from "../../lib/hostRouting";
import type { PendingInstallChoice } from "../../composables/useModelInstallTargets";
import ModelInstallTargetDialog from "./ModelInstallTargetDialog.vue";

const pending = ref<PendingInstallChoice | null>(null);
const choose = vi.fn();
const cancel = vi.fn();

vi.mock("../../composables/useModelInstallTargets", () => ({
  useModelInstallTargets: () => ({ pending, choose, cancel }),
}));

function host(id: string, label: string): RoutableHost {
  return {
    id,
    label,
    url: `http://${id}:7680`,
    status: "ready",
    queueDepth: 0,
    gpu: null,
  };
}

const studio = host("studio", "Studio");
const origin = host("origin", "this server");

beforeEach(() => {
  vi.clearAllMocks();
  pending.value = null;
});

describe("ModelInstallTargetDialog", () => {
  it("renders nothing while no choice is pending", () => {
    const w = mount(ModelInstallTargetDialog);
    expect(w.find("[data-test=install-target-host]").exists()).toBe(false);
  });

  it("names every machine and what happens there", () => {
    pending.value = {
      modelId: "cv:1",
      displayName: "Alpha",
      targets: [
        { host: studio, action: "install" },
        { host: origin, action: "repair" },
      ],
    };
    const w = mount(ModelInstallTargetDialog);

    expect(w.find("[data-test=install-target-host]").classes()).toContain(
      "fixed",
    );
    expect(w.text()).toContain("Alpha");
    const options = w.findAll("[data-test=install-target-option]");
    expect(options).toHaveLength(2);
    expect(options[0].text()).toContain("Studio");
    expect(options[0].text()).toContain("Install");
    expect(options[1].text()).toContain("this server");
    expect(options[1].text()).toContain("Repair");
  });

  it("resolves the pending choice with the chosen machine", async () => {
    const targets: PendingInstallChoice["targets"] = [
      { host: studio, action: "install" },
      { host: origin, action: "repair" },
    ];
    pending.value = { modelId: "cv:1", displayName: "Alpha", targets };
    const w = mount(ModelInstallTargetDialog);

    await w.findAll("[data-test=install-target-option]")[0].trigger("click");
    expect(choose).toHaveBeenCalledWith(targets[0]);
  });

  it("cancels rather than dead-ending", async () => {
    pending.value = {
      modelId: "cv:1",
      displayName: "Alpha",
      targets: [
        { host: studio, action: "install" },
        { host: origin, action: "repair" },
      ],
    };
    const w = mount(ModelInstallTargetDialog);
    await w.find("[data-test=install-target-cancel]").trigger("click");
    expect(cancel).toHaveBeenCalled();
  });
});
