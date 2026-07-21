/**
 * The composer style preset is a look modifier composed into the OUTGOING
 * prompt at submit — the textarea (form.prompt) is never mutated.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { mount, flushPromises } from "@vue/test-utils";
import GenerateView from "./GenerateView.vue";
import { useGenerateFormStore } from "../stores/generateForm";
import { useGenerationStore } from "../stores/generation";
import { useModelStore } from "../stores/models";
import { useConnectionStore } from "../stores/connection";
import { useHostsStore } from "../stores/hosts";
import { useUiStore } from "../stores/ui";
import type { ModelEntry } from "../lib/api/types";

vi.mock("vue-router", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("../lib/api/client", () => ({
  apiJson: vi.fn(() => Promise.resolve([])),
  apiJsonTo: vi.fn(() => Promise.resolve([])),
  apiFetch: vi.fn(),
}));
vi.mock("../lib/ipc", () => ({ ipc: {} }));

const model: ModelEntry = {
  name: "flux-dev:q8",
  family: "flux",
  downloaded: true,
  default_width: 1024,
  default_height: 1024,
  default_steps: 20,
  default_guidance: 4.5,
} as ModelEntry;

describe("GenerateView style-at-submit", () => {
  beforeEach(() => {
    setActivePinia(createPinia());
    const conn = useConnectionStore();
    conn.info = { mode: "local", baseUrl: "http://127.0.0.1:7680", apiKey: "local-key" };
    conn.status = "ready";
    useHostsStore().initialized = true;
    useModelStore().all = [model];
  });
  afterEach(() => (document.body.innerHTML = ""));

  it("appends the style extra to the request prompt, leaving the textarea unchanged", async () => {
    const submit = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) });

    mount(GenerateView, { shallow: true, attachTo: document.body });
    await flushPromises();
    const form = useGenerateFormStore().form;
    form.prompt = "a lighthouse";
    form.model = "flux-dev:q8";
    form.family = "flux";
    form.batchSize = 1;
    form.stylePreset = "cinematic";

    useUiStore().generateTick++;
    await flushPromises();

    expect(submit).toHaveBeenCalledTimes(1);
    const [request] = submit.mock.calls[0]!;
    expect(request.prompt).toBe(
      "a lighthouse, cinematic lighting, film grain, shallow depth of field",
    );
    // The live composer text is untouched.
    expect(form.prompt).toBe("a lighthouse");
  });
});
