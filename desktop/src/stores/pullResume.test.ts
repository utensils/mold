import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { usePullResumeStore } from "./pullResume";
import { useDownloadsStore } from "./downloads";
import { useGenerationStore } from "./generation";
import { useToastStore } from "./toasts";
import type { DownloadJob, GenerateRequest } from "../lib/api/types";

const job = (id: string, model: string, status: DownloadJob["status"]): DownloadJob => ({
  id,
  model,
  status,
  files_done: 1,
  files_total: 1,
  bytes_done: 1,
  bytes_total: 1,
});

const request: GenerateRequest = {
  prompt: "a deer",
  model: "jibmix-flux:fp8",
  width: 1024,
  height: 1024,
  steps: 25,
  guidance: 3.5,
  batch_size: 1,
};

describe("pullResume store", () => {
  beforeEach(() => setActivePinia(createPinia()));

  function armOnPrimary() {
    const store = usePullResumeStore();
    store.arm({
      model: "jibmix-flux:fp8",
      hostId: null,
      hostLabel: "plato",
      request,
      batch: 1,
      route: null,
    });
    return store;
  }

  it("resubmits the generation once the model's pull completes", () => {
    const store = armOnPrimary();
    const generation = useGenerationStore();
    const submit = vi
      .spyOn(generation, "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) } as never);

    useDownloadsStore().history = [job("d1", "jibmix-flux:fp8", "completed")];
    store.check();

    expect(submit).toHaveBeenCalledWith(request, 1, null);
    expect(store.pending).toBeNull();
  });

  it("ignores terminal jobs that predate the arm (a stale history entry must not resume)", () => {
    const downloads = useDownloadsStore();
    downloads.history = [job("old", "jibmix-flux:fp8", "completed")];
    const store = armOnPrimary();
    const submit = vi.spyOn(useGenerationStore(), "submitBatch");

    store.check();
    expect(submit).not.toHaveBeenCalled();
    expect(store.pending).not.toBeNull();

    // A NEW completion after arming resumes.
    downloads.history = [...downloads.history, job("new", "jibmix-flux:fp8", "completed")];
    vi.spyOn(useGenerationStore(), "submitBatch").mockReturnValue({
      jobs: [],
      settled: Promise.resolve([]),
    } as never);
    store.check();
    expect(store.pending).toBeNull();
  });

  it("gives up with a toast when the pull fails or is cancelled", () => {
    const store = armOnPrimary();
    const submit = vi.spyOn(useGenerationStore(), "submitBatch");

    useDownloadsStore().history = [job("d1", "jibmix-flux:fp8", "failed")];
    store.check();

    expect(submit).not.toHaveBeenCalled();
    expect(store.pending).toBeNull();
    expect(useToastStore().items.some((t) => t.kind === "error")).toBe(true);
  });

  it("watches the routed host's bucket when the pull targets a remote host", () => {
    const store = usePullResumeStore();
    store.arm({
      model: "jibmix-flux:fp8",
      hostId: "plato-7680",
      hostLabel: "plato",
      request,
      batch: 2,
      route: null,
    });
    const downloads = useDownloadsStore();
    downloads.hostStates["plato-7680"] = {
      label: "plato",
      target: { baseUrl: "http://plato:7680", apiKey: null },
      subscribed: true,
      abort: null,
      cancelling: [],
      ready: null,
      activeJobs: [],
      queued: [],
      history: [job("d1", "jibmix-flux:fp8", "completed")],
    };
    const submit = vi
      .spyOn(useGenerationStore(), "submitBatch")
      .mockReturnValue({ jobs: [], settled: Promise.resolve([]) } as never);

    store.check();
    expect(submit).toHaveBeenCalledWith(request, 2, null);
  });

  it("cancel() clears the pending resume without submitting", () => {
    const store = armOnPrimary();
    store.cancel();
    useDownloadsStore().history = [job("d1", "jibmix-flux:fp8", "completed")];
    const submit = vi.spyOn(useGenerationStore(), "submitBatch");
    store.check();
    expect(submit).not.toHaveBeenCalled();
  });
});
