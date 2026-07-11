import { defineStore } from "pinia";
import { ipc } from "../lib/ipc";
import {
  emptyRunPodOverview,
  rankRunPodGpus,
  type RunPodCreateInput,
  type RunPodNetworkVolumeCreateInput,
  type RunPodNetworkVolumeUpdateInput,
  type RunPodOverview,
} from "../lib/runpod";

export const useRunPodStore = defineStore("runpod", {
  state: () => ({
    overview: emptyRunPodOverview() as RunPodOverview,
    loading: false,
    mutating: null as string | null,
    error: null as string | null,
    operationError: null as string | null,
    loaded: false,
  }),
  getters: {
    gpus: (state) => rankRunPodGpus(state.overview.gpus),
    runningPods: (state) =>
      state.overview.pods.filter((pod) => pod.desiredStatus.toUpperCase() === "RUNNING"),
  },
  actions: {
    async load() {
      if (this.loading) return;
      this.loading = true;
      this.error = null;
      try {
        this.overview = await ipc.runpodOverview();
        this.loaded = true;
      } catch (error) {
        this.error = error instanceof Error ? error.message : String(error);
      } finally {
        this.loaded = true;
        this.loading = false;
      }
    },
    async saveApiKey(key: string) {
      const trimmed = key.trim();
      if (!trimmed) throw new Error("Enter a RunPod API key.");
      await ipc.secretSet("runpod-api-key", trimmed);
      await this.load();
      if (!this.overview.configured || this.error) {
        await ipc.secretClear("runpod-api-key");
        throw new Error(this.error ?? "RunPod rejected this API key.");
      }
    },
    async disconnect() {
      await ipc.secretClear("runpod-api-key");
      this.overview = emptyRunPodOverview();
      this.error = null;
      this.loaded = true;
    },
    async create(input: RunPodCreateInput) {
      this.mutating = "create";
      this.operationError = null;
      try {
        await ipc.runpodCreate(input);
        await this.load();
      } catch (error) {
        this.operationError = error instanceof Error ? error.message : String(error);
        throw error;
      } finally {
        this.mutating = null;
      }
    },
    async createNetworkVolume(input: RunPodNetworkVolumeCreateInput) {
      this.mutating = "volume:create";
      this.operationError = null;
      try {
        const volume = await ipc.runpodNetworkVolumeCreate(input);
        await this.load();
        return volume;
      } catch (error) {
        this.operationError = error instanceof Error ? error.message : String(error);
        throw error;
      } finally {
        this.mutating = null;
      }
    },
    async updateNetworkVolume(input: RunPodNetworkVolumeUpdateInput) {
      this.mutating = `volume:update:${input.id}`;
      this.operationError = null;
      try {
        const volume = await ipc.runpodNetworkVolumeUpdate(input);
        await this.load();
        return volume;
      } catch (error) {
        this.operationError = error instanceof Error ? error.message : String(error);
        throw error;
      } finally {
        this.mutating = null;
      }
    },
    async deleteNetworkVolume(id: string) {
      this.mutating = `volume:delete:${id}`;
      this.operationError = null;
      try {
        await ipc.runpodNetworkVolumeDelete(id);
        await this.load();
      } catch (error) {
        this.operationError = error instanceof Error ? error.message : String(error);
        throw error;
      } finally {
        this.mutating = null;
      }
    },
    async act(action: "start" | "stop" | "delete", id: string) {
      this.mutating = `${action}:${id}`;
      this.operationError = null;
      try {
        if (action === "start") await ipc.runpodStart(id);
        else if (action === "stop") await ipc.runpodStop(id);
        else await ipc.runpodDelete(id);
        await this.load();
      } catch (error) {
        this.operationError = error instanceof Error ? error.message : String(error);
        throw error;
      } finally {
        this.mutating = null;
      }
    },
    clearOperationError() {
      this.operationError = null;
    },
  },
});
