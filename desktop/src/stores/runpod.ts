import { defineStore } from "pinia";
import { ipc } from "../lib/ipc";
import {
  emptyRunPodOverview,
  friendlyRunPodError,
  isRunPodHostUrl,
  rankRunPodGpus,
  type RunPodCreateInput,
  type RunPodNetworkVolumeCreateInput,
  type RunPodNetworkVolumeUpdateInput,
  type RunPodOverview,
} from "../lib/runpod";
import { useHostsStore } from "./hosts";

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
        this.operationError = friendlyRunPodError(
          error instanceof Error ? error.message : String(error),
        );
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
        this.operationError = friendlyRunPodError(
          error instanceof Error ? error.message : String(error),
        );
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
        this.operationError = friendlyRunPodError(
          error instanceof Error ? error.message : String(error),
        );
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
        this.operationError = friendlyRunPodError(
          error instanceof Error ? error.message : String(error),
        );
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
        else {
          const hosts = useHostsStore();
          const connectedHostIds = hosts.extras
            .filter((host) => isRunPodHostUrl(id, host.url))
            .map((host) => host.id);
          let savedHostIds: string[] = [];
          let cleanupFailed = false;
          try {
            savedHostIds = (await ipc.appSettingsGet()).savedHosts
              .filter((host) => isRunPodHostUrl(id, host.url))
              .map((host) => host.id);
          } catch {
            // The pod deletion may still proceed, but without the saved-host
            // inventory Studio cannot prove every alias was forgotten.
            cleanupFailed = true;
          }
          const connected = new Set(connectedHostIds);
          const hostIds = [...new Set([...connectedHostIds, ...savedHostIds])];
          await ipc.runpodDelete(id);
          // Serialize the settings writes: forget_remote_host mutates its
          // in-memory store under a lock, then saves after releasing the lock.
          // Concurrent aliases could otherwise persist out of order.
          for (const hostId of hostIds) {
            if (connected.has(hostId)) {
              try {
                await hosts.disconnect(hostId);
              } catch {
                // disconnect() retires the live host and its activity before
                // persisting. The stronger forget command below independently
                // removes saved/reconnect state, so it can recover that write.
              }
            }
            try {
              await ipc.forgetRemoteHost(hostId);
            } catch {
              cleanupFailed = true;
            }
          }
          if (cleanupFailed) {
            this.operationError =
              "The RunPod instance was deleted and its activity was cleared, but Studio couldn't fully forget its Mold host. It may reappear after restart; forget it from Machines if it does.";
          }
        }
        await this.load();
      } catch (error) {
        this.operationError = friendlyRunPodError(
          error instanceof Error ? error.message : String(error),
        );
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
