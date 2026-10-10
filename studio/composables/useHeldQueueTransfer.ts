import {
  computed,
  inject,
  provide,
  ref,
  type ComputedRef,
  type InjectionKey,
} from "vue";
import { queueTransferEligible } from "../lib/queueTransferEligibility";
import type { QueueTransferHost } from "../api/queueTransfer";

export function createHeldQueueTransfer(
  hosts: ComputedRef<QueueTransferHost[]>,
) {
  const busy = ref(false);
  const selection = ref<{ source: QueueTransferHost; jobId: string } | null>(
    null,
  );
  const canSend = (hostId: string) => {
    const source = hosts.value.find((host) => host.id === hostId && host.ready);
    return (
      !!source &&
      hosts.value.some(
        (host) =>
          host.ready &&
          host.generates !== false &&
          host.instanceId !== source.instanceId &&
          (!source.transferIdentity ||
            host.transferIdentity !== source.transferIdentity),
      )
    );
  };
  return {
    hosts,
    selection,
    busy,
    close() {
      if (!busy.value) selection.value = null;
    },
    canSend,
    canSendState(hostId: string, state: string) {
      const host = hosts.value.find((host) => host.id === hostId);
      return (
        canSend(hostId) &&
        queueTransferEligible(state, host?.preRenderTransfer === true)
      );
    },
    destinations: computed(() =>
      hosts.value.filter(
        (host) =>
          host.ready &&
          host.generates !== false &&
          host.instanceId !== selection.value?.source.instanceId &&
          (!selection.value?.source.transferIdentity ||
            host.transferIdentity !== selection.value.source.transferIdentity),
      ),
    ),
    open(hostId: string, jobId: string) {
      if (busy.value || !canSend(hostId)) return;
      const source = hosts.value.find((host) => host.id === hostId)!;
      selection.value = {
        source: { ...source, target: { ...source.target } },
        jobId,
      };
    },
  };
}
export type HeldQueueTransferController = ReturnType<
  typeof createHeldQueueTransfer
>;
const key: InjectionKey<HeldQueueTransferController> = Symbol(
  "held-queue-transfer",
);
export function provideHeldQueueTransfer(
  hosts: ComputedRef<QueueTransferHost[]>,
) {
  const controller = createHeldQueueTransfer(hosts);
  provide(key, controller);
  return controller;
}
export function useHeldQueueTransfer() {
  return inject(key, null);
}
