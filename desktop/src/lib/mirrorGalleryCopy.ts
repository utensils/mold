import { watch } from "vue";
import {
  copyGalleryPrint,
  copyMachineIdentity,
  type CopyOptions,
  type CopyMachine,
} from "@studio/api/galleryCopy";
import { copyJournal, type SyncDocumentStorage } from "@studio/lib/galleryCopyJournal";
import { mirrorMetadataMatches } from "@studio/lib/galleryMirrorIdentity";
import { json } from "@studio/api/galleryCopyWire";
import type { ConnectionMode } from "@studio/api/connectionMode";
import type { ApiTarget } from "@studio/api/client";

export interface MirrorSourceContext {
  id: string;
  instanceId?: string | null;
  current?(): boolean;
  onWarning?(message: string): void;
  /** Untouched recipe tokens when the caller already captured an exact offer. */
  rawMetadata?: string;
}
interface CurrentSource {
  instanceId?: string | null | undefined;
  baseUrl?: string | null | undefined;
  apiKey?: string | null | undefined;
  connectionMode?: ConnectionMode | undefined;
}
export function captureMirrorSource(
  id: string,
  instanceId: string | null | undefined,
  target: ApiTarget,
  resolve: () => CurrentSource | undefined,
  onWarning?: (message: string) => void,
): MirrorSourceContext {
  const captured = { ...target };
  return {
    id,
    ...(onWarning ? { onWarning } : {}),
    instanceId: instanceId ?? null,
    current() {
      const live = resolve();
      return (
        !!live &&
        (live.instanceId ?? null) === (instanceId ?? null) &&
        live.baseUrl === captured.baseUrl &&
        (live.apiKey ?? null) === (captured.apiKey ?? null) &&
        (live.connectionMode ?? "secure") === (captured.connectionMode ?? "secure")
      );
    },
  };
}
interface Dependencies {
  ensureLocal(): Promise<ApiTarget>;
  nativeHooks(): Pick<
    CopyOptions,
    "nativeImportOutput" | "nativeTransferSources" | "nativeMatchesOutput" | "nativeLegacyOffer"
  >;
  storage: SyncDocumentStorage;
}
/** One copy contract for explicit local saves, automatic caches and Library Sync. */
export async function mirrorGalleryCopy(
  source: ApiTarget,
  filename: string,
  metadata: object | null | undefined,
  deps: Dependencies,
  context?: MirrorSourceContext,
): Promise<string> {
  const sourceTarget = { ...source };
  const owner = context ? { ...context } : undefined;
  const expected = metadata == null ? undefined : JSON.stringify(metadata);
  const abort = new AbortController();
  const current = () =>
    owner?.current?.() !== false &&
    source.baseUrl === sourceTarget.baseUrl &&
    source.apiKey === sourceTarget.apiKey &&
    source.connectionMode === sourceTarget.connectionMode;
  const check = () => {
    abort.signal.throwIfAborted();
    if (!current()) throw new Error("The source machine or connection changed. Select Save again.");
  };
  check();
  const stop = watch(
    current,
    (valid) => {
      if (!valid)
        abort.abort(new Error("The source machine or connection changed. Select Save again."));
    },
    { flush: "sync" },
  );
  try {
    const sourceMachine: CopyMachine = {
      id: owner?.id ?? `external:${sourceTarget.baseUrl}`,
      label: "Source machine",
      instanceId: owner?.instanceId ?? "pending",
      target: sourceTarget,
    };
    copyMachineIdentity(sourceMachine); // Reject secret-bearing routes before any journal write.
    const local = { ...(await deps.ensureLocal()) };
    check();
    let destination: CopyMachine | undefined;
    for (const machine of [
      sourceMachine,
      { id: "local", label: "This device", instanceId: "pending", target: local },
    ]) {
      const status = await json(machine.target, "/api/status", abort.signal);
      check();
      if (typeof status.value.instance_id !== "string" || !status.value.instance_id)
        throw new Error("Reconnect the machine before saving this print.");
      if (
        machine === sourceMachine &&
        owner?.instanceId &&
        owner.instanceId !== status.value.instance_id
      )
        throw new Error("The source machine installation changed. Reconnect before saving.");
      machine.instanceId = status.value.instance_id;
      if (machine !== sourceMachine) destination = machine;
    }
    const receipt = await copyGalleryPrint({
      ...deps.nativeHooks(),
      source: sourceMachine,
      destination: destination!,
      filename,
      journal: copyJournal(deps.storage),
      signal: abort.signal,
      current: () => {
        check();
        return true;
      },
      verifySourceOffer(offer) {
        check();
        // A legacy JS listing seed is a display hint, never byte authority. Compare
        // its parsed projection as before; all copied recipe tokens remain raw.
        if (
          owner?.rawMetadata
            ? !mirrorMetadataMatches(owner.rawMetadata, offer.rawMetadata)
            : expected !== undefined &&
              !mirrorMetadataMatches(expected, JSON.stringify(offer.metadata))
        )
          throw new Error("The print's settings changed. Refresh the Library before saving it.");
      },
    });
    check();
    if (receipt.stage !== "complete" || !receipt.destinationFilename)
      throw new Error("Saving this print has not finished. Sync again to repair it.");
    if (receipt.outcome === "legacy_inputs_unavailable")
      owner?.onWarning?.(
        "The print was copied, but this older print did not save its original inputs.",
      );
    return receipt.destinationFilename;
  } finally {
    stop();
  }
}
