import { binding } from "@studio/api/secureCrypto";
import { deviceIdentity, saveSecureHost, secureHostForOrigin } from "@studio/api/secureStore";
import { ipc } from "./ipc";
import { podProxyUrl, type RunPodCreateInput } from "./runpod";

/** Provisioning grants the existing installation access; private device keys never enter IPC. */
export async function provisionSecureRunPod(input: RunPodCreateInput) {
  const identity = await deviceIdentity();
  try {
    const origin = input.networkVolumeId
      ? await ipc.runpodVolumeOrigin(input.networkVolumeId)
      : null;
    const resume = origin ? await secureHostForOrigin(origin) : null;
    const result = await ipc.runpodCreate(input, binding(identity), resume);
    try {
      await saveSecureHost(result.profile);
      await ipc.runpodRememberPairing(result.pod.id, binding(identity), result.profile);
    } catch {
      throw new Error(
        "The instance was created, but this app could not save its secure connection. Use in Mold to recover it; do not rent another instance.",
      );
    }
    return result.pod;
  } finally {
    identity.close();
    identity.free();
  }
}

export async function requireSecureRunPod(id: string): Promise<void> {
  const existing = await secureHostForOrigin(podProxyUrl(id));
  const identity = await deviceIdentity();
  try {
    const profile = existing ?? (await ipc.runpodPairing(id, binding(identity)));
    if (!profile)
      throw new Error(
        "Pair this instance using a fresh pairing link from its Mold host, then choose Use in Mold again.",
      );
    if (!existing) await saveSecureHost(profile);
    await ipc.runpodRememberPairing(id, binding(identity), profile);
  } finally {
    identity.close();
    identity.free();
  }
}
