import { invoke, isTauri } from "@tauri-apps/api/core";
import {
  configureSecurePlatform,
  listSecureHosts,
  secureHostForOrigin,
  deviceIdentity,
} from "@studio/api/secureStore";
import { configureSecureMediaAdapter } from "@studio/api/secureMedia";
import { secureWebsocketFor } from "@studio/api/relayTransport";
import { hex, type Identity, type SecureHostProfile } from "@studio/api/secureCrypto";

function nativeHost(profile: SecureHostProfile, identity: Identity, origin: string) {
  const bytes = identity.toBytes();
  try {
    const websocket = secureWebsocketFor(profile, origin) ?? profile.websocket_url;
    if (!websocket) throw new Error("This machine has no secure route.");
    return {
      protocol: 3,
      origin,
      websocket,
      namespace: profile.namespace,
      host_key: profile.host_key,
      host_noise_key: profile.host_noise_key,
      host_binding_signature: profile.host_binding_signature,
      device_identity: hex(bytes),
      api_key: profile.apiKey,
      grant: profile.grant,
    };
  } finally {
    bytes.fill(0);
  }
}
/** IPC receives private state only inside this trusted packaged WebView process. */
export async function installSecureNative(): Promise<void> {
  if (!isTauri()) return;
  configureSecurePlatform(async (profile, identity) => {
    for (const origin of profile.origins)
      await invoke("secure_host_register", { host: nativeHost(profile, identity, origin) });
  });
  const media = new Map<string, string>();
  configureSecureMediaAdapter({
    async open(url) {
      const profile = await secureHostForOrigin(url.origin);
      if (!profile) throw new Error("Pair this machine securely to play its media.");
      const identity = await deviceIdentity();
      try {
        const result = await invoke<{ id: string; url: string }>("secure_media_open", {
          host: nativeHost(profile, identity, url.origin),
          path: url.pathname + url.search,
        });
        media.set(result.url, result.id);
        return result.url;
      } finally {
        identity.close();
        identity.free();
      }
    },
    close(url) {
      const id = media.get(url);
      if (id) {
        media.delete(url);
        void invoke("secure_media_close", { id });
      }
    },
  });
  window.addEventListener("mold:secure-host-forgotten", (event) => {
    for (const origin of (event as CustomEvent<{ origins: string[] }>).detail.origins)
      void invoke("secure_host_forget", { origin });
  });
  for (const profile of await listSecureHosts()) await secureHostForOrigin(profile.origins[0]!);
}
