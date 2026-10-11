import { SecureChannel } from "./secureChannel";
import {
  checkGrant,
  checkInvite,
  claimBytes,
  hex,
  loadSecureCrypto,
  type Identity,
} from "./secureCrypto";
import { deviceIdentity, saveSecureHost } from "./secureStore";
import { parseConnectionEndpoints } from "./connectionRoutes";
import type {
  MobilePairingPayload,
  PairingClaim,
  PairingClientIdentity,
} from "./pairing";

export async function claimSecurePairing(
  payload: MobilePairingPayload,
  client: PairingClientIdentity,
): Promise<PairingClaim> {
  await loadSecureCrypto();
  const invitation = checkInvite(payload, payload.invite, payload.invite_seed);
  let identity: Identity | undefined;
  let channel: SecureChannel | undefined;
  try {
    identity = await deviceIdentity();
    // Try pinned encrypted LAN routes even from HTTPS: browser local-network
    // permissions decide availability. A refusal falls through to the cloud.
    const candidates = parseConnectionEndpoints(payload.endpoints ?? []);
    if (
      !candidates.some((endpoint) => endpoint.url === payload.base_url) &&
      !payload.websocket_url
    )
      candidates.push({ url: new URL(payload.base_url).origin, kind: "lan" });
    const direct = candidates.filter((endpoint) => endpoint.kind !== "relay");
    let failure: unknown;
    for (const endpoint of direct.slice(0, 2)) {
      const url = new URL("/_mold/secure", endpoint.url);
      url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
      try {
        channel = await SecureChannel.connect({
          host: payload,
          identity,
          invite: { certificate: payload.invite, identity: invitation },
          websocket: url.href,
          // Pairing is an explicit user action and may need a browser local-
          // network permission response. Ordinary health probes stay at 2.5 s.
          signal: AbortSignal.timeout(15000),
        });
        break;
      } catch (error) {
        failure = error;
      }
    }
    if (!channel && payload.websocket_url)
      channel = await SecureChannel.connect({
        host: payload,
        identity,
        invite: { certificate: payload.invite, identity: invitation },
      });
    if (!channel)
      throw (
        failure ?? new Error("No secure address is available for this host.")
      );
    const response = await channel.fetch(
      new Request(new URL("/api/pairing/claim", payload.base_url), {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          client_name: client.name,
          client_kind: client.kind,
          proof: hex(
            invitation.sign(
              claimBytes(
                payload,
                payload.invite,
                channel.handshakeHash(),
                identity,
              ),
            ),
          ),
        }),
      }),
    );
    const text = await boundedClaim(response);
    if (!response.ok)
      throw new Error(
        "This pairing code could not be redeemed. Create a fresh code on the host.",
      );
    const claim = JSON.parse(text) as PairingClaim;
    if (
      claim.instance_id !== payload.instance_id ||
      !claim.api_key?.startsWith("mold_pair_")
    )
      throw new Error("The host returned an invalid secure pairing.");
    checkGrant(payload, identity, claim.grant);
    const endpoints = parseConnectionEndpoints(
      claim.endpoints ?? payload.endpoints ?? [],
    );
    const baseOrigin = new URL(payload.base_url).origin;
    if (
      !endpoints.some((endpoint) => endpoint.url === baseOrigin) &&
      endpoints.length < 8
    )
      endpoints.push({
        url: baseOrigin,
        kind:
          payload.websocket_url && baseOrigin.startsWith("https:")
            ? "relay"
            : "lan",
      });
    const origins = [
      ...new Set([
        new URL(payload.base_url).origin,
        ...endpoints.map((endpoint) => endpoint.url),
      ]),
    ].slice(0, 8);
    await saveSecureHost({
      namespace: payload.namespace,
      host_key: payload.host_key,
      host_noise_key: payload.host_noise_key,
      host_binding_signature: payload.host_binding_signature,
      websocket_url: payload.websocket_url,
      grant: claim.grant,
      instance_id: claim.instance_id,
      endpoints,
      origins,
      apiKey: claim.api_key,
    });
    // The transport owns the bearer. General host registries must not copy it
    // into localStorage, QR codes, URLs, or UI state.
    return { ...claim, api_key: null };
  } finally {
    channel?.close();
    identity?.close();
    identity?.free();
    invitation.close();
    invitation.free();
  }
}

async function boundedClaim(response: Response): Promise<string> {
  if (!response.body) throw new Error("Missing pairing response.");
  const reader = response.body.getReader();
  const output = new Uint8Array(65536);
  let size = 0;
  try {
    for (;;) {
      const part = await reader.read();
      if (part.done) break;
      if (size + part.value.length > output.length)
        throw new Error("Invalid pairing response size.");
      output.set(part.value, size);
      size += part.value.length;
    }
    return new TextDecoder().decode(output.subarray(0, size));
  } finally {
    output.fill(0);
    await reader.cancel();
  }
}
