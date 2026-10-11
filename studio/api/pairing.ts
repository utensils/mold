import {
  parseConnectionEndpoints,
  type ConnectionEndpoint,
} from "./connectionRoutes";
import type { ApiTarget } from "./client";
import { apiFetchTo, apiJsonTo } from "./client";
import type {
  SecureGrant,
  SecureHostIdentity,
  SecureInvite,
} from "./secureCrypto";
import { claimSecurePairing } from "./securePairing";

export interface PairingSession extends SecureHostIdentity {
  endpoints?: ConnectionEndpoint[];
  protocol: 3;
  invite_seed: string;
  invite: SecureInvite;
  auth_required: boolean;
  instance_id: string;
  hostname: string | null;
}

export interface PairingClaim {
  endpoints?: ConnectionEndpoint[];
  api_key: string | null;
  instance_id: string;
  hostname: string | null;
  grant: SecureGrant;
}

export interface PairingClientIdentity {
  name: string;
  kind: "iphone" | "ipad" | "android" | "mobile" | "browser" | "desktop";
}

export interface PairedClient {
  id: string;
  name: string;
  client_kind: string;
  created_at_ms: number;
  last_used_at_ms: number | null;
}

export interface PairedClientsResponse {
  auth_required: boolean;
  pairing_available: boolean;
  clients: PairedClient[];
}

export interface MobilePairingPayload extends SecureHostIdentity {
  endpoints?: ConnectionEndpoint[];
  type: "mold.mobile-pairing";
  version: 2;
  base_url: string;
  invite_seed: string;
  invite: SecureInvite;
  instance_id: string;
  name: string;
}

/**
 * Where a pairing code points: a universal link the Mold Studio Companion
 * claims (utensils.io/.well-known/apple-app-site-association), and on a phone
 * without it, a page that says what to install. The payload rides in the
 * fragment, which a browser never sends to utensils.io.
 */
export const MOBILE_PAIRING_LINK = "https://utensils.io/mold/pair";

export function mobilePairingUrl(payload: MobilePairingPayload): string {
  const fields = new URLSearchParams();
  fields.set("version", String(payload.version));
  fields.set("base_url", payload.base_url);
  fields.set("instance_id", payload.instance_id);
  fields.set("name", payload.name);
  fields.set("namespace", payload.namespace);
  fields.set("host_key", payload.host_key);
  fields.set("host_noise_key", payload.host_noise_key);
  fields.set("host_binding_signature", payload.host_binding_signature);
  if (payload.websocket_url) fields.set("websocket_url", payload.websocket_url);
  fields.set("invite_seed", payload.invite_seed);
  fields.set("invite", JSON.stringify(payload.invite));
  if (payload.endpoints?.length)
    fields.set(
      "endpoints",
      JSON.stringify(
        parseConnectionEndpoints(payload.endpoints).map(({ kind, url }) => ({
          kind,
          url,
        })),
      ),
    );
  return `${MOBILE_PAIRING_LINK}#${fields.toString()}`;
}

/**
 * The form-encoded fields of a pairing link: the fragment of
 * `https://utensils.io/mold/pair#…`. Private invitation material must stay in
 * the fragment, which is never transmitted to the website server.
 */
function pairingLinkFields(url: URL): URLSearchParams | null {
  if (url.username || url.password) return null;
  const link = new URL(MOBILE_PAIRING_LINK);
  if (
    url.protocol === link.protocol &&
    url.hostname === link.hostname &&
    !url.port &&
    url.pathname.replace(/\/$/, "") === link.pathname &&
    !url.search
  ) {
    return new URLSearchParams(url.hash.slice(1));
  }
  return null;
}

export function createPairingSession(
  target: ApiTarget,
): Promise<PairingSession> {
  return apiJsonTo<PairingSession>(target, "/api/pairing/sessions", {
    method: "POST",
  });
}

export function claimPairingSession(
  payload: MobilePairingPayload,
  client: PairingClientIdentity = { name: "Mold mobile", kind: "mobile" },
): Promise<PairingClaim> {
  return claimSecurePairing(payload, client);
}

export function listPairedClients(
  target: ApiTarget,
): Promise<PairedClientsResponse> {
  return apiJsonTo<unknown>(target, "/api/pairing/clients").then((value) => {
    if (
      !value ||
      typeof value !== "object" ||
      typeof (value as PairedClientsResponse).auth_required !== "boolean" ||
      typeof (value as PairedClientsResponse).pairing_available !== "boolean" ||
      !Array.isArray((value as PairedClientsResponse).clients)
    ) {
      throw new Error(
        "This Mold host does not support paired access management yet.",
      );
    }
    return value as PairedClientsResponse;
  });
}

export async function revokePairedClient(
  target: ApiTarget,
  id: string,
): Promise<void> {
  await apiFetchTo(target, `/api/pairing/clients/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export function parseMobilePairingPayload(raw: string): MobilePairingPayload {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    try {
      const fields = pairingLinkFields(new URL(raw.trim()));
      if (!fields) throw new Error();
      value = {
        type: "mold.mobile-pairing",
        version: Number(fields.get("version")),
        base_url: fields.get("base_url"),
        namespace: fields.get("namespace"),
        host_key: fields.get("host_key"),
        host_noise_key: fields.get("host_noise_key"),
        host_binding_signature: fields.get("host_binding_signature"),
        websocket_url: fields.get("websocket_url"),
        invite_seed: fields.get("invite_seed"),
        invite: fields.has("invite") ? JSON.parse(fields.get("invite")!) : null,
        instance_id: fields.get("instance_id"),
        name: fields.get("name"),
        ...(fields.has("endpoints")
          ? { endpoints: JSON.parse(fields.get("endpoints")!) }
          : {}),
      };
    } catch {
      throw new Error("That QR code is not a Mold pairing code.");
    }
  }
  if (!value || typeof value !== "object")
    throw new Error("That QR code is not a Mold pairing code.");
  const payload = value as Partial<MobilePairingPayload>;
  if (
    payload.type !== "mold.mobile-pairing" ||
    payload.version !== 2 ||
    typeof payload.base_url !== "string" ||
    !/^https?:\/\//i.test(payload.base_url) ||
    !/^[0-9a-f]{32}$/.test(payload.namespace ?? "") ||
    !/^[0-9a-f]{64}$/.test(payload.host_key ?? "") ||
    !/^[0-9a-f]{64}$/.test(payload.host_noise_key ?? "") ||
    !/^[0-9a-f]{128}$/.test(payload.host_binding_signature ?? "") ||
    !/^[0-9a-f]{64}$/.test(payload.invite_seed ?? "") ||
    (payload.websocket_url !== null &&
      typeof payload.websocket_url !== "string") ||
    !payload.invite ||
    !/^[0-9a-f]{32}$/.test(payload.invite.id) ||
    !/^[0-9a-f]{64}$/.test(payload.invite.signingKey) ||
    !/^[0-9a-f]{128}$/.test(payload.invite.signature) ||
    !Number.isSafeInteger(payload.invite.expiresAt) ||
    payload.invite.expiresAt <= 0 ||
    !Number.isSafeInteger(payload.invite.epoch) ||
    payload.invite.epoch < 0 ||
    typeof payload.instance_id !== "string" ||
    typeof payload.name !== "string"
  ) {
    throw new Error("That QR code is not a supported Mold pairing code.");
  }
  if (payload.endpoints !== undefined)
    payload.endpoints = parseConnectionEndpoints(payload.endpoints);
  return payload as MobilePairingPayload;
}
