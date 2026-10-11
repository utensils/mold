import { secureHostForOrigin } from "./secureStore";
import {
  registerRegularConnection,
  forgetRegularConnection,
} from "./directConnection";
import { directOrigin, rememberConnectionMode } from "./connectionMode";
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

export interface SecurePairingSession extends SecureHostIdentity {
  mode?: "secure";
  endpoints?: ConnectionEndpoint[];
  protocol: 3;
  invite_seed: string;
  invite: SecureInvite;
  auth_required: boolean;
  instance_id: string;
  hostname: string | null;
}

export interface SecurePairingClaim {
  mode?: "secure";
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

export interface SecureMobilePairingPayload extends SecureHostIdentity {
  mode?: "secure";
  endpoints?: ConnectionEndpoint[];
  type: "mold.mobile-pairing";
  version: 2;
  base_url: string;
  invite_seed: string;
  invite: SecureInvite;
  instance_id: string;
  name: string;
}

export interface RegularMobilePairingPayload {
  type: "mold.mobile-pairing";
  version: 2;
  mode: "regular";
  base_url: string;
  instance_id: string;
  name: string;
  pairing_token: string;
  expires_at: number;
}
export interface RegularPairingSession {
  mode: "regular";
  base_url: string;
  pairing_token: string;
  expires_at: number;
  instance_id: string;
  hostname: string | null;
  auth_required: boolean;
}
export interface RegularPairingClaim {
  mode: "regular";
  api_key: string;
  instance_id: string;
  hostname: string | null;
  base_url: string;
}
export type MobilePairingPayload =
  SecureMobilePairingPayload | RegularMobilePairingPayload;
export type PairingSession = SecurePairingSession | RegularPairingSession;
export type PairingClaim = SecurePairingClaim | RegularPairingClaim;
export function pairingExpiresAt(
  payload: MobilePairingPayload | PairingSession,
): number {
  return payload.mode === "regular"
    ? payload.expires_at
    : payload.invite.expiresAt;
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
  if (payload.mode === "regular") {
    fields.set("mode", "regular");
    fields.set("pairing_token", payload.pairing_token);
    fields.set("expires_at", String(payload.expires_at));
    return `${MOBILE_PAIRING_LINK}#${fields.toString()}`;
  }
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
  options?: { mode: "regular"; direct_url: string },
): Promise<PairingSession> {
  return apiJsonTo<PairingSession>(target, "/api/pairing/sessions", {
    method: "POST",
    ...(options
      ? {
          headers: { "content-type": "application/json" },
          body: JSON.stringify(options),
        }
      : {}),
  });
}

export async function claimPairingSession(
  payload: MobilePairingPayload,
  client: PairingClientIdentity = { name: "Mold mobile", kind: "mobile" },
): Promise<PairingClaim> {
  if (payload.mode !== "regular") {
    const claim = await claimSecurePairing(payload, client);
    const profile = await secureHostForOrigin(
      new URL(payload.base_url).origin,
      false,
    );
    for (const origin of profile?.origins ?? [payload.base_url]) {
      await forgetRegularConnection(origin);
      rememberConnectionMode(origin, "secure");
    }
    return claim;
  }
  const origin = directOrigin(payload.base_url);
  if (payload.expires_at <= Date.now() / 1000)
    throw new Error("That pairing code expired. Create a new one on the host.");
  const claim = await apiJsonTo<RegularPairingClaim>(
    { baseUrl: origin, apiKey: null, connectionMode: "regular" },
    "/api/pairing/regular/claim",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        token: payload.pairing_token,
        client_name: client.name,
        client_kind: client.kind,
      }),
    },
  );
  if (
    claim.mode !== "regular" ||
    claim.instance_id !== payload.instance_id ||
    claim.base_url !== origin ||
    typeof claim.api_key !== "string" ||
    !claim.api_key.startsWith("mold_pair_")
  )
    throw new Error("The host returned an invalid Regular pairing.");
  await registerRegularConnection(origin, claim.api_key);
  return claim;
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

export function parseMobilePairingPayload(input: string): MobilePairingPayload {
  let value: unknown;
  try {
    value = JSON.parse(input);
  } catch {
    try {
      const fields = pairingLinkFields(new URL(input.trim()));
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
        ...(fields.has("mode") ? { mode: fields.get("mode") } : {}),
        ...(fields.get("mode") === "regular"
          ? {
              pairing_token: fields.get("pairing_token"),
              expires_at: Number(fields.get("expires_at")),
            }
          : {}),
        ...(fields.has("endpoints")
          ? { endpoints: JSON.parse(fields.get("endpoints")!) }
          : {}),
      };
      if (fields.get("mode") === "regular") {
        const regular = value as Record<string, unknown>;
        for (const name of [
          "namespace",
          "host_key",
          "host_noise_key",
          "host_binding_signature",
          "websocket_url",
          "invite_seed",
          "invite",
        ])
          if (!fields.has(name)) delete regular[name];
      }
    } catch {
      throw new Error("That QR code is not a Mold pairing code.");
    }
  }
  if (!value || typeof value !== "object")
    throw new Error("That QR code is not a Mold pairing code.");
  const raw = value as Record<string, unknown>;
  if (raw.mode === "regular") {
    try {
      if (
        raw.type !== "mold.mobile-pairing" ||
        raw.version !== 2 ||
        typeof raw.base_url !== "string" ||
        typeof raw.instance_id !== "string" ||
        !raw.instance_id ||
        typeof raw.name !== "string" ||
        typeof raw.pairing_token !== "string" ||
        !/^[0-9a-f]{64}$/.test(raw.pairing_token) ||
        !Number.isSafeInteger(raw.expires_at) ||
        (raw.expires_at as number) <= 0 ||
        [
          "namespace",
          "host_key",
          "host_noise_key",
          "host_binding_signature",
          "websocket_url",
          "invite_seed",
          "invite",
          "endpoints",
        ].some((key) => key in raw)
      )
        throw new Error();
      directOrigin(raw.base_url);
      return raw as unknown as RegularMobilePairingPayload;
    } catch {
      throw new Error("That QR code is not a supported Mold pairing code.");
    }
  }
  const payload = value as Partial<SecureMobilePairingPayload>;
  if (
    (payload.mode !== undefined && payload.mode !== "secure") ||
    payload.type !== "mold.mobile-pairing" ||
    payload.version !== 2 ||
    typeof payload.base_url !== "string" ||
    !validPairingAddress(payload.base_url, false) ||
    !/^[0-9a-f]{32}$/.test(payload.namespace ?? "") ||
    !/^[0-9a-f]{64}$/.test(payload.host_key ?? "") ||
    !/^[0-9a-f]{64}$/.test(payload.host_noise_key ?? "") ||
    !/^[0-9a-f]{128}$/.test(payload.host_binding_signature ?? "") ||
    !/^[0-9a-f]{64}$/.test(payload.invite_seed ?? "") ||
    (payload.websocket_url !== null &&
      (typeof payload.websocket_url !== "string" ||
        !validPairingAddress(payload.websocket_url, true))) ||
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
  return payload as SecureMobilePairingPayload;
}

function validPairingAddress(value: string, websocket: boolean): boolean {
  try {
    const url = new URL(value);
    return (
      value === value.trim() &&
      !/[\r\n\t]/.test(value) &&
      !url.username &&
      !url.password &&
      !url.search &&
      !url.hash &&
      (websocket
        ? url.protocol === "wss:"
        : ["http:", "https:"].includes(url.protocol) && url.pathname === "/")
    );
  } catch {
    return false;
  }
}
