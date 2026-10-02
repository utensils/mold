import {
  parseConnectionEndpoints,
  type ConnectionEndpoint,
} from "./connectionRoutes";
import type { ApiTarget } from "./client";
import { apiFetchTo, apiJsonTo } from "./client";

export interface PairingSession {
  endpoints?: ConnectionEndpoint[];
  token: string | null;
  expires_at: number | null;
  auth_required: boolean;
  instance_id: string;
  hostname: string | null;
}

export interface PairingClaim {
  endpoints?: ConnectionEndpoint[];
  api_key: string | null;
  instance_id: string;
  hostname: string | null;
}

export interface PairingClientIdentity {
  name: string;
  kind: "iphone" | "ipad" | "android" | "mobile";
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

export interface MobilePairingPayload {
  endpoints?: ConnectionEndpoint[];
  type: "mold.mobile-pairing";
  version: 1;
  base_url: string;
  token: string | null;
  expires_at: number | null;
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
  if (payload.token !== null) fields.set("token", payload.token);
  if (payload.expires_at !== null)
    fields.set("expires_at", String(payload.expires_at));
  fields.set("instance_id", payload.instance_id);
  fields.set("name", payload.name);
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
 * `https://utensils.io/mold/pair#…`, or the query of the older
 * `mold://pair?…` (which the Tauri app still opens). `null` for anything else.
 */
function pairingLinkFields(url: URL): URLSearchParams | null {
  if (url.username || url.password) return null;
  if (url.protocol === "mold:") {
    return url.hostname === "pair" && !url.hash ? url.searchParams : null;
  }
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
  baseUrl: string,
  token: string | null,
  client: PairingClientIdentity = { name: "Mold mobile", kind: "mobile" },
): Promise<PairingClaim> {
  return apiJsonTo<PairingClaim>(
    { baseUrl, apiKey: null },
    "/api/pairing/claim",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        token,
        client_name: client.name,
        client_kind: client.kind,
      }),
    },
  );
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
      const expiresAt = fields.get("expires_at");
      value = {
        type: "mold.mobile-pairing",
        version: Number(fields.get("version")),
        base_url: fields.get("base_url"),
        token: fields.get("token"),
        expires_at: expiresAt === null ? null : Number(expiresAt),
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
    payload.version !== 1 ||
    typeof payload.base_url !== "string" ||
    !/^https?:\/\//i.test(payload.base_url) ||
    (payload.token !== null && typeof payload.token !== "string") ||
    (payload.expires_at !== null &&
      (typeof payload.expires_at !== "number" ||
        !Number.isFinite(payload.expires_at))) ||
    typeof payload.instance_id !== "string" ||
    typeof payload.name !== "string"
  ) {
    throw new Error("That QR code is not a supported Mold pairing code.");
  }
  if (payload.endpoints !== undefined)
    payload.endpoints = parseConnectionEndpoints(payload.endpoints);
  return payload as MobilePairingPayload;
}
