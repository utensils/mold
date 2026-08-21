import type { ApiTarget } from "./client";
import { apiFetchTo, apiJsonTo } from "./client";

export interface PairingSession {
  token: string | null;
  expires_at: number | null;
  auth_required: boolean;
  instance_id: string;
  hostname: string | null;
}

export interface PairingClaim {
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
  type: "mold.mobile-pairing";
  version: 1;
  base_url: string;
  token: string | null;
  expires_at: number | null;
  instance_id: string;
  name: string;
}

export function mobilePairingUrl(payload: MobilePairingPayload): string {
  const url = new URL("mold://pair");
  url.searchParams.set("version", String(payload.version));
  url.searchParams.set("base_url", payload.base_url);
  if (payload.token !== null) url.searchParams.set("token", payload.token);
  if (payload.expires_at !== null)
    url.searchParams.set("expires_at", String(payload.expires_at));
  url.searchParams.set("instance_id", payload.instance_id);
  url.searchParams.set("name", payload.name);
  return url.toString();
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
      const url = new URL(raw);
      if (
        url.protocol !== "mold:" ||
        url.hostname !== "pair" ||
        url.username ||
        url.password ||
        url.hash
      ) {
        throw new Error();
      }
      const expiresAt = url.searchParams.get("expires_at");
      value = {
        type: "mold.mobile-pairing",
        version: Number(url.searchParams.get("version")),
        base_url: url.searchParams.get("base_url"),
        token: url.searchParams.get("token"),
        expires_at: expiresAt === null ? null : Number(expiresAt),
        instance_id: url.searchParams.get("instance_id"),
        name: url.searchParams.get("name"),
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
  return payload as MobilePairingPayload;
}
