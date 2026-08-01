import type { ApiTarget } from "./client";
import { apiJsonTo } from "./client";

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

export interface MobilePairingPayload {
  type: "mold.mobile-pairing";
  version: 1;
  base_url: string;
  token: string | null;
  expires_at: number | null;
  instance_id: string;
  name: string;
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
): Promise<PairingClaim> {
  return apiJsonTo<PairingClaim>(
    { baseUrl, apiKey: null },
    "/api/pairing/claim",
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ token }),
    },
  );
}

export function parseMobilePairingPayload(raw: string): MobilePairingPayload {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    throw new Error("That QR code is not a Mold pairing code.");
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
    (payload.expires_at !== null && typeof payload.expires_at !== "number") ||
    typeof payload.instance_id !== "string" ||
    typeof payload.name !== "string"
  ) {
    throw new Error("That QR code is not a supported Mold pairing code.");
  }
  return payload as MobilePairingPayload;
}
