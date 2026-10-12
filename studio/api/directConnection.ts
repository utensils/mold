import { apiJsonTo, parseCurrentServerStatus } from "./client";
import { directOrigin, rememberConnectionMode } from "./connectionMode";
export interface RegularConnection {
  baseUrl: string;
  apiKey: string | null;
  name: string;
  instanceId: string;
  connectionMode: "regular";
}
/** A successful explicit direct probe is the only way a manual choice is saved. */
export async function connectRegularHost(
  address: string,
  apiKey: string | null,
): Promise<RegularConnection> {
  const baseUrl = directOrigin(address);
  const target = { baseUrl, apiKey, connectionMode: "regular" as const };
  const status = parseCurrentServerStatus(
    await apiJsonTo(target, "/api/status", {
      signal: AbortSignal.timeout(10000),
    }),
  );
  await registerRegularConnection(baseUrl, apiKey);
  return {
    ...target,
    name: status.hostname || new URL(baseUrl).hostname,
    instanceId: status.instance_id,
  };
}

let platformRegister:
  ((origin: string, apiKey: string | null) => Promise<void>) | undefined;
let platformForget: ((origin: string) => Promise<void>) | undefined;
export function configureRegularPlatform(
  register: typeof platformRegister,
  forget: typeof platformForget,
): void {
  platformRegister = register;
  platformForget = forget;
}
export async function registerRegularConnection(
  origin: string,
  apiKey: string | null,
): Promise<void> {
  await platformRegister?.(directOrigin(origin), apiKey);
  rememberConnectionMode(origin, "regular");
}
export async function forgetRegularConnection(origin: string): Promise<void> {
  await platformForget?.(new URL(origin).origin);
}
