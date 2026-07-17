import { ipc } from "./ipc";

export const HF_TOKEN_HEADER = "X-Mold-HF-Token";
export const CIVITAI_TOKEN_HEADER = "X-Mold-Civitai-Token";

/**
 * Catalog credentials are forwarded only to an explicitly remote Mold host.
 * They never leave the desktop in ordinary generation/gallery requests and
 * never replace the remote server's own valid credentials.
 */
export async function catalogCredentialHeaders(forwardToRemote: boolean): Promise<Headers> {
  const headers = new Headers();
  if (!forwardToRemote) return headers;

  const [hfToken, civitaiToken] = await Promise.all([
    ipc.secretGet("hf-token"),
    ipc.secretGet("civitai-token"),
  ]);
  if (hfToken?.trim()) headers.set(HF_TOKEN_HEADER, hfToken.trim());
  if (civitaiToken?.trim()) headers.set(CIVITAI_TOKEN_HEADER, civitaiToken.trim());
  return headers;
}
