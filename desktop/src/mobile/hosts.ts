// Mobile and desktop intentionally share host normalization and secret-key
// slugs. Keeping one implementation prevents the same remote from acquiring
// different identities across the two clients.
export {
  hostIdFromUrl as remoteHostId,
  normalizeHostUrl as normalizeRemoteAddress,
} from "../lib/hosts";
