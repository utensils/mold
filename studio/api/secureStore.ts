import {
  Identity,
  checkGrant,
  checkHost,
  loadSecureCrypto,
  type SecureHostProfile,
  type SecureGrant,
} from "./secureCrypto";

interface Sealed {
  revision: string;
  iv: Uint8Array<ArrayBuffer>;
  cipher: ArrayBuffer;
}
let database: Promise<IDBDatabase> | undefined;
let profiles: Promise<SecureHostProfile[]> | undefined;
let platformRegister:
  | ((profile: SecureHostProfile, identity: Identity) => Promise<void>)
  | undefined;
let platformForget: ((profile: SecureHostProfile) => Promise<void>) | undefined;
const platformTasks = new Map<string, Promise<void>>();
function serializePlatform(
  namespace: string,
  operation: () => Promise<void>,
): Promise<void> {
  const task = (platformTasks.get(namespace) ?? Promise.resolve())
    .catch(() => {})
    .then(operation);
  platformTasks.set(namespace, task);
  const finished = () => {
    if (platformTasks.get(namespace) === task) platformTasks.delete(namespace);
  };
  void task.then(finished, finished);
  return task;
}
const registered = new Map<string, string>();
export function configureSecurePlatform(
  register: typeof platformRegister,
  forget?: typeof platformForget,
): void {
  platformRegister = register;
  platformForget = forget;
  registered.clear();
}
async function registerPlatform(profile: SecureHostProfile): Promise<void> {
  if (
    !platformRegister ||
    registered.get(profile.namespace) === JSON.stringify(profile)
  )
    return;
  await serializePlatform(profile.namespace, async () => {
    const sealed = await get<Sealed>(`host:${profile.namespace}`);
    if (!sealed || !platformRegister) return;
    const plain = await open(`host:${profile.namespace}`, sealed);
    let current: SecureHostProfile;
    try {
      current = JSON.parse(
        new TextDecoder().decode(plain),
      ) as SecureHostProfile;
    } finally {
      plain.fill(0);
    }
    // A queued registration cannot recreate a forgotten or replaced grant.
    if (JSON.stringify(current) !== JSON.stringify(profile)) return;
    const identity = await deviceIdentity();
    try {
      await platformRegister(profile, identity);
      registered.set(profile.namespace, JSON.stringify(profile));
    } finally {
      identity.close();
      identity.free();
    }
  });
}
function changed(): void {
  profiles = undefined;
  if (typeof window !== "undefined")
    window.dispatchEvent(new Event("mold:secure-pairings-changed"));
}
const updates =
  typeof BroadcastChannel !== "undefined"
    ? new BroadcastChannel("mold-secure-v3")
    : null;
updates?.addEventListener("message", () => {
  registered.clear();
  changed();
});

function db(): Promise<IDBDatabase> {
  database ??= new Promise((resolve, reject) => {
    const request = indexedDB.open("mold-secure-v3", 1);
    request.onupgradeneeded = () => request.result.createObjectStore("secrets");
    request.onsuccess = () => {
      request.result.onversionchange = () => {
        request.result.close();
        database = undefined;
      };
      resolve(request.result);
    };
    request.onerror = () => {
      database = undefined;
      reject(new Error("Secure device storage is unavailable."));
    };
  });
  return database;
}
async function get<T>(key: string): Promise<T | undefined> {
  const database = await db();
  return new Promise((resolve, reject) => {
    const request = database
      .transaction("secrets", "readonly")
      .objectStore("secrets")
      .get(key);
    request.onsuccess = () => resolve(request.result as T | undefined);
    request.onerror = () =>
      reject(new Error("Could not read secure device storage."));
  });
}
async function put<T>(key: string, value: T, onlyIfAbsent = false): Promise<T> {
  const database = await db();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction("secrets", "readwrite");
    const store = transaction.objectStore("secrets");
    let selected = value;
    const request = store.get(key);
    request.onsuccess = () => {
      if (onlyIfAbsent && request.result !== undefined)
        selected = request.result as T;
      else store.put(value, key);
    };
    transaction.oncomplete = () => resolve(selected);
    transaction.onerror = transaction.onabort = () =>
      reject(new Error("Could not save secure device storage."));
  });
}
async function wrappingKey(): Promise<CryptoKey> {
  const existing = await get<CryptoKey>("wrapping-key");
  if (existing) return existing;
  if (!crypto.subtle)
    throw new Error("Open the secure Studio website to pair this browser.");
  const generated = await crypto.subtle.generateKey(
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"],
  );
  return put("wrapping-key", generated, true);
}
async function seal(
  key: string,
  bytes: Uint8Array<ArrayBuffer>,
): Promise<Sealed> {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const cipher = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv, additionalData: new TextEncoder().encode(key) },
    await wrappingKey(),
    bytes,
  );
  return { revision: crypto.randomUUID(), iv, cipher };
}
async function open(
  key: string,
  sealed: Sealed,
): Promise<Uint8Array<ArrayBuffer>> {
  try {
    return new Uint8Array(
      await crypto.subtle.decrypt(
        {
          name: "AES-GCM",
          iv: sealed.iv,
          additionalData: new TextEncoder().encode(key),
        },
        await wrappingKey(),
        sealed.cipher,
      ),
    );
  } catch {
    throw new Error(
      "This browser's secure pairing could not be read. Pair it again from the host.",
    );
  }
}

/** Nonextractable WebCrypto storage key; decrypted identity lives only during protocol use.
 * Browser code and its independently trusted update origin remain trusted principals. */
export async function deviceIdentity(): Promise<Identity> {
  await loadSecureCrypto();
  let stored = await get<Sealed>("device-identity");
  if (!stored) {
    const generated = Identity.generate();
    const bytes = new Uint8Array(generated.toBytes());
    try {
      stored = await put(
        "device-identity",
        await seal("device-identity", bytes),
        true,
      );
    } finally {
      bytes.fill(0);
      generated.close();
      generated.free();
    }
  }
  const bytes = await open("device-identity", stored);
  try {
    return Identity.fromBytes(bytes);
  } finally {
    bytes.fill(0);
  }
}

export async function saveSecureHost(
  profile: SecureHostProfile,
): Promise<void> {
  const identity = await deviceIdentity();
  try {
    checkHost(profile);
    checkGrant(profile, identity, profile.grant);
  } finally {
    identity.close();
    identity.free();
  }
  if (
    !profile.apiKey ||
    profile.origins.length < 1 ||
    profile.origins.length > 8
  )
    throw new Error("Invalid secure host pairing.");
  for (const origin of profile.origins) {
    const url = new URL(origin);
    if (
      !["http:", "https:"].includes(url.protocol) ||
      url.origin !== origin ||
      url.username ||
      url.password
    )
      throw new Error("Invalid secure host address.");
  }
  const key = `host:${profile.namespace}`;
  const encoded = new TextEncoder().encode(JSON.stringify(profile));
  try {
    await put(key, await seal(key, encoded));
    changed();
    await registerPlatform(profile);
    updates?.postMessage("changed");
  } finally {
    encoded.fill(0);
  }
}

export async function listSecureHosts(): Promise<SecureHostProfile[]> {
  if (typeof indexedDB === "undefined") return [];
  profiles ??= (async () => {
    const database = await db();
    const keys = await new Promise<IDBValidKey[]>((resolve, reject) => {
      const request = database
        .transaction("secrets", "readonly")
        .objectStore("secrets")
        .getAllKeys();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () =>
        reject(new Error("Secure pairings could not be read."));
    });
    if (keys.length > 130) throw new Error("Too many stored secure pairings.");
    const result: SecureHostProfile[] = [];
    for (const key of keys) {
      if (typeof key !== "string" || !key.startsWith("host:")) continue;
      const sealed = await get<Sealed>(key);
      if (!sealed) continue;
      const plain = await open(key, sealed);
      try {
        result.push(
          JSON.parse(new TextDecoder().decode(plain)) as SecureHostProfile,
        );
      } finally {
        plain.fill(0);
      }
    }
    return result;
  })().catch((error: unknown) => {
    profiles = undefined;
    throw error;
  });
  return profiles;
}
export async function secureHostForOrigin(
  origin: string,
): Promise<SecureHostProfile | null> {
  const profile =
    (await listSecureHosts()).find((profile) =>
      profile.origins.includes(origin),
    ) ?? null;
  if (profile) await registerPlatform(profile);
  return profile;
}

export async function forgetSecureHost(namespace: string): Promise<void> {
  await serializePlatform(namespace, async () => {
    const removed = (await listSecureHosts()).find(
      (profile) => profile.namespace === namespace,
    );
    if (removed) await platformForget?.(removed);
    const database = await db();
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction("secrets", "readwrite");
      transaction.objectStore("secrets").delete(`host:${namespace}`);
      transaction.oncomplete = () => resolve();
      transaction.onerror = transaction.onabort = () =>
        reject(new Error("Could not remove secure pairing."));
    });
    if (removed && typeof window !== "undefined")
      window.dispatchEvent(
        new CustomEvent("mold:secure-host-forgotten", {
          detail: { origins: removed.origins },
        }),
      );
    registered.delete(namespace);
    changed();
    updates?.postMessage("removed");
  });
}

/** Forget every route for this pinned installation, including a changed LAN address. */
export async function forgetSecureHostForMachine(
  origin: string,
  instanceId?: string | null,
): Promise<void> {
  for (const profile of await listSecureHosts()) {
    if (
      profile.origins.includes(origin.replace(/\/$/, "")) ||
      (instanceId && profile.instance_id === instanceId)
    )
      await forgetSecureHost(profile.namespace);
  }
}

/** Discovery may update aliases, never revive a deleted pairing or roll back its authority. */
export async function updateSecureHostRoutes(
  namespace: string,
  expectedGrant: string,
  endpoints: SecureHostProfile["endpoints"],
  origins: string[],
): Promise<void> {
  await updateStoredProfile(namespace, expectedGrant, (profile) => ({
    ...profile,
    endpoints,
    origins,
  }));
}

/** A late renewal cannot recreate a forgotten host or overwrite a replacement pairing. */
export async function renewSecureHostGrant(
  profile: SecureHostProfile,
  grant: SecureGrant,
): Promise<boolean> {
  return updateStoredProfile(
    profile.namespace,
    profile.grant.signature,
    (current) => ({ ...current, grant }),
  );
}

async function updateStoredProfile(
  namespace: string,
  expectedGrant: string,
  update: (profile: SecureHostProfile) => SecureHostProfile,
): Promise<boolean> {
  const key = `host:${namespace}`;
  const current = await get<Sealed>(key);
  if (!current) return false;
  const plain = await open(key, current);
  let profile: SecureHostProfile;
  try {
    profile = JSON.parse(new TextDecoder().decode(plain)) as SecureHostProfile;
  } finally {
    plain.fill(0);
  }
  if (profile.grant.signature !== expectedGrant) return false;
  const bytes = new TextEncoder().encode(JSON.stringify(update(profile)));
  let next: Sealed;
  try {
    next = await seal(key, bytes);
  } finally {
    bytes.fill(0);
  }
  const database = await db();
  let updated = false;
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction("secrets", "readwrite");
    const store = transaction.objectStore("secrets");
    const request = store.get(key);
    request.onsuccess = () => {
      const stored = request.result as Sealed | undefined;
      if (
        stored &&
        stored.revision === current.revision &&
        stored.iv.join() === current.iv.join()
      ) {
        store.put(next, key);
        updated = true;
      }
    };
    transaction.oncomplete = () => resolve();
    transaction.onerror = transaction.onabort = () =>
      reject(new Error("Could not update secure addresses."));
  });
  if (updated) {
    profiles = undefined;
    updates?.postMessage("changed");
  }
  return updated;
}
