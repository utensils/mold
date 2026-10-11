/* tslint:disable */
/* eslint-disable */

export class DecodedRecord {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    kind(): number;
    payload(): Uint8Array;
    streamId(): number;
}

export class Identity {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    binding(): Uint8Array;
    close(): void;
    static fromBytes(bytes: Uint8Array): Identity;
    static generate(): Identity;
    publicNoiseKey(): Uint8Array;
    publicSigningKey(): Uint8Array;
    sign(bytes: Uint8Array): Uint8Array;
    toBytes(): Uint8Array;
}

export class Session {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    close(): void;
    static create(identity: Identity, config: string): Session;
    established(): boolean;
    handshakeHash(): Uint8Array;
    handshakeRead(bytes: Uint8Array): void;
    handshakeWrite(): Uint8Array;
    open(bytes: Uint8Array): Uint8Array;
    seal(bytes: Uint8Array): Uint8Array;
}

export function decodeRecord(bytes: Uint8Array): DecodedRecord;

export function encodeRecord(kind: number, stream_id: number, payload: Uint8Array): Uint8Array;

export function namespace(_public: Uint8Array): string;

export function verifySignature(_public: Uint8Array, message: Uint8Array, signature: Uint8Array): boolean;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_decodedrecord_free: (a: number, b: number) => void;
    readonly __wbg_identity_free: (a: number, b: number) => void;
    readonly __wbg_session_free: (a: number, b: number) => void;
    readonly decodeRecord: (a: number, b: number, c: number) => void;
    readonly decodedrecord_kind: (a: number) => number;
    readonly decodedrecord_payload: (a: number, b: number) => void;
    readonly decodedrecord_streamId: (a: number) => number;
    readonly encodeRecord: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly identity_binding: (a: number, b: number) => void;
    readonly identity_close: (a: number) => void;
    readonly identity_fromBytes: (a: number, b: number, c: number) => void;
    readonly identity_generate: (a: number) => void;
    readonly identity_publicNoiseKey: (a: number, b: number) => void;
    readonly identity_publicSigningKey: (a: number, b: number) => void;
    readonly identity_sign: (a: number, b: number, c: number, d: number) => void;
    readonly identity_toBytes: (a: number, b: number) => void;
    readonly namespace: (a: number, b: number, c: number) => void;
    readonly session_close: (a: number) => void;
    readonly session_create: (a: number, b: number, c: number, d: number) => void;
    readonly session_established: (a: number) => number;
    readonly session_handshakeHash: (a: number, b: number) => void;
    readonly session_handshakeRead: (a: number, b: number, c: number, d: number) => void;
    readonly session_handshakeWrite: (a: number, b: number) => void;
    readonly session_open: (a: number, b: number, c: number, d: number) => void;
    readonly session_seal: (a: number, b: number, c: number, d: number) => void;
    readonly verifySignature: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly __wbindgen_export: (a: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export2: (a: number, b: number) => number;
    readonly __wbindgen_export3: (a: number, b: number, c: number) => void;
    readonly __wbindgen_export4: (a: number, b: number, c: number, d: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
