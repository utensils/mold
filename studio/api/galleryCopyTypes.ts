import type { ApiTarget } from "./client";
import type {
  MirrorOrganizationState,
  MirrorOrganizationIntent,
} from "../lib/mirrorOrganizationPolicy";
import type { MirrorOrganization } from "./galleryMirrorOrganization";

export interface CopyMachine {
  id: string;
  label: string;
  instanceId: string;
  target: ApiTarget;
}
interface Member {
  member_id?: string;
  role: string;
  position: string;
  sink: string;
  size_bytes: number;
  sha256: string;
}
export interface CopyOffer {
  archive_identity_sha256: string;
  output_sha256: string;
  output_size_bytes: number;
  metadata: object;
  members: Member[];
  rawMetadata: string;
  video_artifacts_complete?: boolean;
}
export interface CopyReceipt {
  version: 1;
  source: string;
  destination: string;
  sourceFilename: string;
  destinationFilename?: string;
  digest: string;
  size: number;
  organization?: MirrorOrganizationState;
  pendingOrganization?: MirrorOrganizationIntent;
  origin?: string;
  sourceOrganization?: MirrorOrganization;
  observedAt?: number;
  importAttempted?: boolean;
  outcome?: "complete" | "legacy_inputs_unavailable";
  recipe: string;
  /** Exact recipe verified on the destination; archive-only source facts may differ. */
  destinationRecipe?: string;
  stage: "intent" | "output" | "sources" | "organization" | "complete";
}
export interface CopyJournal {
  runExclusive?<T>(
    operation: (lockSignal?: AbortSignal) => Promise<T>,
    signal?: AbortSignal,
  ): Promise<T>;
  read(key: string): Promise<CopyReceipt | null>;
  write(key: string, receipt: CopyReceipt): Promise<void>;
  matching?(
    destination: string,
    digest: string,
    recipe: string,
  ): Promise<CopyReceipt | null>;
  contributors?(
    destination: string,
    filename: string,
    excluding: string,
  ): Promise<MirrorOrganizationState[]>;
  lineage?(
    source: string,
    filename: string,
    digest: string,
    recipe: string,
  ): Promise<CopyReceipt | null>;
  original?(origin: string): Promise<CopyReceipt | null>;
}
export interface CopyOptions {
  source: CopyMachine;
  destination: CopyMachine;
  filename: string;
  current(): boolean;
  /** Optional immutable listing/recipe fence, checked before any destination write. */
  verifySourceOffer?(offer: CopyOffer): void;
  journal: CopyJournal;
  signal?: AbortSignal;
  deliberate?: boolean;
  progress?(stage: CopyReceipt["stage"]): void;
  nativeLegacyOffer?(
    machine: CopyMachine,
    filename: string,
    signal?: AbortSignal,
  ): Promise<CopyOffer>;
  nativeImportOutput?(
    source: CopyMachine,
    destination: CopyMachine,
    filename: string,
    expected: CopyOffer,
    signal?: AbortSignal,
  ): Promise<string>;
  nativeTransferSources?(
    source: CopyMachine,
    destination: CopyMachine,
    sourceFilename: string,
    destinationFilename: string,
    sourceOffer: CopyOffer,
    destinationOffer: CopyOffer,
    signal?: AbortSignal,
  ): Promise<void>;
  nativeMatchesOutput?(
    source: CopyMachine,
    destination: CopyMachine,
    sourceFilename: string,
    destinationFilename: string,
    sourceOffer: CopyOffer,
    destinationOffer: CopyOffer,
    signal?: AbortSignal,
  ): Promise<boolean>;
}
