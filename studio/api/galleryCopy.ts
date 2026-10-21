import { ApiError, apiFetchTo } from "./client";
import {
  captureMirrorOrganization,
  applyMirrorOrganization,
  type MirrorOrganization,
} from "./galleryMirrorOrganization";
import { retainedSourceMediaDisclosable } from "./gallerySourceMedia";
import {
  declaredVideoArtifacts,
  requireVideoArtifactTransfer,
  transferGalleryVideoArtifacts,
} from "./galleryVideoArtifacts";
import {
  embeddedMirrorMetadata,
  mirrorMetadataMatches,
} from "../lib/galleryMirrorIdentity";
import type { CopyOffer, CopyOptions, CopyReceipt } from "./galleryCopyTypes";
import {
  MAX_BYTES,
  MAX_JSON,
  bounded,
  json,
  offer,
  hash,
  sameOutput,
  copyMachineIdentity,
  legacyInventoryHasNoMembers,
} from "./galleryCopyWire";
export type {
  CopyMachine,
  CopyOffer,
  CopyOptions,
  CopyReceipt,
  CopyJournal,
} from "./galleryCopyTypes";
export { copyMachineIdentity, rawJsonProperty } from "./galleryCopyWire";

let tail: Promise<unknown> = Promise.resolve();
/** Globally serialized, bounded transfer. Incomplete receipts survive failures for repair. */
export function copyGalleryPrint(options: CopyOptions): Promise<CopyReceipt> {
  const run = tail
    .catch(() => undefined)
    .then(() =>
      options.journal.runExclusive
        ? options.journal.runExclusive(
            (lockSignal) =>
              transfer(
                lockSignal ? { ...options, signal: lockSignal } : options,
              ),
            options.signal,
          )
        : transfer(options),
    );
  tail = run;
  return run;
}
async function transfer(options: CopyOptions): Promise<CopyReceipt> {
  const { source, destination, filename, journal, signal } = options;
  const limit =
    options.nativeImportOutput &&
    options.nativeTransferSources &&
    options.nativeMatchesOutput
      ? 512 * 1024 * 1024
      : MAX_BYTES;
  const check = () => {
    signal?.throwIfAborted();
    if (!options.current())
      throw new Error(
        "The source, destination, or connection changed. Select Copy again.",
      );
  };
  check();
  if (
    !filename ||
    filename === "." ||
    filename === ".." ||
    /[\\/\u0000]/.test(filename)
  )
    throw new Error("Choose a valid Library filename.");
  if (
    !source.instanceId ||
    !destination.instanceId ||
    copyMachineIdentity(source) === copyMachineIdentity(destination)
  )
    throw new Error("Choose two verified, different machines.");
  const sourceIdentity = copyMachineIdentity(source),
    destinationIdentity = copyMachineIdentity(destination);
  // Recheck the servers themselves: a saved route can point at a reinstalled host.
  for (const machine of [source, destination]) {
    const status = await json(machine.target, "/api/status", signal);
    check();
    if (status.value.instance_id !== machine.instanceId)
      throw new Error(
        "The machine installation changed. Reconnect before copying.",
      );
  }
  const origin = await offer(
    source,
    filename,
    signal,
    limit,
    options.nativeLegacyOffer,
  );
  check();
  const artifacts = declaredVideoArtifacts(origin.metadata);
  if (artifacts) {
    await requireVideoArtifactTransfer(source, destination, check, signal);
    check();
    if (origin.video_artifacts_complete !== true)
      throw new Error(
        "The original video's saved audio and clip history are incomplete. Repair the source before copying.",
      );
  }
  let legacyInputsUnavailable = false;
  if (
    !origin.members.length &&
    retainedSourceMediaDisclosable(origin.metadata)
  ) {
    const inventory = await json(
      source.target,
      `/api/gallery/source-media/${encodeURIComponent(filename)}`,
      signal,
    );
    check();
    if (!legacyInventoryHasNoMembers(inventory.value))
      throw new Error(
        "This print's original source media could not be verified. The copy would be incomplete.",
      );
    legacyInputsUnavailable = true;
  }
  let downloaded: Uint8Array<ArrayBuffer> | undefined;
  const outputBytes = async () => {
    if (!downloaded) {
      downloaded = await bounded(
        await apiFetchTo(
          source.target,
          `/api/gallery/image/${encodeURIComponent(filename)}`,
          {
            signal: signal ?? null,
            headers: { "x-mold-source-instance": source.instanceId },
          },
        ),
        origin.output_size_bytes,
        signal,
      );
      check();
      if (
        downloaded.length !== origin.output_size_bytes ||
        (await hash(downloaded)) !== origin.output_sha256
      )
        throw new Error("The original print changed while downloading.");
      check();
    }
    return downloaded;
  };
  const matches = async (
    candidate: CopyOffer,
    candidateFilename = filename,
  ) => {
    if (sameOutput(origin, candidate)) return true;
    if (
      candidate.output_sha256 !== origin.output_sha256 ||
      candidate.output_size_bytes !== origin.output_size_bytes
    )
      return false;
    if (options.nativeMatchesOutput)
      return options.nativeMatchesOutput(
        source,
        destination,
        filename,
        candidateFilename,
        origin,
        candidate,
        signal,
      );
    return sameOutput(
      origin,
      candidate,
      embeddedMirrorMetadata(await outputBytes()),
    );
  };
  const missingOutput = async (name: string): Promise<boolean> => {
    const listings = await Promise.all(
      ["/api/gallery", "/api/gallery?view=trash"].map((path) =>
        json(destination.target, path, signal),
      ),
    );
    check();
    if (listings.some(({ value }) => !Array.isArray(value)))
      throw new Error(
        "The destination could not confirm its Library inventory.",
      );
    return !listings[0]!.value.some(
      (row: { filename?: string }) => row.filename === name,
    );
  };
  const recipe = await hash(new TextEncoder().encode(origin.rawMetadata));
  check();
  const key = JSON.stringify([
    sourceIdentity,
    destinationIdentity,
    filename,
    origin.output_sha256,
    recipe,
  ]);
  if (origin.members.length) {
    const capability = await json(
      destination.target,
      "/api/capabilities",
      signal,
    );
    check();
    if (capability.value.retained_media_transfer?.protocol_version !== 1)
      throw new Error(
        "The destination cannot retain this print's source media. Update that machine before copying.",
      );
  }
  let organization = await captureMirrorOrganization(source.target, filename);
  check();
  const prior = await journal.read(key);
  check();
  const matching =
    prior ??
    (await journal.matching?.(
      destinationIdentity,
      origin.output_sha256,
      recipe,
    )) ??
    null;
  check();
  const lineage =
    (await journal.lineage?.(
      sourceIdentity,
      filename,
      origin.output_sha256,
      recipe,
    )) ?? null;
  check();
  const original =
    lineage?.origin ?? JSON.stringify([sourceIdentity, filename]);
  const copiedSource =
    !!lineage && original !== JSON.stringify([sourceIdentity, filename]);
  const verifiedRoot = (await journal.original?.(original)) ?? null;
  const root = copiedSource ? (verifiedRoot ?? lineage) : null;
  check();
  const sourceSnapshot = organization;
  if (copiedSource && root?.sourceOrganization)
    organization = root.sourceOrganization;
  const echoesOrigin =
    original ===
    JSON.stringify([
      destinationIdentity,
      matching?.destinationFilename ?? filename,
    ]);
  const receipt: CopyReceipt = {
    version: 1,
    source: sourceIdentity,
    destination: destinationIdentity,
    sourceFilename: filename,
    digest: origin.output_sha256,
    size: origin.output_size_bytes,
    recipe,
    stage: "intent",
    origin: original,
    outcome: legacyInputsUnavailable ? "legacy_inputs_unavailable" : "complete",
    sourceOrganization: organization,
    observedAt: copiedSource
      ? (root?.observedAt ?? 0)
      : Math.max(Date.now(), (verifiedRoot?.observedAt ?? 0) + 1),
    ...(prior?.importAttempted ? { importAttempted: true } : {}),
    ...(prior?.pendingOrganization
      ? { pendingOrganization: prior.pendingOrganization }
      : {}),
    ...(prior?.organization ? { organization: prior.organization } : {}),
    ...(matching?.recipe === recipe &&
    matching?.digest === origin.output_sha256 &&
    matching.size === origin.output_size_bytes &&
    matching.destination === destinationIdentity &&
    matching.destinationFilename
      ? { destinationFilename: matching.destinationFilename }
      : {}),
  };
  const checkpoint = async (
    stage: CopyReceipt["stage"],
    preserveReceipt = false,
  ) => {
    if (!preserveReceipt) check();
    receipt.stage = stage;
    await journal.write(key, { ...receipt });
    check();
    options.progress?.(stage);
  };
  await checkpoint("intent");
  let targetCopyOffer: CopyOffer | undefined;
  if (receipt.destinationFilename) {
    try {
      targetCopyOffer = await offer(
        destination,
        receipt.destinationFilename,
        signal,
        limit,
        options.nativeLegacyOffer,
      );
    } catch (failure) {
      if (
        !(failure instanceof ApiError) ||
        failure.status !== 404 ||
        !(await missingOutput(receipt.destinationFilename))
      )
        throw failure;
      if (options.deliberate === false && matching?.destinationFilename) {
        // Verified graph copies remain removed even when this source edge is
        // new. Do not adopt another source's organization ownership.
        const removed: CopyReceipt = prior ?? {
          ...receipt,
          destinationFilename: matching.destinationFilename,
          stage: "complete",
        };
        await journal.write(key, removed);
        check();
        return removed;
      }
      delete receipt.destinationFilename;
    }
    check();
    if (
      targetCopyOffer &&
      !(await matches(targetCopyOffer, receipt.destinationFilename ?? filename))
    )
      throw new Error(
        "The destination copy has different output bytes or generation settings.",
      );
  }
  if (!receipt.destinationFilename) {
    // Recover an import whose successful reply was lost before its receipt was durable.
    const candidates = [filename];
    if (prior?.stage === "intent") {
      const listing = await json(destination.target, "/api/gallery", signal);
      check();
      if (!Array.isArray(listing.value))
        throw new Error(
          "The destination returned an invalid Library inventory.",
        );
      candidates.push(
        ...listing.value
          .map((row: { filename: string }) => row.filename)
          .filter((name: string) => name !== filename),
      );
    }
    for (const candidate of candidates) {
      let found: CopyOffer;
      try {
        found = await offer(
          destination,
          candidate,
          signal,
          limit,
          options.nativeLegacyOffer,
        );
      } catch (failure) {
        if (failure instanceof ApiError && failure.status === 404) continue;
        throw failure;
      }
      check();
      if (await matches(found, candidate)) {
        receipt.destinationFilename = candidate;
        targetCopyOffer = found;
        break;
      }
    }
  }
  if (!receipt.destinationFilename && options.nativeImportOutput) {
    receipt.importAttempted = true;
    await checkpoint("intent");
    receipt.destinationFilename = await options.nativeImportOutput(
      source,
      destination,
      filename,
      origin,
      signal,
    );
    if (
      !receipt.destinationFilename ||
      /[\\/\u0000]/.test(receipt.destinationFilename)
    )
      throw new Error("The destination returned an invalid filename.");
  }
  if (!receipt.destinationFilename) {
    const bytes = await outputBytes();
    const embedded = embeddedMirrorMetadata(bytes);
    if (
      embedded &&
      !mirrorMetadataMatches(origin.rawMetadata, embedded, embedded)
    )
      throw new Error(
        "The output's embedded recipe conflicts with its archive.",
      );
    const synthetic = embedded
      ? false
      : (sourceSnapshot.item.metadata_synthetic ?? false);
    const timestamp = sourceSnapshot.item.timestamp;
    const descriptor = new TextEncoder().encode(
      `{"metadata":${origin.rawMetadata},"metadata_synthetic":${JSON.stringify(synthetic)}${Number.isSafeInteger(timestamp) && timestamp! >= 0 ? `,"timestamp":${timestamp}` : ""}}`,
    );
    const prefix = new Uint8Array(12);
    const view = new DataView(prefix.buffer);
    view.setUint32(0, descriptor.length);
    view.setBigUint64(4, BigInt(bytes.length));
    receipt.importAttempted = true;
    await checkpoint("intent");
    check();
    const imported = await bounded(
      await apiFetchTo(
        destination.target,
        `/api/gallery/import/${encodeURIComponent(filename)}`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/vnd.mold.gallery-import",
            "x-mold-destination-instance": destination.instanceId,
          },
          body: new Blob([prefix, descriptor, bytes]),
          signal: signal ?? null,
        },
      ),
      MAX_JSON,
      signal,
    );
    check();
    receipt.destinationFilename = JSON.parse(
      new TextDecoder().decode(imported),
    ).filename;
    if (
      !receipt.destinationFilename ||
      /[\\/\u0000]/.test(receipt.destinationFilename)
    )
      throw new Error("The destination returned an invalid filename.");
  }
  await checkpoint("output", true);
  const targetName = receipt.destinationFilename!;
  targetCopyOffer ??= await offer(
    destination,
    targetName,
    signal,
    limit,
    options.nativeLegacyOffer,
  );
  check();
  if (
    !(await matches(targetCopyOffer, receipt.destinationFilename ?? filename))
  )
    throw new Error(
      "The destination copy has different output bytes or generation settings.",
    );
  receipt.destinationRecipe = await hash(
    new TextEncoder().encode(targetCopyOffer.rawMetadata),
  );
  await checkpoint("output");
  if (origin.members.length && options.nativeTransferSources) {
    check();
    await options.nativeTransferSources(
      source,
      destination,
      filename,
      targetName,
      origin,
      targetCopyOffer,
      signal,
    );
    check();
  } else if (origin.members.length) {
    const descriptor = new TextEncoder().encode(
      JSON.stringify({
        archive_identity_sha256: targetCopyOffer.archive_identity_sha256,
        members: origin.members.map(({ member_id: _id, ...rest }) => rest),
      }),
    );
    if (descriptor.length > 64 * 1024)
      throw new Error("Retained source-media descriptor is too large.");
    const prefix = new Uint8Array(4);
    new DataView(prefix.buffer).setUint32(0, descriptor.length);
    const chunks: BlobPart[] = [prefix, descriptor];
    for (const member of origin.members) {
      check();
      const bytes = await bounded(
        await apiFetchTo(
          source.target,
          `/api/gallery/source-media/${encodeURIComponent(filename)}/${encodeURIComponent(member.member_id!)}`,
          {
            signal: signal ?? null,
            headers: { "x-mold-source-instance": source.instanceId },
          },
        ),
        member.size_bytes,
        signal,
      );
      check();
      if (
        bytes.length !== member.size_bytes ||
        (await hash(bytes)) !== member.sha256
      )
        throw new Error("Retained source media changed during copying.");
      chunks.push(bytes);
    }
    check();
    const result = JSON.parse(
      new TextDecoder().decode(
        await bounded(
          await apiFetchTo(
            destination.target,
            `/api/gallery/source-media/${encodeURIComponent(targetName)}/transfer`,
            {
              method: "PUT",
              headers: {
                "Content-Type": "application/vnd.mold.retained-media-transfer",
                "x-mold-destination-instance": destination.instanceId,
                "x-mold-gallery-archive-identity":
                  targetCopyOffer.archive_identity_sha256,
              },
              body: new Blob(chunks),
              signal: signal ?? null,
            },
          ),
          MAX_JSON,
          signal,
        ),
      ),
    );
    check();
    if (
      result.archive_identity_sha256 !==
        targetCopyOffer.archive_identity_sha256 ||
      result.member_count !== origin.members.length
    )
      throw new Error(
        "Source media transfer was not confirmed. Copy again to repair this print.",
      );
  }
  if (artifacts) {
    await transferGalleryVideoArtifacts({
      source,
      destination,
      sourceFilename: filename,
      destinationFilename: targetName,
      sourceArchive: origin.archive_identity_sha256,
      destinationArchive: targetCopyOffer.archive_identity_sha256,
      artifacts,
      check,
      ...(signal ? { signal } : {}),
    });
    check();
  }
  await checkpoint("sources");
  const contributors =
    (await journal.contributors?.(destinationIdentity, targetName, key)) ?? [];
  check();
  // Copies may carry local scalar edits; only the verified original edge's
  // last-applied values can travel as that origin's scalar authority.
  if (copiedSource && lineage?.organization && !root?.sourceOrganization) {
    organization.item = {
      ...organization.item,
      ...(Object.hasOwn(lineage.organization, "title")
        ? { title: lineage.organization.title ?? null }
        : {}),
      ...(Object.hasOwn(lineage.organization, "favorite")
        ? { favorite: lineage.organization.favorite }
        : {}),
    };
  }
  const applied =
    options.deliberate === false && echoesOrigin
      ? undefined
      : await applyMirrorOrganization(
          destination.target,
          targetName,
          organization,
          {
            ...(receipt.organization ? { prior: receipt.organization } : {}),
            contributors,
            deliberate: options.deliberate ?? true,
            current: () => !signal?.aborted && options.current(),
            origin: original,
            scalarAuthority:
              !copiedSource ||
              !!root?.sourceOrganization ||
              !!lineage.organization,
            ...(copiedSource && lineage && !root?.sourceOrganization
              ? {
                  scalarFields: ["title", "favorite"].filter(
                    (field) =>
                      lineage.organization &&
                      Object.hasOwn(lineage.organization, field),
                  ),
                }
              : {}),
            observeOnly:
              options.deliberate === false &&
              !prior?.organization &&
              !receipt.importAttempted,
            ...(receipt.pendingOrganization
              ? { pending: receipt.pendingOrganization }
              : {}),
            intent: async (state) => {
              receipt.pendingOrganization = state;
              await checkpoint("sources");
            },
          },
        );
  check();
  if (applied) receipt.organization = applied;
  delete receipt.pendingOrganization;
  await checkpoint("organization");
  const finalOrganization = await captureMirrorOrganization(
    source.target,
    filename,
  );
  check();
  const organizationKey = (snapshot: MirrorOrganization) =>
    JSON.stringify([
      snapshot.item.title ?? null,
      snapshot.item.favorite === true,
      [...(snapshot.item.tags ?? [])].sort(),
      snapshot.item.metadata_synthetic ?? false,
      snapshot.collections
        .map((collection) => [
          collection.slug ?? collection.name,
          collection.hidden === true,
        ])
        .sort((a, b) => String(a[0]).localeCompare(String(b[0]))),
    ]);
  if (organizationKey(sourceSnapshot) !== organizationKey(finalOrganization))
    throw new Error(
      "Source organization changed during copying. Copy again to reconcile it.",
    );
  const finalSource = await offer(
    source,
    filename,
    signal,
    limit,
    options.nativeLegacyOffer,
  );
  check();
  if (!(await matches(finalSource)))
    throw new Error("The source output changed during copying.");
  if (origin.archive_identity_sha256 !== finalSource.archive_identity_sha256)
    throw new Error(
      "Source media changed during copying. Copy again to repair it.",
    );
  const finalDestination = await offer(
    destination,
    targetName,
    signal,
    limit,
    options.nativeLegacyOffer,
  );
  check();
  if (!(await matches(finalDestination, targetName)))
    throw new Error("The destination output changed during copying.");
  if (
    artifacts &&
    (finalSource.video_artifacts_complete !== true ||
      finalDestination.video_artifacts_complete !== true)
  )
    throw new Error(
      "The video's saved audio and clip history could not be verified. Copy again to repair it.",
    );
  const comparable = (o: CopyOffer) =>
    JSON.stringify(
      o.members.map((member) => [
        member.role,
        member.position,
        member.sink,
        member.size_bytes,
        member.sha256,
      ]),
    );
  if (
    !legacyInputsUnavailable &&
    comparable(origin) !== comparable(finalDestination)
  )
    throw new Error(
      "The destination's retained sources could not be verified. Copy again to repair it.",
    );
  for (const machine of [source, destination]) {
    const status = await json(machine.target, "/api/status", signal);
    check();
    if (status.value.instance_id !== machine.instanceId)
      throw new Error("The machine installation changed during copying.");
  }
  await checkpoint("complete");
  return receipt;
}
