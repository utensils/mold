import Foundation

/// Reading a host's capabilities, one documented absence rule at a time.
///
/// The app asks these questions and never the raw optionals, because the
/// answer to "is this missing" is different for every field: some absences are
/// a definitive no, some are an older host that can do it anyway, and some are
/// the presence of a number rather than a flag.
public extension Capabilities {

    // MARK: - Generation

    /// The presence of the batch number is how a client knows the host
    /// generates at all -- there is no separate boolean.
    var generates: Bool { queue?.heterogeneousBatchMaxOutputs != nil }

    var maxBatchOutputs: Int { queue?.heterogeneousBatchMaxOutputs ?? 1 }

    /// Durable work survives a dropped connection. Where this is true, a job
    /// whose stream died is still running and must NOT be dead-lettered.
    var hasDurableQueue: Bool { queue?.durableQueue ?? false }

    var canReorderQueue: Bool { queue?.canReorder ?? false }
    var canCancelAllQueued: Bool { queue?.canCancelAll ?? false }
    var canPauseQueue: Bool { queue?.canPause ?? false }
    var canPauseOneJob: Bool { queue?.canPauseJob ?? false }

    // MARK: - Gallery

    /// Tags, titles, favourites and collections. Absent means an older host
    /// with no organization tables, so the UI hides the controls rather than
    /// offering edits that will 404.
    var canOrganize: Bool { gallery?.organize ?? false }

    /// One request applying one change to many prints. Absent means fall back
    /// to a loop of single-print edits, which still works.
    var canBulkMutate: Bool { gallery?.bulkMutations ?? false }

    /// Absence means an older host that predates the field, so the app falls
    /// back to unconditional listing rather than assuming it is unsupported.
    var supportsConditionalGallery: Bool { gallery?.conditionalGet ?? false }

    /// Row-level gallery events let the app update one tile instead of
    /// re-listing 1,500.
    var supportsGalleryRowEvents: Bool { gallery?.rowEvents ?? false }

    /// A host that does not persist outputs hands back bytes and keeps
    /// nothing, so there is no library to show for it.
    var persistsOutputs: Bool { gallery?.persistsOutputs ?? true }

    /// Deleting moves to the trash rather than unlinking. Where this is false,
    /// Delete is permanent and must say so.
    var trashEnabled: Bool { gallery?.trash?.enabled ?? false }

    /// Days until a trashed print is purged. **Nil means kept forever** --
    /// both when the host says 0 and when it says nothing at all. Showing a
    /// literal 0 here would read as "purged today", which is the opposite.
    var trashRetentionDays: Int? {
        guard let days = gallery?.trash?.retentionDays, days > 0 else { return nil }
        return days
    }

    // MARK: - Events

    /// Absent means an older host with no event stream, so the app polls.
    var hasEvents: Bool { events?.available ?? false }

    // MARK: - Prompt expansion

    /// UNKNOWN, not no. Hosts expanded prompts before they advertised it, so
    /// the app offers the control and lets the request be the answer.
    var mayExpandPrompts: Bool { expand?.configured ?? true }

    /// Remix is a separate endpoint precisely so an older host fails closed
    /// instead of silently returning the wrong transform, so absence is no.
    var canRemixPrompts: Bool { expand?.remix ?? false }

    /// The local expander this host would use but does not have installed.
    /// Nil when it is installed, when the backend is an API, or when the host
    /// predates the field -- in none of which is there anything to offer.
    var expanderModelToPull: String? {
        guard let expand, expand.modelPresent == false else { return nil }
        return expand.model
    }

    // MARK: - Conditioning

    /// Definitively NO when absent: mold advertises this block only when the
    /// identity runtime is actually available.
    var supportsIdentity: Bool { identity != nil }

    /// How many photographs one identity may be built from.
    ///
    /// `multi_photo` is the gate, not `max_photos`: a host that does not
    /// understand `id_images` takes exactly one however large its advertised
    /// maximum (`types.rs:11563-11572`), and absence of the whole block is a
    /// definitive no (`types.rs:11555-11557`).
    var maxIdentityPhotos: Int {
        guard let identity else { return 0 }
        guard identity.multiPhoto == true else { return 1 }
        return max(identity.maxPhotos ?? 1, 1)
    }

    /// Advertised-but-off is real and common: the upload protocol needs
    /// API-key auth, so every keyless host reports `available: false`. That is
    /// not a reason to refuse a small reference -- it means send it inline.
    var canUploadLargeReferences: Bool { referenceUploads?.available ?? false }

    // MARK: - Machines and models

    var canSeeDevices: Bool { devices?.available ?? false }

    /// A live enable/disable that the host will actually honour.
    var canChangeDeviceLifecycle: Bool { devices?.lifecycle ?? false }

    /// Enabling a device for the next restart, which some runtimes allow even
    /// where a live change is not authoritative.
    var canEnableDeviceAtRestart: Bool { devices?.restartEnable ?? false }

    /// Whether scheduler V2 owns dispatch on this host. Absent means a host
    /// with no V2 scheduler at all, so no -- and a live device change needs
    /// THIS as well as `devices.lifecycle`: one says the route exists, the
    /// other says the runtime will honour what it does.
    var dispatchIsAuthoritative: Bool { dispatch?.v2Authoritative ?? false }

    var canBrowsePeers: Bool { discovery?.canBrowse ?? false }

    var canBrowseCatalog: Bool { catalog?.available ?? false }

    /// The families this host will search for. Empty means it browses none.
    var catalogFamilies: [String] { catalog?.families ?? [] }

    var catalogSortOptions: [String] { catalog?.sort ?? [] }

    /// Licence acceptance is a route, and an older host without it simply
    /// never gates a model.
    var hasLicenses: Bool { licenses ?? false }

    /// Framewise clip upscaling. Its own `disclosure` sentence rides with it
    /// and is shown verbatim wherever the action is offered.
    var canUpscaleClips: Bool { videoUpscale?.available ?? false }
}
