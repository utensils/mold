import Foundation
import MoldClient

// Every `MoldBackend` route, one line each, answered by `FakeBackend`'s stubs.
// Mechanical on purpose: a new requirement on `MoldBackend` fails to compile
// here until it gets its one line, named by its Swift selector.
extension FakeBackend {
    // MARK: MoldActivityBackend

    public func activity() async throws -> ActiveWorkSnapshot { try await respond("activity()", []) }

    // MARK: MoldCatalogBackend

    public func installCatalogEntry(id: String) async throws -> CatalogInstall { try await respond("installCatalogEntry(id:)", [id]) }
    public func searchCatalog(_ query: CatalogQuery) async throws -> CatalogListing { try await respond("searchCatalog(_:)", [query]) }
    public func catalogEntry(id: String) async throws -> CatalogEntry { try await respond("catalogEntry(id:)", [id]) }
    public func catalogCredentials() async throws -> CatalogCredentialStatus { try await respond("catalogCredentials()", []) }
    @discardableResult public func setCatalogCredential(_ provider: String, token: String) async throws -> CatalogCredentialStatus { try await respond("setCatalogCredential(_:token:)", [provider, token]) }
    @discardableResult public func clearCatalogCredential(_ provider: String) async throws -> CatalogCredentialStatus { try await respond("clearCatalogCredential(_:)", [provider]) }

    // MARK: MoldChainBackend

    public func createChainJob(_ request: AutoChainRequest, operationId: String) async throws -> CreateChainJobResponse { try await respond("createChainJob(_:operationId:)", [request, operationId]) }
    public func chainJobEvents(id: String) -> AsyncThrowingStream<ChainJobEvent, Error> { stream("chainJobEvents(id:)", [id]) }
    public func chainJob(id: String) async throws -> ChainJobDetail { try await respond("chainJob(id:)", [id]) }
    public func cancelChainJob(id: String) async throws { let _: Void = try await respond("cancelChainJob(id:)", [id]) }
    public func resumeChainJob(id: String) async throws { let _: Void = try await respond("resumeChainJob(id:)", [id]) }
    public func chainLimits(model: String, fps: Int?) async throws -> ChainLimits { try await respond("chainLimits(model:fps:)", [model, fps]) }

    // MARK: MoldConfigBackend

    public func config() async throws -> ConfigListing { try await respond("config()", []) }
    @discardableResult public func setConfig(_ key: String, to value: ConfigScalar) async throws -> ConfigEntry { try await respond("setConfig(_:to:)", [key, value]) }
    @discardableResult public func resetConfig(_ key: String) async throws -> ConfigEntry { try await respond("resetConfig(_:)", [key]) }
    public func configProfiles() async throws -> ConfigProfiles { try await respond("configProfiles()", []) }
    public func pairingSession() async throws -> PairingSession { try await respond("pairingSession()", []) }
    public func pairedClients() async throws -> PairedClients { try await respond("pairedClients()", []) }
    public func revokePairedClient(_ id: String) async throws { let _: Void = try await respond("revokePairedClient(_:)", [id]) }

    // MARK: MoldCreateBackend

    public func expand(_ request: ExpandRequest) async throws -> ExpandResponse { try await respond("expand(_:)", [request]) }
    public func remix(_ request: RemixRequest) async throws -> RemixResponse { try await respond("remix(_:)", [request]) }
    public func history(limit: Int, query: String) async throws -> HistoryListing { try await respond("history(limit:query:)", [limit, query]) }
    public func history(limit: Int) async throws -> HistoryListing { try await respond("history(limit:)", [limit]) }
    public func clearHistory(keeping keep: Int?) async throws { let _: Void = try await respond("clearHistory(keeping:)", [keep]) }
    public func loras(compatibleWith model: String) async throws -> [LoraInfo] { try await respond("loras(compatibleWith:)", [model]) }

    // MARK: MoldDownloadsBackend

    public func startDownload(_ request: DownloadRequest) async throws -> DownloadTicket { try await respond("startDownload(_:)", [request]) }
    public func cancelDownload(id: String) async throws { let _: Void = try await respond("cancelDownload(id:)", [id]) }
    public func downloads() async throws -> DownloadsListing { try await respond("downloads()", []) }

    // MARK: MoldGalleryBackend

    public func gallery(etag: String?) async throws -> Fetched<[GalleryPrint]> { try await respond("gallery(etag:)", [etag]) }
    public func trashedPrints(etag: String?) async throws -> Fetched<[GalleryPrint]> { try await respond("trashedPrints(etag:)", [etag]) }
    public func patch(_ filename: String, with patch: GalleryPatch) async throws { let _: Void = try await respond("patch(_:with:)", [filename, patch]) }
    public func mutate(_ mutation: GalleryBulkMutation) async throws { let _: Void = try await respond("mutate(_:)", [mutation]) }
    public func trash(_ filenames: [String]) async throws { let _: Void = try await respond("trash(_:)", [filenames]) }
    public func restoreFromTrash(_ filenames: [String]) async throws { let _: Void = try await respond("restoreFromTrash(_:)", [filenames]) }
    public func deleteTrashed(_ filenames: [String]) async throws { let _: Void = try await respond("deleteTrashed(_:)", [filenames]) }
    public func deleteForever(_ filenames: [String]) async throws { let _: Void = try await respond("deleteForever(_:)", [filenames]) }
    @discardableResult public func importPrint(_ item: GalleryImport, as filename: String) async throws -> String { try await respond("importPrint(_:as:)", [item, filename]) }
    public func media(_ filename: String, trashed: Bool) async throws -> Data { try await respond("media(_:trashed:)", [filename, trashed]) }
    public func mediaFile(_ filename: String, trashed: Bool) async throws -> URL { try await respond("mediaFile(_:trashed:)", [filename, trashed]) }
    public func thumbnail(_ filename: String, size: Int, trashed: Bool) async throws -> Data { try await respond("thumbnail(_:size:trashed:)", [filename, size, trashed]) }
    public func exportOptions() async throws -> ExportOptions { try await respond("exportOptions()", []) }
    public func export(_ filename: String, format: String) async throws -> Data { try await respond("export(_:format:)", [filename, format]) }
    public func export(_ filename: String, request: MeshExportRequest) async throws -> Data { try await respond("export(_:request:)", [filename, request]) }
    public func playableURL(for filename: String) async throws -> URL { try await respond("playableURL(for:)", [filename]) }

    // MARK: MoldGenerationBackend

    public func placementPreview(_ request: GenerateRequest, copies: Int) async throws -> PlacementPreview { try await respond("placementPreview(_:copies:)", [request, copies]) }
    public func submit(_ admission: BatchAdmission) async throws -> BatchStatus { try await respond("submit(_:)", [admission]) }
    public func batchStatus(id: String) async throws -> BatchStatus { try await respond("batchStatus(id:)", [id]) }
    public func batchStatus(clientBatchId: String) async throws -> BatchStatus { try await respond("batchStatus(clientBatchId:)", [clientBatchId]) }
    public func jobPreview(jobId: String) async throws -> JobProgress? { try await respond("jobPreview(jobId:)", [jobId]) }
    public func cancelBatch(id: String) async throws { let _: Void = try await respond("cancelBatch(id:)", [id]) }

    // MARK: MoldLicencesBackend

    public func licenses() async throws -> [ThirdPartyLicense] { try await respond("licenses()", []) }
    @discardableResult public func acceptLicenses(_ accept: [LicenseAcceptance]) async throws -> [ThirdPartyLicense] { try await respond("acceptLicenses(_:)", [accept]) }

    // MARK: MoldMachinesBackend

    public func devices() async throws -> DeviceState { try await respond("devices()", []) }
    @discardableResult public func setDevice(_ id: String, enabled: Bool) async throws -> DeviceInfo { try await respond("setDevice(_:enabled:)", [id, enabled]) }
    public func resources() async throws -> ResourceSnapshot { try await respond("resources()", []) }
    public func resourceStream() -> AsyncThrowingStream<ResourceSnapshot, Error> { stream("resourceStream()", []) }
    public func peers() async throws -> [DiscoveryPeer] { try await respond("peers()", []) }

    // MARK: MoldModelsBackend

    @discardableResult public func deleteModel(_ model: String) async throws -> ModelRemoval { try await respond("deleteModel(_:)", [model]) }
    public func modelComponents(_ model: String) async throws -> ModelComponentsResponse { try await respond("modelComponents(_:)", [model]) }
    public func loadModel(_ model: String, gpu: Int?) async throws { let _: Void = try await respond("loadModel(_:gpu:)", [model, gpu]) }
    public func unloadModel(model: String?, gpu: Int?) async throws { let _: Void = try await respond("unloadModel(model:gpu:)", [model, gpu]) }

    // MARK: MoldOrganizationBackend

    public func collections() async throws -> [Collection] { try await respond("collections()", []) }
    public func createCollection(name: String, description: String?) async throws -> Collection { try await respond("createCollection(name:description:)", [name, description]) }
    public func updateCollection(id: String, change: CollectionChange) async throws -> Collection { try await respond("updateCollection(id:change:)", [id, change]) }
    public func deleteCollection(id: String) async throws { let _: Void = try await respond("deleteCollection(id:)", [id]) }
    public func tags() async throws -> [TagCount] { try await respond("tags()", []) }
    @discardableResult public func renameTag(_ name: String, to newName: String) async throws -> TagCount { try await respond("renameTag(_:to:)", [name, newName]) }
    public func deleteTag(_ name: String) async throws { let _: Void = try await respond("deleteTag(_:)", [name]) }
    public func emptyTrash() async throws { let _: Void = try await respond("emptyTrash()", []) }

    // MARK: MoldQueueBackend

    public func queue() async throws -> QueueListing { try await respond("queue()", []) }
    public func queueInputs(id: String) async throws -> [QueueInput] { try await respond("queueInputs(id:)", [id]) }
    public func queueInputThumbnail(id: String, index: Int?) async throws -> Data {
        if let index { return try await respond("queueInputThumbnail(id:index:)", [id, index]) }
        return try await queueInputThumbnail(id: id)
    }
    public func queueInputThumbnail(id: String) async throws -> Data { try await respond("queueInputThumbnail(id:)", [id]) }
    public func queueJob(id: String) async throws -> QueueJobDetail { try await respond("queueJob(id:)", [id]) }
    public func cancelJob(id: String) async throws { let _: Void = try await respond("cancelJob(id:)", [id]) }
    public func cancelHeldJob(id: String) async throws -> Bool { try await respond("cancelHeldJob(id:)", [id]) }
    public func pauseJob(id: String) async throws { let _: Void = try await respond("pauseJob(id:)", [id]) }
    public func resumeJob(id: String) async throws { let _: Void = try await respond("resumeJob(id:)", [id]) }
    public func reorderJob(id: String, position: Int) async throws { let _: Void = try await respond("reorderJob(id:position:)", [id, position]) }
    public func retryJob(_ authority: QueueAuthority) async throws { let _: Void = try await respond("retryJob(_:)", [authority]) }
    @discardableResult public func cancelAllQueued() async throws -> QueueCancelResult { try await respond("cancelAllQueued()", []) }
    public func batchStatuses(batchIds: [String]) async throws -> BatchStatusListing { try await respond("batchStatuses(batchIds:)", [batchIds]) }
    public func exportHeldJob(_ authority: QueueAuthority) async throws -> Data { try await respond("exportHeldJob(_:)", [authority]) }
    public func admitTransfer(clientBatchId: String, portable: Data, destinationInstance: String) async throws -> BatchStatus { try await respond("admitTransfer(clientBatchId:portable:destinationInstance:)", [clientBatchId, portable, destinationInstance]) }
    public func completeTransfer(_ authority: QueueAuthority) async throws { let _: Void = try await respond("completeTransfer(_:)", [authority]) }

    // MARK: MoldQueueGateBackend

    @discardableResult public func pauseQueue() async throws -> QueuePauseState { try await respond("pauseQueue()", []) }
    @discardableResult public func resumeQueue() async throws -> QueuePauseState { try await respond("resumeQueue()", []) }

    // MARK: MoldRetainedMediaBackend

    public func retainedSourceMedia(for filename: String) async throws -> RetainedSourceMedia.Inventory { try await respond("retainedSourceMedia(for:)", [filename]) }
    public func retainedSourceMediaBytes(for filename: String, member memberId: String) async throws -> Data { try await respond("retainedSourceMediaBytes(for:member:)", [filename, memberId]) }
    public func retainedMediaReuseSession(for filename: String, members memberIds: [String], target: GenerateRequest) async throws -> RetainedSourceMedia.ReuseSession { try await respond("retainedMediaReuseSession(for:members:target:)", [filename, memberIds, target]) }

    // MARK: MoldStatusBackend

    public func status() async throws -> ServerStatus { try await respond("status()", []) }
    public func capabilities() async throws -> Capabilities { try await respond("capabilities()", []) }
    public func models() async throws -> [Model] { try await respond("models()", []) }

    // MARK: MoldStreamsBackend

    public func events() -> AsyncThrowingStream<MoldEvent, Error> { stream("events()", []) }
    public func batchEvents(id: String) -> AsyncThrowingStream<BatchStatus, Error> { stream("batchEvents(id:)", [id]) }
    public func downloadEvents() -> AsyncThrowingStream<DownloadEvent, Error> { stream("downloadEvents()", []) }

    // MARK: MoldUpscaleBackend

    public func upscaleLibraryImage(filename: String, model: String, tileSize: Int?) async throws -> GalleryImageUpscale { try await respond("upscaleLibraryImage(filename:model:tileSize:)", [filename, model, tileSize]) }
    public func startFramewiseUpscale(filename: String, model: String, tileSize: Int?) async throws -> VideoUpscaleJob { try await respond("startFramewiseUpscale(filename:model:tileSize:)", [filename, model, tileSize]) }
    public func framewiseUpscales() async throws -> [VideoUpscaleJob] { try await respond("framewiseUpscales()", []) }
    public func framewiseUpscale(id: String) async throws -> VideoUpscaleJob { try await respond("framewiseUpscale(id:)", [id]) }
    public func transitionFramewiseUpscale(id: String, to transition: FramewiseTransition) async throws -> VideoUpscaleJob { try await respond("transitionFramewiseUpscale(id:to:)", [id, transition]) }
}

public extension FakeBackend {
    func export(_ filename: String, request: VideoExportRequest) async throws -> Data {
        try await respond("exportVideo(_:request:)", [filename, request])
    }
    func generationAsset(_ filename: String, assetID: String) async throws -> Data {
        try await respond("generationAsset(_:assetID:)", [filename, assetID])
    }
}
