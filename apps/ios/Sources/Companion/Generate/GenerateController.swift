import Foundation
import MoldClient

/// Generate's state (DESIGN.md §5.1): what is being made, on which machine,
/// with which model and recipe, and what the machine is doing about it.
///
/// Every control's range, default and presence comes from the model's own
/// generation profile (`RenderDraft.adopting`); switching model PARKS what the
/// new recipe cannot read rather than dropping it. Generate never turns into
/// Stop: a press while a render runs queues another batch.
@Observable
final class GenerateController {
    let sequence: NativeSequenceSession
    var sequenceLimits: ChainLimits?
    var videoDuration: VideoDurationAuthoring?
    let videoPlanner = VideoPlanController()
    let videoJobs: VideoJobCoordinator
    let videoRestarts: VideoRestartController
    var durationError: String?
    var durationSource: VideoLibraryExtensionSource?
    var durationSourceLoading = false
    var durationSourceError: String?
    var durationSourceToken = UUID()
    @ObservationIgnored var durationSourceRestoration: DurationSourceRestoration?
    @ObservationIgnored var durationTask: Task<Void, Never>?

    let retainedReuse = RetainedReuse()
    private(set) var sourceMediaRevision = 0
    private(set) var mediaRevisions: [RetainedSourceMedia.Field: Int] = [:]
    var draft = RenderDraft() {
        didSet {
            sequence.hasVisibleReferences = !draft.media.generationReferences.isEmpty
            if oldValue.fps != draft.fps, let recipe, let temporal = recipe.temporal {
                let length = temporal.lengthSelection(fps: draft.fps, frames: draft.frames,
                    family: model?.family, model: modelName, sourceImage: recipe.capabilities.sourceImage)
                draft.fps = length.fps
                draft.frames = length.frames
            }
            for field in RetainedSourceMedia.changedDraftFields(from: oldValue.media, to: draft.media) {
                mediaRevisions[field, default: 0] += 1
            }
            if oldValue.media.sourceImage != draft.media.sourceImage
                || oldValue.media.sourceImageOriginal != draft.media.sourceImageOriginal {
                sourceMediaRevision += 1
            }
        }
    }
    /// Still picture, Short clip, 3-D object.
    var kind: PrintKind = .picture
    var referencePreparation: String?
    var modelName: String?
    var recipeID: String?
    /// "Auto" follows the Default machine (or the first that is up and has
    /// the model); pinned is one machine.
    var machine: MachineChoice = .auto { didSet { MachineChoice.save(machine) } }

    var run: RunState = .idle
    /// Batches admitted while another was on screen, followed in turn.
    var queued: [ActiveBatch] = []
    var activeBatch: ActiveBatch?

    @ObservationIgnored let hosts: HostStore
    @ObservationIgnored let ledger: PendingLedger
    @ObservationIgnored let drafts: DraftStore
    @ObservationIgnored var runTask: Task<Void, Never>?
    @ObservationIgnored var followingID: UUID?
    @ObservationIgnored var followingSuspended = false
    @ObservationIgnored var admissions: [UUID: PendingAdmission] = [:]
    @ObservationIgnored var currentAdmissionID: UUID?
    /// Told, synchronously, when a batch reaches its end here -- finished,
    /// failed, or stopped (`.idle`) -- with THAT batch, before the next one
    /// takes the canvas. The Live Activity and notifications hang off it.
    @ObservationIgnored var settled: ((ActiveBatch, RunState) -> Void)?
    /// The model each kind last used, so switching kinds and back restores it.
    @ObservationIgnored var lastModel: [PrintKind: String] = [:]
    /// Cold launch restores before the machines have supplied their profiles.
    @ObservationIgnored var restoringChoice = false

    init(hosts: HostStore, ledger: PendingLedger = .shared,
         drafts: DraftStore = DraftStore(directory: URL.applicationSupportDirectory.appending(path: "io.utensils.mold.companion")),
         initialMachine: MachineChoice? = nil, sequence: NativeSequenceSession = NativeSequenceSession(),
         videoJobs: VideoJobCoordinator = VideoJobCoordinator(),
         videoRestarts: VideoRestartController = VideoRestartController()) {
        self.videoRestarts = videoRestarts
        self.videoJobs = videoJobs
        self.sequence = sequence
        self.hosts = hosts
        self.ledger = ledger
        self.drafts = drafts
        machine = initialMachine ?? MachineChoice.load()
        restoreDraft()
        if sequence.draft != nil { kind = .clip }
        sequence.restoreSource = { [weak self] source, host in
            self?.restoreSequenceRecipe(source, host: host)
        }
        sequence.restoreReferences = { [weak self] refs in
            self?.draft.media.generationReferences = refs.map(AuthoringReference.canonical)
        }
        hosts.listen { [weak self] id, event in
            guard let self, !self.followingSuspended, self.activeBatch?.host == id else { return }
            if case .resyncRequired = event { self.resumeFollowing() }
        }
    }

    // MARK: - What can be chosen

    /// Every installed model on a machine that answers and makes this kind,
    /// merged by name across the fleet, grouped by family.
    var families: [(family: String, models: [Model])] {
        var seen: [String: Model] = [:]
        for host in hosts.upHosts {
            for model in hosts.models[host.id] ?? [] where model.downloaded != false
                && model.runtimeAvailable != false && model.makes.contains(kind) {
                seen[model.name] = seen[model.name] ?? model
            }
        }
        let grouped = Dictionary(grouping: seen.values, by: \.family)
        return grouped.keys.sorted {
            Model.familyPrecedes($0, $1)
        }.map { family in
            (family, grouped[family]!.sorted { $0.name < $1.name })
        }
    }

    var model: Model? {
        guard let modelName else { return nil }
        if let target, let found = hosts.models[target.id]?.first(where: { $0.name == modelName }) { return found }
        for host in hosts.upHosts {
            if let found = hosts.models[host.id]?.first(where: { $0.name == modelName }) { return found }
        }
        return nil
    }

    /// The recipe that will run: the chosen one if this model has it, else
    /// the model's default for this kind.
    var recipe: GenerationRecipe? {
        guard let profile = model?.generationProfile else { return nil }
        let selected: GenerationRecipe?
        if let recipeID, let chosen = profile.recipe(named: recipeID), chosen.makes == kind { selected = chosen }
        else {
            selected = profile.recipes.first { $0.makes == kind && $0.id == profile.defaultRecipeId }
                ?? profile.recipes.first { $0.makes == kind }
        }
        return selected?.resolvingReferenceCapabilities(family: model?.family, model: modelName)
    }

    var referenceBatchLimit: Int {
        guard let host = target else { return 1 }
        let limit = hosts.capabilities[host.id]?.maxBatchOutputs ?? 1
        let request = RenderRequest.one(draft, model: modelName ?? "")
        return max(1, ReferenceUploadPolicy.batchLimit(requests: [request], apiKey: host.apiKey,
            capabilities: hosts.capabilities[host.id]?.referenceUploads, batchLimit: limit))
    }

    /// Recipes this model offers for this kind, when there is a choice.
    var recipes: [GenerationRecipe] {
        model?.generationProfile?.recipes.filter { $0.makes == kind } ?? []
    }

    /// The machine this press of Generate goes to.
    var target: MoldHost? {
        target(requiringNormalization: draft.media.generationReferences.contains { $0.freshVideo != nil })
    }

    /// Picking originals in Auto uses the same eligible-host order as this prepared attempt.
    var referenceImportTarget: MoldHost? { target(requiringNormalization: true) ?? target }

    private func target(requiringNormalization: Bool) -> MoldHost? {
        guard let modelName else { return nil }
        let holds = { (host: MoldHost) in
            self.hosts.models[host.id]?.contains { $0.name == modelName } == true
                && (!requiringNormalization || VideoNormalizationPolicy.capability(
                    self.hosts.capabilities[host.id]?.referenceUploads, apiKey: host.apiKey) != nil)
        }
        switch machine {
        case let .pinned(id):
            return hosts.host(id).flatMap { hosts.isUp($0) && holds($0) ? $0 : nil }
        case .auto:
            if let preferred = hosts.preferredHost, hosts.isUp(preferred), holds(preferred) { return preferred }
            return hosts.upHosts.first(where: holds)
        }
    }

    // MARK: - Choosing

    func setKind(_ new: PrintKind) {
        guard new != kind || restoringChoice else { return }
        if new != .clip { sequence.startFresh() }
        retainedReuse.clear()
        if !restoringChoice, let modelName { lastModel[kind] = modelName }
        restoringChoice = false
        kind = new
        let remembered = lastModel[new].flatMap { name in families.flatMap(\.models).first { $0.name == name } }
        if let pick = remembered ?? families.first?.models.first { choose(pick) } else { modelName = nil }
    }

    func choose(_ model: Model) {
        retainedReuse.clear()
        restoringChoice = false
        let isNew = model.name != modelName
        modelName = model.name
        lastModel[kind] = model.name
        recipeID = nil
        if let recipe { draft = draft.adopting(recipe, isNewModel: isNew, for: model) }
        else { draft.supportsDurationPrediction = false }
    }

    func chooseRecipe(_ id: String) {
        retainedReuse.clear()
        guard let model else { return }
        recipeID = id
        if let recipe { draft = draft.adopting(recipe, isNewModel: false, for: model) }
        else { draft.supportsDurationPrediction = false }
    }

    /// Keep a saved choice while its machine reconnects; only an explicit
    /// choice may replace a draft whose profile has not arrived yet.
    func settleChoice() {
        guard !sequence.restoringSource else { return }
        if let source = sequence.continuation, let host = sequence.continuationHost {
            restoreSequenceRecipe(source, host: host)
            return
        }
        if restoringChoice, let model, let profile = model.generationProfile {
            let restoredRecipe = recipeID.flatMap { profile.recipe(named: $0) }
                ?? profile.recipe(named: profile.defaultRecipeId)
                ?? profile.recipes.first
            if let restoredRecipe {
                kind = sequence.draft != nil ? .clip : restoredRecipe.makes
                lastModel[kind] = model.name
                draft = draft.adopting(restoredRecipe, isNewModel: false, for: model)
            }
            restoringChoice = false
        }
        guard !restoringChoice else { return }
        if model == nil, let first = families.first?.models.first { choose(first) }
    }

    /// Why Generate cannot run right now, in words -- `nil` when it can.
    var blocker: String? {
        if let refusal = durationSourceOpeningRefusal { return refusal }
        if let refusal = retainedReuse.sourcePictureRefusal(in: draft) { return refusal }
        if draft.media.generationReferences.contains(where: { $0.media.authority == "descriptor" }),
           !retainedReuse.probing, !retainedReuse.canHydrateReferences(draft.media.generationReferences) {
            return String(localized: "Reattach this print’s reference media before generating.")
        }
        if retainedReuse.probing { return String(localized: "Restoring the print’s source media…") }
        if retainedReuse.canDiscard {
            let outgoing = modelName.map { RenderRequest.one(draft, model: $0,
                maxIdentityPhotos: target.flatMap { hosts.capabilities[$0.id]?.maxIdentityPhotos } ?? 0) }
            if let refusal = retainedReuse.restorationRefusal(for: outgoing) { return refusal }
        }
        if hosts.hosts.isEmpty { return String(localized: "Add a machine to start generating.") }
        if hosts.upHosts.isEmpty { return String(localized: "No machine is answering. Check Machines to reconnect.") }
        if restoringChoice, model == nil {
            return String(localized: "The saved model is unavailable. Check Machines or choose another model.")
        }
        guard model != nil else { return String(localized: "Install a model that makes this under Models.") }
        guard target != nil else { return String(localized: "No machine that has this model is answering.") }
        if sequence.draft != nil, let modelName, let host = target {
            return sequence.refusal(base: NativeSequenceSession.baseRequest(draft, model: modelName,
                maxIdentityPhotos: hosts.capabilities[host.id]?.maxIdentityPhotos ?? 0), limits: sequenceLimits)
        }
        if usesVideoDuration, let refusal = durationBlocker { return refusal }
        // The draft's own refusal (MoldClient), the Mac's words exactly.
        if sequence.continuation == nil, let recipe {
            var validation = draft
            if usesVideoDuration, let scenes = videoDuration?.draft, scenes.scenesEnabled,
               let first = scenes.scenes.first { validation.prompt = first.prompt }
            if let refusal = validation.refusal(for: recipe, retainedFields: retainedReferenceFields) { return refusal }
        }
        return nil
    }
}
