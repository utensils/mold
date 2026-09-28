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
    var draft = RenderDraft()
    /// Still picture, Short clip, 3-D object.
    var kind: PrintKind = .picture
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
    /// The model each kind last used, so switching kinds and back restores it.
    @ObservationIgnored var lastModel: [PrintKind: String] = [:]

    init(hosts: HostStore, ledger: PendingLedger = .shared,
         drafts: DraftStore = DraftStore(directory: URL.applicationSupportDirectory.appending(path: "io.utensils.mold.companion"))) {
        self.hosts = hosts
        self.ledger = ledger
        self.drafts = drafts
        machine = MachineChoice.load()
        restoreDraft()
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
        return grouped.keys.sorted().map { family in
            (family, grouped[family]!.sorted { $0.name < $1.name })
        }
    }

    var model: Model? {
        guard let modelName else { return nil }
        for host in hosts.upHosts {
            if let found = hosts.models[host.id]?.first(where: { $0.name == modelName }) { return found }
        }
        return nil
    }

    /// The recipe that will run: the chosen one if this model has it, else
    /// the model's default for this kind.
    var recipe: GenerationRecipe? {
        guard let profile = model?.generationProfile else { return nil }
        if let recipeID, let chosen = profile.recipe(named: recipeID), chosen.makes == kind { return chosen }
        return profile.recipes.first { $0.makes == kind && $0.id == profile.defaultRecipeId }
            ?? profile.recipes.first { $0.makes == kind }
    }

    /// Recipes this model offers for this kind, when there is a choice.
    var recipes: [GenerationRecipe] {
        model?.generationProfile?.recipes.filter { $0.makes == kind } ?? []
    }

    /// The machine this press of Generate goes to.
    var target: MoldHost? {
        guard let modelName else { return nil }
        let holds = { (host: MoldHost) in self.hosts.models[host.id]?.contains { $0.name == modelName } == true }
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
        guard new != kind else { return }
        if let modelName { lastModel[kind] = modelName }
        kind = new
        let remembered = lastModel[new].flatMap { name in families.flatMap(\.models).first { $0.name == name } }
        if let pick = remembered ?? families.first?.models.first { choose(pick) } else { modelName = nil }
    }

    func choose(_ model: Model) {
        let isNew = model.name != modelName
        modelName = model.name
        lastModel[kind] = model.name
        recipeID = nil
        if let recipe { draft = draft.adopting(recipe, isNewModel: isNew, for: model) }
    }

    func chooseRecipe(_ id: String) {
        guard let model else { return }
        recipeID = id
        if let recipe { draft = draft.adopting(recipe, isNewModel: false, for: model) }
    }

    /// A model that has vanished (uninstalled, machine gone) gives way to the
    /// first one that can make this kind, rather than an empty menu.
    func settleChoice() {
        if model == nil, let first = families.first?.models.first { choose(first) }
    }

    /// Why Generate cannot run right now, in words -- `nil` when it can.
    var blocker: String? {
        if hosts.hosts.isEmpty { return String(localized: "Add a machine to start generating.") }
        guard model != nil else { return String(localized: "Install a model that makes this under Models.") }
        guard target != nil else { return String(localized: "No machine that has this model is answering.") }
        // The draft's own refusal (MoldClient), the Mac's words exactly.
        if let recipe, let refusal = draft.refusal(for: recipe) { return refusal }
        return nil
    }
}
