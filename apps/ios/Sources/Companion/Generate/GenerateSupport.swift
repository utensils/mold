import Foundation
import MoldClient

/// Which machine Generate uses: "Auto" (the Default, else the first that is
/// up and has the model), or one machine by id. Remembered across launches.
enum MachineChoice: Hashable, Codable {
    case auto
    case pinned(MoldHost.ID)

    private static let key = "generate.machine"

    static func load() -> MachineChoice {
        guard let data = UserDefaults.standard.data(forKey: key),
              let choice = try? JSONDecoder().decode(MachineChoice.self, from: data) else { return .auto }
        return choice
    }

    static func save(_ choice: MachineChoice) {
        UserDefaults.standard.set(try? JSONEncoder().encode(choice), forKey: key)
    }
}

/// Every batch a machine has admitted and the app has not yet seen settle, in
/// the App Group: the background refresh reads it to notify, the Live
/// Activity to update, and the app to follow again after it was away.
final class PendingLedger: @unchecked Sendable {
    static let shared = PendingLedger(url: (AppGroup.container
        ?? URL.applicationSupportDirectory).appending(path: "pending-batches.json"))

    let url: URL
    private let lock = NSLock()

    init(url: URL) { self.url = url }

    var batches: [ActiveBatch] {
        lock.lock(); defer { lock.unlock() }
        return read()
    }

    func add(_ batch: ActiveBatch) {
        update { list in list.removeAll { $0.clientBatchId == batch.clientBatchId }; list.append(batch) }
    }

    func remove(_ clientBatchId: String) {
        update { $0.removeAll { $0.clientBatchId == clientBatchId } }
    }

    private func update(_ change: (inout [ActiveBatch]) -> Void) {
        lock.lock(); defer { lock.unlock() }
        var list = read()
        change(&list)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? JSONEncoder().encode(list).write(to: url, options: .atomic)
    }

    private func read() -> [ActiveBatch] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        return (try? JSONDecoder().decode([ActiveBatch].self, from: data)) ?? []
    }
}

extension GenerateController {
    /// The draft survives the app being terminated (DESIGN.md §4): written on
    /// every Generate and when the app goes to the background.
    func saveDraft() {
        drafts.save(DraftDescriptor(draft, model: modelName, family: model?.family, recipeID: recipe?.id ?? recipeID))
    }

    func restoreDraft() {
        guard let saved = drafts.load() else { return }
        saved.apply(to: &draft)
        modelName = saved.model
        recipeID = saved.recipeID
        restoringChoice = saved.model != nil
        // Kind comes from the saved recipe once its profile arrives, without
        // adopting defaults over the user's restored draft.
        if model != nil { settleChoice() }
    }

    /// "Use These Settings": the print's whole recipe back in the composer,
    /// on the machine that made it. A model this fleet no longer has leaves
    /// the choice alone and says so through `blocker`.
    func reuse(_ entry: LibraryEntry) {
        var reused = RenderDraft(reusing: entry.print.metadata)
        reused.media = draft.media
        draft = reused
        if let name = entry.print.metadata.model {
            modelName = name
            recipeID = nil
            if let made = model?.makes.first { kind = made }
            if let recipe, let model { draft = draft.adopting(recipe, isNewModel: false, for: model) }
        }
        machine = .pinned(entry.hostID)
    }
}

extension GenerateController {
    /// More options' Reset: the recipe's own defaults back, the prompt and
    /// every picture kept -- a reset is not a way to lose work.
    func resetOptions() {
        guard let recipe, let model else { return }
        var fresh = RenderDraft()
        fresh.prompt = draft.prompt
        fresh.media = draft.media
        fresh.media.sourceFit = .default
        fresh.title = draft.title
        fresh.tags = draft.tags
        fresh.collectionName = draft.collectionName
        draft = fresh.adopting(recipe, isNewModel: true, for: model)
    }
}
