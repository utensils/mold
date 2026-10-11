import MoldClient

/// Preview the same reconciliation used when the user explicitly chooses a model.
struct SourceMediaReuseOffer: Identifiable {
    let recipe: GenerationRecipe
    let choices: [SourceMediaAttachment]
    var id: String { recipe.id }

    static func offers(model: Model, member: RetainedSourceMedia.Member, draft: RenderDraft) -> [Self] {
        guard model.isReady, model.runtimeReady != false else { return [] }
        return (model.generationProfile?.recipes ?? []).compactMap { recipe in
            let recipe = recipe.resolvingReferenceCapabilities(family: model.family, model: model.name)
            let reconciled = draft.adopting(recipe, isNewModel: false, for: model)
            let choices = SourceMediaAttachment.choices(member: member, recipe: recipe, media: reconciled.media)
            return choices.isEmpty ? nil : Self(recipe: recipe, choices: choices)
        }
    }
}

/// The button and right-click menu share this declared action list.
enum SourceMediaReuseAction: Hashable {
    case open, chooseModel, attach(SourceMediaAttachment)

    static func menu(choices: [SourceMediaAttachment], canOpen: Bool, knownKind: Bool, busy: Bool) -> [RowAction<Self>] {
        var result: [RowAction<Self>] = canOpen ? [.init(kind: .open, title: "Open source")] : []
        guard knownKind else { return result }
        if !result.isEmpty { result.append(.separator) }
        result += choices.map { .init(kind: .attach($0), title: $0.title, isDisabled: busy) }
        if !choices.isEmpty { result.append(.separator) }
        result.append(.init(kind: .chooseModel, title: "Choose model…", isDisabled: busy))
        return result
    }
}
