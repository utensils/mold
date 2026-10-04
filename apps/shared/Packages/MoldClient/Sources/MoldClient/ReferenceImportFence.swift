import Foundation

/// Async pickers may finish after a model switch or a reordered/removed slot.
/// An import applies only to the media snapshot and destination it was opened for.
public nonisolated struct ReferenceImportFence: Hashable, Sendable {
    private let model: String?
    private let host: UUID?
    private let recipe: GenerationRecipe?
    private let media: DraftMedia
    public init(model: String?, host: UUID?, recipe: GenerationRecipe?, media: DraftMedia) {
        self.model = model; self.host = host; self.recipe = recipe; self.media = media
    }
    public func isCurrent(model: String?, host: UUID?, recipe: GenerationRecipe?, media: DraftMedia) -> Bool {
        self.model == model && self.host == host && self.recipe == recipe && self.media == media
    }
}
