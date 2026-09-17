import Foundation
import MoldClient

/// Applying a machine's stored per-model defaults on top of a recipe's own
/// numbers. Split out of `GenerateController.swift` to stay under the line
/// cap; `applyStoredDefaults` is not `private` because a same-type extension
/// in another file cannot see a `private` member, only `internal`.
@MainActor
extension GenerateController {
    /// Puts a machine's stored per-model defaults on top of the recipe's own
    /// numbers -- but only on a NEW model; `applying` is already a no-op on a
    /// kept draft, and this skips the store read entirely in that case.
    ///
    /// If this host's listing has never been read, nothing is applied yet;
    /// a refresh is kicked off and, once it lands, applied retroactively --
    /// but only if this is STILL the selected model and host by then. A
    /// second model choice made while that refresh was in flight makes its
    /// answer moot, and re-applying it over whatever is now on screen would
    /// silently overwrite a choice made in between.
    func applyStoredDefaults(
        for model: Model, on host: MoldHost.ID, recipe: GenerationRecipe, isNewModel: Bool
    ) {
        guard isNewModel else { return }
        guard defaults.hasLoaded(on: host) else {
            Task { [weak self] in
                await self?.defaults.refresh(on: host)
                guard let self, self.modelName == model.name, self.hostID == host else { return }
                self.draft = self.draft.applying(
                    self.defaults.defaults(for: model.name, on: host), recipe: recipe, isNewModel: true)
            }
            return
        }
        draft = draft.applying(defaults.defaults(for: model.name, on: host), recipe: recipe, isNewModel: true)
    }
}
