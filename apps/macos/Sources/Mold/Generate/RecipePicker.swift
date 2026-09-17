import MoldClient
import SwiftUI

/// Choosing which of a model's recipes to run -- LTX-2's ten pipelines and
/// `auto`, today; most models have exactly one and this draws nothing.
///
/// A second toolbar `Menu` beside the model picker, not a row inside the
/// capsule: changing the recipe re-reads every control -- steps, guidance,
/// size, the output format, and every capability-gated inspector group
/// (decision 11, M4 design) -- and a control with that blast radius belongs
/// beside the other choice that does the same thing.
struct RecipePicker: View {
    let recipes: [GenerationRecipe]
    let selected: GenerationRecipe?
    let choose: (GenerationRecipe) -> Void

    var body: some View {
        switch RecipePicker.resolve(recipes: recipes, selected: selected) {
        case .hidden:
            EmptyView()
        case let .menu(options, selectedID):
            Menu {
                ForEach(options) { recipe in
                    Button {
                        choose(recipe)
                    } label: {
                        // The recipe's own `label`, verbatim -- the app never
                        // rewrites what the server calls a pipeline.
                        if recipe.id == selectedID {
                            Label(recipe.label, systemImage: "checkmark")
                        } else {
                            Text(recipe.label)
                        }
                    }
                }
            } label: {
                Label(selected?.label ?? "Recipe", systemImage: "list.bullet")
            }
            .labelStyle(.titleAndIcon)
            .fixedSize()
            .help("Choose which of this model's recipes to run")
        }
    }
}

extension RecipePicker {
    /// What the toolbar item shows, resolved purely from the model's recipe
    /// list -- no view needed to test it.
    enum Resolution: Equatable {
        case hidden
        case menu([GenerationRecipe], selected: String?)
    }

    /// A model with one recipe has nothing to pick between; the item does
    /// not draw at all rather than offering a menu with one disabled entry.
    static func resolve(recipes: [GenerationRecipe], selected: GenerationRecipe?) -> Resolution {
        guard recipes.count > 1 else { return .hidden }
        return .menu(recipes, selected: selected?.id)
    }
}
