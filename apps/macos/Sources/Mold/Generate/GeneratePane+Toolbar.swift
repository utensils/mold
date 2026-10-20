import MoldClient
import SwiftUI

// The pane's toolbar. Split from the pane purely for size, mirroring
// `LibraryPane+Toolbar.swift`.
extension GeneratePane {

    @ToolbarContentBuilder var toolbar: some ToolbarContent {
        ToolbarItem {
            ModelPicker(
                host: host,
                families: host.map { machine in
                    models.families(on: machine.id).compactMap { family in
                        let eligible = family.models.filter { controller.sequence.draft == nil || $0.makes.contains(.clip) }
                        return eligible.isEmpty ? nil : (family: family.family, models: eligible)
                    }
                } ?? [],
                selected: selectedModel
            ) { model in
                if let host { controller.select(model: model, on: host.id) }
            }.disabled(controller.sequence.continuation != nil || controller.sequence.preparing)
        }
        ToolbarItem {
            RecipePicker(
                recipes: (selectedModel?.generationProfile?.recipes ?? []).filter {
                    controller.sequence.draft == nil || $0.makes == .clip
                },
                selected: recipe
            ) { recipe in
                controller.selectRecipe(recipe, in: selectedModel?.generationProfile)
            }.disabled(controller.sequence.continuation != nil || controller.sequence.preparing)
        }
        // No inspector switch here: it belongs over the column it opens,
        // and `trailingColumn` is what knows where that is.
    }
}
