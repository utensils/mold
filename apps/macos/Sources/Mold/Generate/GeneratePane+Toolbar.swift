import MoldClient
import SwiftUI

// The pane's toolbar. Split from the pane purely for size, mirroring
// `LibraryPane+Toolbar.swift`.
extension GeneratePane {

    @ToolbarContentBuilder var toolbar: some ToolbarContent {
        ToolbarItem {
            ModelPicker(
                host: host,
                families: host.map { models.families(on: $0.id) } ?? [],
                selected: selectedModel
            ) { model in
                if let host { controller.select(model: model, on: host.id) }
            }
        }
        ToolbarItem {
            RecipePicker(
                recipes: selectedModel?.generationProfile?.recipes ?? [],
                selected: recipe
            ) { recipe in
                controller.selectRecipe(recipe)
            }
        }
        ToolbarItem {
            Button { showsInspector.toggle() } label: {
                Label("Inspector", systemImage: "sidebar.trailing")
            }
            .help(showsInspector ? "Hide the inspector" : "Show the inspector")
        }
    }
}
