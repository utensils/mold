import MoldClient
import SwiftUI

/// Form controls each get a whole row. Composer chips cannot share the
/// narrow sheet width at accessibility sizes, and Options has no place here.
struct OptionsControls: View {
    @Environment(GenerateController.self) private var generate
    let recipe: GenerationRecipe

    var body: some View {
        ShapeChip(resolution: recipe.resolution, short: false)
            .accessibilityIdentifier("options-shape")
        if recipe.steps.mode != .fixed {
            StepperChip(title: String(localized: "Steps"), value: generate.draft.steps,
                        range: recipe.steps.min ... recipe.steps.max) { generate.draft.steps = $0 }
                .accessibilityIdentifier("options-steps")
        }
        if generate.kind == .picture {
            StepperChip(title: String(localized: "Batch"), value: generate.draft.batchSize,
                        range: 1 ... generate.referenceBatchLimit) {
                generate.draft.batchSize = $0
            }
            .accessibilityIdentifier("options-batch")
        }
        if let temporal = recipe.temporal {
            LengthChip(temporal: temporal)
                .accessibilityIdentifier("options-length")
        }
    }
}
