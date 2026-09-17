import MoldClient
import MoldStyle
import SwiftUI

/// The controls a recipe actually has, wrapping onto as many lines as they
/// need. The actions (the hint, Stop, Generate) are NOT among them: they
/// take a row of their own under this one (M8 decision 10), so Generate
/// never moves when a control appears, a run starts, or the hint changes.
///
/// Nothing here is conditional on a model name or a family. The server says
/// what each control may be and this renders that answer -- which is why a
/// model added to mold tomorrow gets correct controls with no change here.
struct ControlsRow: View {
    let recipe: GenerationRecipe
    let maxBatch: Int
    @Binding var draft: RenderDraft

    var body: some View {
        WrappingHStack(horizontalSpacing: 18, verticalSpacing: 10) {
            ControlLabel("Machine") { MachineControl() }
            shapeControl
            if let temporal = recipe.temporal {
                ControlLabel("Length") { LengthControl(temporal: temporal, draft: $draft) }
            }
            if recipe.steps.hasSomethingToShow {
                ControlLabel("Steps") { steps }
            }
            if recipe.guidance.hasSomethingToShow {
                ControlLabel("Guidance") { guidance }
            }
            // The conditioning the REQUEST will carry, not whatever the well
            // is holding: an exclusive recipe parks one of the two, and
            // strength applies only to a source image that actually ships.
            if draft.media.requestConditioning.carriesSource,
               recipe.capabilities.supportsStrength == true {
                ControlLabel("Strength") {
                    SliderControl(value: $draft.strength, range: 0...1, step: 0.05) {
                        Text(draft.strength, format: .number.precision(.fractionLength(2)))
                    }
                }
            }
            ControlLabel("Seed") { SeedControl(draft: $draft) }
            ControlLabel("Batch") { BatchControl(maximum: maxBatch, draft: $draft) }
        }
    }

    /// `ShapeControl.resolve` decides whether there is anything to show at
    /// all -- a recipe with no canvas (a text-only model, say) draws nothing
    /// rather than an empty labelled slot.
    @ViewBuilder private var shapeControl: some View {
        if case .hidden = ShapeControl.resolve(resolution: recipe.resolution, width: draft.width, height: draft.height) {
            EmptyView()
        } else {
            ControlLabel("Shape") { ShapeControl(resolution: recipe.resolution, draft: $draft) }
        }
    }

    @ViewBuilder private var steps: some View {
        if recipe.steps.mode.isAdjustable {
            StepsControl(control: recipe.steps, draft: $draft)
        } else if let note = recipe.steps.note {
            // A pinned control still has something to say. The note is the
            // server's own words and is shown as written.
            Text(note).font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private var guidance: some View {
        if recipe.guidance.mode.isAdjustable {
            GuidanceControl(control: recipe.guidance, draft: $draft)
        } else if let note = recipe.guidance.note {
            Text(note).font(.caption).foregroundStyle(.secondary)
        }
    }
}

/// A caption over a control, on one baseline so a row of mixed controls lines
/// up whatever each one is.
struct ControlLabel<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption).foregroundStyle(.tertiary)
            content.frame(height: Chrome.fieldHeight)
        }
    }
}
