import MoldClient
import MoldStyle
import SwiftUI

/// The controls a recipe actually has.
///
/// Nothing here is conditional on a model name or a family. The server says
/// what each control may be and this renders that answer -- which is why a
/// model added to mold tomorrow gets correct controls with no change here.
struct ControlsRow: View {
    let recipe: GenerationRecipe
    @Binding var draft: RenderDraft

    var body: some View {
        HStack(alignment: .bottom, spacing: 18) {
            if recipe.resolution.hasCanvas {
                ControlLabel("Size") { SizeMenu(resolution: recipe.resolution, draft: $draft) }
            }
            if recipe.steps.hasSomethingToShow {
                ControlLabel("Steps") { steps }
            }
            if recipe.guidance.hasSomethingToShow {
                ControlLabel("Guidance") { guidance }
            }
            ControlLabel("Seed") { SeedControl(draft: $draft) }
            Spacer(minLength: 0)
        }
    }

    @ViewBuilder private var steps: some View {
        if recipe.steps.mode.isAdjustable {
            HStack(spacing: 6) {
                Slider(
                    value: Binding(
                        get: { Double(draft.steps) },
                        set: { draft.steps = Int($0.rounded()) }
                    ),
                    in: Double(recipe.steps.min)...Double(recipe.steps.max),
                    step: Double(recipe.steps.step)
                )
                .controlSize(.small)
                .frame(minWidth: 80, maxWidth: 130)
                Text(draft.steps.formatted())
                    .monospacedDigit()
                    .frame(minWidth: 22, alignment: .trailing)
            }
        } else if let note = recipe.steps.note {
            // A pinned control still has something to say. The note is the
            // server's own words and is shown as written.
            Text(note).font(.caption).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private var guidance: some View {
        if recipe.guidance.mode.isAdjustable {
            HStack(spacing: 6) {
                Slider(value: $draft.guidance,
                       in: recipe.guidance.min...recipe.guidance.max,
                       step: recipe.guidance.step)
                    .controlSize(.small)
                    .frame(minWidth: 80, maxWidth: 130)
                Text(draft.guidance, format: .number.precision(.fractionLength(1)))
                    .monospacedDigit()
                    .frame(minWidth: 28, alignment: .trailing)
            }
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
