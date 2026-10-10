import MoldClient
import SwiftUI

struct SourceFitSection: View {
    @Environment(GenerateController.self) private var generate
    let recipe: GenerationRecipe

    var body: some View {
        let modes = SourceFitOptions.resolve(recipe: recipe, media: generate.draft.media)
        if !modes.isEmpty {
            Section {
                Picker("Fit", selection: Binding(
                    get: { generate.draft.media.sourceFit.mode },
                    set: { generate.draft.media.sourceFit = SourceFitOptions.policy(
                        for: $0, supportsMask: modes.contains(.padRepaint)) })) {
                    ForEach(modes, id: \.self) { Text($0.label).tag($0) }
                }
                .accessibilityIdentifier("options-source-fit")
                Text(generate.draft.media.sourceFit.mode.help)
                    .font(.caption).foregroundStyle(.secondaryText)
                if case let .cropFill(x, y) = generate.draft.media.sourceFit {
                    Picker("Horizontal position", selection: Binding(
                        get: { x ?? .center },
                        set: { generate.draft.media.sourceFit = .cropFill(alignX: $0, alignY: y ?? .center) })) {
                        Text("Left").tag(SourceFitAlignX.left)
                        Text("Center").tag(SourceFitAlignX.center)
                        Text("Right").tag(SourceFitAlignX.right)
                    }
                    Picker("Vertical position", selection: Binding(
                        get: { y ?? .center },
                        set: { generate.draft.media.sourceFit = .cropFill(alignX: x ?? .center, alignY: $0) })) {
                        Text("Top").tag(SourceFitAlignY.top)
                        Text("Center").tag(SourceFitAlignY.center)
                        Text("Bottom").tag(SourceFitAlignY.bottom)
                    }
                }
            } header: {
                SectionHeader(BoundaryFramePolicy.resolve(capabilities: recipe.capabilities) == nil
                    ? String(localized: "Source image") : String(localized: "Boundary frames"))
            }
        }
    }
}
