import MoldClient
import SwiftUI

struct BoundaryFrameWells: View {
    @Environment(GenerateController.self) private var generate
    @ScaledMetric(relativeTo: .body) private var side: CGFloat = 72

    var body: some View {
        if let caps = generate.recipe?.capabilities, BoundaryFramePolicy.resolve(capabilities: caps) != nil {
            VStack(alignment: .leading, spacing: 8) {
                Text("Opening and closing frames").font(.headline)
                HStack(alignment: .top, spacing: 12) {
                    frame(first: true, caps: caps)
                    frame(first: false, caps: caps)
                }
                if BoundaryFramePolicy.resolve(capabilities: caps) == "wan-pair" {
                    Text("Choose both frames, or leave both empty.").font(.caption).foregroundStyle(.secondaryText)
                }
            }
            .onChange(of: generate.draft.frames) { _, _ in BoundaryFramePolicy.apply(to: &generate.draft, capabilities: caps) }
        }
    }
    private func frame(first: Bool, caps: RecipeCapabilities) -> some View {
        Well(title: first ? "First frame" : "Last frame",
             image: BoundaryFramePolicy.image(first: first, draft: generate.draft, capabilities: caps),
             side: side, accepting: PictureImport.identityReadable,
             set: { BoundaryFramePolicy.set(first: first, picture: $0, draft: &generate.draft, capabilities: caps, recipe: generate.recipe) },
             clear: { BoundaryFramePolicy.set(first: first, picture: nil, draft: &generate.draft, capabilities: caps, recipe: generate.recipe) })
    }
}
