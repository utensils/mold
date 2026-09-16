import SwiftUI

/// One mark per denoising step.
///
/// Deliberately unanimated. Each segment is a step that actually happened, so
/// it appears when the host reports it and not a frame before. A bar that
/// eased smoothly between reports would be inventing progress the model has
/// not made -- a prettier lie.
struct StepSegments: View {
    let done: Int
    let total: Int

    var body: some View {
        HStack(spacing: 2) {
            ForEach(0..<max(total, 1), id: \.self) { index in
                Rectangle()
                    .fill(index < done ? AnyShapeStyle(.tint) : AnyShapeStyle(.quaternary))
                    .frame(height: 3)
            }
        }
        .accessibilityElement()
        .accessibilityLabel("Step \(done) of \(total)")
    }
}
