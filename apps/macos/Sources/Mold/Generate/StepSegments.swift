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
        .accessibilityLabel(Self.reading(done: done, total: total).label)
        .accessibilityValue(Self.reading(done: done, total: total).value)
    }

    /// Pure, so VoiceOver's two strings are pinned with no view rendered.
    /// Split from the label the old single-string version folded them into
    /// (design S7): a value re-announces as the strip moves, where a label
    /// alone would be re-read as an unchanged name.
    static func reading(done: Int, total: Int) -> (label: String, value: String) {
        ("Progress", "Step \(done) of \(total)")
    }
}
