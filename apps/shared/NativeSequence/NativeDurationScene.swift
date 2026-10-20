import MoldClient
import SwiftUI

struct NativeDurationField: View {
    @Environment(\.dynamicTypeSize) private var typeSize
#if os(iOS)
    @Environment(\.horizontalSizeClass) private var widthClass
#endif
    @FocusState private var editing: Bool
    let title: String
    @Binding var seconds: String
    private var stacked: Bool {
#if os(iOS)
        typeSize.isAccessibilitySize || widthClass == .compact
#else
        typeSize.isAccessibilitySize
#endif
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if stacked { Text(title) }
            HStack {
                if !stacked { Text(title) }
                TextField("Seconds", text: $seconds).textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 120).accessibilityLabel("\(title) in seconds")
                    .focused($editing)
#if os(iOS)
                    .keyboardType(.decimalPad)
#else
                    .focusedValue(\.editingText, editing ? true : nil)
#endif
                Text("seconds").foregroundStyle(.secondaryText)
                Menu {
                    ForEach([5, 10, 15, 30, 60], id: \.self) { value in
                        Button("\(value) seconds") { seconds = String(value) }
                    }
                } label: { Label("Duration presets", systemImage: "chevron.down") }
                .labelStyle(.iconOnly).frame(minWidth: 44, minHeight: 44)
            }.frame(minHeight: 44)
        }
        .accessibilityIdentifier("video-duration")
    }
}

struct NativeDurationScene: View {
    @FocusState private var editing: Bool
    @Binding var scene: VideoDurationDraft.Scene
    let number: Int
    let remove: () -> Void
    let canRemove: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Scene \(number)").font(.headline)
                Spacer(minLength: 8)
                Button("Remove scene \(number)", systemImage: "trash", role: .destructive, action: remove)
                    .labelStyle(.iconOnly).disabled(!canRemove)
                    .frame(minWidth: 44, minHeight: 44)
            }
            TextField("Describe this scene…", text: $scene.prompt, axis: .vertical)
                .lineLimit(2 ... 5).accessibilityLabel("Scene \(number) prompt")
                .focused($editing)
#if os(macOS)
                .focusedValue(\.editingText, editing ? true : nil)
#endif
            NativeDurationField(title: "Scene duration", seconds: $scene.seconds)
        }
        .padding(.vertical, 8)
    }
}

#if os(macOS)
// iOS uses its audited asset-backed token; macOS keeps system secondary text.
extension ShapeStyle where Self == Color {
    static var secondaryText: Color { .secondary }
}
#endif
