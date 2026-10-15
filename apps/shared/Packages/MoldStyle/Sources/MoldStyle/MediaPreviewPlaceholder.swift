import SwiftUI

/// A quiet, non-error surface occupying the same square as a media preview.
public struct MediaPreviewPlaceholder: View {
    public enum State: Sendable, CaseIterable, Hashable {
        case loading, empty, noStillPreview, unavailable

        public var label: String {
            switch self {
            case .loading: String(localized: "Loading source preview")
            case .empty: String(localized: "No source image")
            case .noStillPreview: String(localized: "No still preview")
            case .unavailable: String(localized: "Source preview unavailable")
            }
        }
    }

    public let state: State
    public init(state: State) { self.state = state }

    public var body: some View {
        RoundedRectangle(cornerRadius: 6)
            .fill(.quaternary.opacity(0.4))
            .overlay {
                RoundedRectangle(cornerRadius: 6)
                    .strokeBorder(.secondary.opacity(0.2), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
            }
            .overlay {
                if state == .loading {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: "photo")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(state.label)
    }
}
