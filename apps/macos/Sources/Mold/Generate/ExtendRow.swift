import MoldClient
import MoldStyle
import SwiftUI
import UniformTypeIdentifiers

/// "Start from": an extend continuation, with its own overlap field hidden
/// until somebody reveals it.
///
/// The app never sends an overlap nobody set (decision 12, M4 design) --
/// revealing the field draws a control but writes nothing to the draft by
/// itself; only moving it does, through `RenderDraft.snappedOverlap`.
struct ExtendRow: View {
    let temporal: TemporalProfile?
    @Binding var draft: RenderDraft

    @State private var overlapRevealed = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            MediaWell(
                systemImage: "play.rectangle", allowedTypes: [.movie],
                placeholder: "No clip", attachment: attachmentBinding
            )
            if draft.media.extendVideo != nil { overlapControl }
        }
    }

    @ViewBuilder private var overlapControl: some View {
        if overlapRevealed, let temporal {
            LabeledSection("Overlap") {
                Stepper(value: overlapBinding(temporal: temporal),
                        in: 1...Swift.max(temporal.frames.max - 1, 1),
                        step: Swift.max(temporal.frames.step, 1)) {
                    Text((draft.media.extendOverlapFrames ?? draft.snappedOverlap(1, temporal: temporal)).formatted())
                        .monospacedDigit()
                }
            }
        } else {
            Button("Set overlap…") { overlapRevealed = true }
                .buttonStyle(.plain)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    /// `settingExtend` parks keyframes and the source image the moment a
    /// clip is chosen (`RenderDraft+Clip.swift`); clearing drops the
    /// overlap and the reveal together, so re-adding a continuation starts
    /// from the same "nothing set" state decision 12 asks for.
    private var attachmentBinding: Binding<MediaWell.Attachment?> {
        Binding(
            get: {
                draft.media.extendVideo.map {
                    MediaWell.Attachment(base64: $0, name: draft.media.extendVideoName ?? "")
                }
            },
            set: { newValue in
                if let newValue {
                    draft.media.settingExtend(video: newValue.base64, name: newValue.name)
                } else {
                    draft.media.extendVideo = nil
                    draft.media.extendVideoName = nil
                    draft.media.extendOverlapFrames = nil
                    // Clearing the well is how a person leaves extend mode --
                    // bring back whatever `addingKeyframe` parked on the way
                    // in, the same rule that method applies in reverse.
                    if draft.media.keyframes.isEmpty, !draft.media.parked.keyframes.isEmpty {
                        draft.media.keyframes = draft.media.parked.keyframes
                        draft.media.parked.keyframes = []
                    }
                    overlapRevealed = false
                }
            }
        )
    }

    /// Displayed value falls back to the nearest on-grid overlap so the
    /// Stepper never shows a value it would immediately reject; the draft
    /// itself stays `nil` until this binding's setter actually fires.
    private func overlapBinding(temporal: TemporalProfile) -> Binding<Int> {
        Binding(
            get: { draft.media.extendOverlapFrames ?? draft.snappedOverlap(1, temporal: temporal) },
            set: { draft.media.extendOverlapFrames = draft.snappedOverlap($0, temporal: temporal) }
        )
    }
}
